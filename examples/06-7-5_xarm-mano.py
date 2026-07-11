"""xArm MANO Online IK Server

Serve warm-started xArm online IK from live MANO keypoints. The webpolicy
server expects observations with a `kp3d` array shaped `(21, 3)` and returns the
first joint configuration from the solved short horizon.
"""

import argparse
from pathlib import Path
import time
from typing import Any

import jaxlie
import numpy as np
import pyroki as pk
import viser
import yourdfpy
from pyroki.collision import HalfSpace, RobotCollision
from scipy.spatial.transform import Rotation as R
from viser.extras import ViserUrdf
from webpolicy.base_policy import BasePolicy
from webpolicy.server import Server

import pyroki_snippets as pks


PALM_KEYPOINT = 0
THUMB_TIP_KEYPOINT = 4
INDEX_TIP_KEYPOINT = 8
TARGET_LINK_NAME = "link_tcp"
XARM_ARM_WARMSTART_DEG = np.array([0.0, -45.0, 0.0, 35.0, 0.0, 65.0, 90.0])


def load_xarm_urdf(robot_urdf_path: Path) -> yourdfpy.URDF:
    """Load the xArm URDF while resolving relative mesh paths."""
    base_path = robot_urdf_path.parent

    def filename_handler(fname: str) -> str:
        return yourdfpy.filename_handler_magic(fname, dir=base_path)

    return yourdfpy.URDF.load(robot_urdf_path, filename_handler=filename_handler)


def xarm_default_cfg(urdf: yourdfpy.URDF) -> np.ndarray:
    """Return xArm default config with the requested arm posture."""
    default_robot = pk.Robot.from_urdf(urdf)
    default_cfg = np.array(default_robot.joint_var_cls.default_factory())
    default_cfg[:7] = np.deg2rad(XARM_ARM_WARMSTART_DEG)
    return default_cfg


def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-6)


def construct_gripper_axes(
    left_pos: np.ndarray,
    right_pos: np.ndarray,
    midpoint_pos: np.ndarray,
    palm_or_eef_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the same lateral and approach-plane axes used by 09-7_xarm.py."""
    line_a = normalize(right_pos - left_pos)
    line_b = normalize(midpoint_pos - palm_or_eef_pos)
    line_c = normalize(np.cross(line_a, line_b))
    line_d = normalize(np.cross(line_a, line_c))
    return line_a, line_d


def axes_frame(line_a: np.ndarray, line_d: np.ndarray) -> np.ndarray:
    """Return an orthonormal frame whose first and second columns are A and D."""
    x_axis = normalize(line_a)
    y_axis = normalize(line_d - x_axis * np.dot(line_d, x_axis))
    z_axis = normalize(np.cross(x_axis, y_axis))
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def mano_keypoints_to_target_pose(
    kp3d: np.ndarray,
    tcp_frame_from_axes_frame: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert MANO keypoints into a target TCP position and scalar-first quat."""
    palm_target = kp3d[PALM_KEYPOINT]
    left_target = kp3d[THUMB_TIP_KEYPOINT]
    right_target = kp3d[INDEX_TIP_KEYPOINT]
    tcp_target = (left_target + right_target) / 2.0

    target_a_axis, target_d_axis = construct_gripper_axes(
        left_target,
        right_target,
        tcp_target,
        palm_target,
    )
    target_axes_frame = axes_frame(target_a_axis, target_d_axis)
    target_tcp_frame = target_axes_frame @ tcp_frame_from_axes_frame
    target_wxyz = R.from_matrix(target_tcp_frame).as_quat(scalar_first=True)
    aperture = float(np.linalg.norm(right_target - left_target))
    return tcp_target, target_wxyz, aperture


def get_tcp_frame_from_axes_frame(
    robot: pk.Robot,
    link_indices: dict[str, int],
    cfg: np.ndarray,
) -> np.ndarray:
    """Calibrate how 09-style gripper axes map to the link_tcp frame."""
    T_world_link = jaxlie.SE3(robot.forward_kinematics(cfg=cfg))
    link_pos = np.array(T_world_link.translation())
    link_wxyz_xyz = np.array(T_world_link.wxyz_xyz)

    left_tip_pos = link_pos[link_indices["left_tip"]]
    right_tip_pos = link_pos[link_indices["right_tip"]]
    tcp_pos = link_pos[link_indices["tcp"]]
    eef_pos = link_pos[link_indices["eef"]]
    robot_a_axis, robot_d_axis = construct_gripper_axes(
        left_tip_pos,
        right_tip_pos,
        tcp_pos,
        eef_pos,
    )
    robot_axes_frame = axes_frame(robot_a_axis, robot_d_axis)
    tcp_frame = R.from_quat(
        link_wxyz_xyz[link_indices["tcp"], :4],
        scalar_first=True,
    ).as_matrix()
    return robot_axes_frame.T @ tcp_frame


class XarmManoPolicy(BasePolicy):
    """webpolicy policy that maps MANO keypoints to xArm joint commands."""

    def __init__(
        self,
        urdf: yourdfpy.URDF,
        robot: pk.Robot,
        robot_coll: RobotCollision,
        len_traj: int,
        dt: float,
        enable_viser: bool,
        viser_port: int,
    ) -> None:
        self.urdf = urdf
        self.robot = robot
        self.robot_coll = robot_coll
        self.len_traj = len_traj
        self.dt = dt
        self.link_indices = self._get_link_indices()
        self.plane_coll = HalfSpace.from_point_and_normal(
            np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
        )
        self.world_coll = [self.plane_coll]
        self.sol_traj = np.array(robot.joint_var_cls.default_factory())[None].repeat(
            len_traj, axis=0
        )
        self.tcp_frame_from_axes_frame = get_tcp_frame_from_axes_frame(
            robot,
            self.link_indices,
            self.sol_traj[0],
        )
        self.server = None
        self.urdf_vis = None
        self.target_frame = None
        self.keypoint_handle = None
        self.planned_frame_handle = None
        self.timing_handle = None
        if enable_viser:
            self._setup_viser(viser_port)

    def _get_link_indices(self) -> dict[str, int]:
        link_names = {
            "eef": "xarm_gripper_base_link",
            "left_tip": "left_tip",
            "right_tip": "right_tip",
            "tcp": TARGET_LINK_NAME,
        }
        missing = [name for name in link_names.values() if name not in self.robot.links.names]
        if missing:
            raise ValueError(f"xArm link(s) missing from URDF: {missing}")
        return {key: self.robot.links.names.index(name) for key, name in link_names.items()}

    def _setup_viser(self, viser_port: int) -> None:
        self.server = viser.ViserServer(port=viser_port)
        self.server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
        self.urdf_vis = ViserUrdf(self.server, self.urdf, root_node_name="/robot")
        self.target_frame = self.server.scene.add_frame(
            "/mano_target_tcp",
            axes_length=0.08,
            axes_radius=0.004,
        )
        self.keypoint_handle = self.server.scene.add_point_cloud(
            "/mano_keypoints",
            np.zeros((21, 3)),
            np.array([[80, 180, 255]] * 21, dtype=np.uint8),
            point_size=0.008,
            point_shape="circle",
        )
        self.planned_frame_handle = self.server.scene.add_batched_axes(
            "/planned_tcp_frames",
            axes_length=0.04,
            axes_radius=0.002,
            batched_positions=np.zeros((self.len_traj, 3)),
            batched_wxyzs=np.array([[1.0, 0.0, 0.0, 0.0]] * self.len_traj),
        )
        self.timing_handle = self.server.gui.add_number(
            "Elapsed (ms)", 0.001, disabled=True
        )
        self.urdf_vis.update_cfg(self.sol_traj[0])

    def reset(self, payload: dict | None = None) -> None:
        self.sol_traj = np.array(self.robot.joint_var_cls.default_factory())[None].repeat(
            self.len_traj,
            axis=0,
        )
        if self.urdf_vis is not None:
            self.urdf_vis.update_cfg(self.sol_traj[0])

    def step(self, obs: dict) -> dict:
        start_time = time.time()
        kp3d = self._extract_kp3d(obs)
        target_position, target_wxyz, aperture = mano_keypoints_to_target_pose(
            kp3d,
            self.tcp_frame_from_axes_frame,
        )
        self.sol_traj, sol_pos, sol_wxyz = pks.solve_online_planning(
            robot=self.robot,
            robot_coll=self.robot_coll,
            world_coll=self.world_coll,
            target_link_name=TARGET_LINK_NAME,
            target_position=target_position,
            target_wxyz=target_wxyz,
            timesteps=self.len_traj,
            dt=self.dt,
            start_cfg=self.sol_traj[0],
            prev_sols=self.sol_traj,
        )
        elapsed_ms = (time.time() - start_time) * 1000.0
        self._update_viser(kp3d, target_position, target_wxyz, sol_pos, sol_wxyz, elapsed_ms)

        q = self.sol_traj[0]
        return {
            "q": q,
            "target_position": target_position,
            "target_wxyz": target_wxyz,
            "aperture": aperture,
            "elapsed_ms": elapsed_ms,
        }

    def _extract_kp3d(self, obs: dict[str, Any]) -> np.ndarray:
        kp3d = obs.get("kp3d", obs.get("mano_kp3d", obs.get("keypoints")))
        if kp3d is None:
            raise ValueError("Observation must contain `kp3d`, `mano_kp3d`, or `keypoints`.")
        kp3d = np.asarray(kp3d, dtype=np.float64)
        if kp3d.shape == (1, 21, 3):
            kp3d = kp3d[0]
        if kp3d.shape != (21, 3):
            raise ValueError(f"Expected MANO keypoints with shape (21, 3), got {kp3d.shape}.")
        if not np.isfinite(kp3d).all():
            raise ValueError("MANO keypoints contain NaN or inf values.")
        return kp3d

    def _update_viser(
        self,
        kp3d: np.ndarray,
        target_position: np.ndarray,
        target_wxyz: np.ndarray,
        sol_pos: np.ndarray,
        sol_wxyz: np.ndarray,
        elapsed_ms: float,
    ) -> None:
        if self.server is None:
            return
        assert self.urdf_vis is not None and self.target_frame is not None
        with self.server.atomic():
            self.urdf_vis.update_cfg(self.sol_traj[0])
            self.target_frame.position = target_position
            self.target_frame.wxyz = target_wxyz
            if self.keypoint_handle is not None:
                self.keypoint_handle.points = kp3d
            if self.planned_frame_handle is not None:
                planned_positions = np.array(sol_pos).reshape((-1, 3))
                planned_wxyzs = np.array(sol_wxyz).reshape((-1, 4))
                if hasattr(self.planned_frame_handle, "batched_positions"):
                    self.planned_frame_handle.batched_positions = planned_positions
                    self.planned_frame_handle.batched_wxyzs = planned_wxyzs
                else:
                    self.planned_frame_handle.positions_batched = planned_positions
                    self.planned_frame_handle.wxyzs_batched = planned_wxyzs
            if self.timing_handle is not None:
                self.timing_handle.value = elapsed_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8085)
    parser.add_argument("--viser-port", type=int, default=8080)
    parser.add_argument("--no-viser", action="store_true")
    parser.add_argument("--len-traj", type=int, default=5)
    parser.add_argument("--dt", type=float, default=0.1)
    args = parser.parse_args()

    asset_dir = Path(__file__).parent / "retarget_helpers" / "hand"
    robot_urdf_path = asset_dir / "xarm" / "xarm7_standalone.urdf"
    try:
        urdf = load_xarm_urdf(robot_urdf_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Expected xArm assets at `examples/retarget_helpers/hand/xarm`."
        ) from exc

    robot = pk.Robot.from_urdf(urdf, default_joint_cfg=xarm_default_cfg(urdf))
    robot_coll = RobotCollision.from_urdf(urdf)
    if TARGET_LINK_NAME not in robot.links.names:
        raise ValueError(f"xArm target link '{TARGET_LINK_NAME}' is not present.")

    policy = XarmManoPolicy(
        urdf=urdf,
        robot=robot,
        robot_coll=robot_coll,
        len_traj=args.len_traj,
        dt=args.dt,
        enable_viser=not args.no_viser,
        viser_port=args.viser_port,
    )
    metadata = {
        "policy": "xarm_mano_online_ik",
        "target_link": TARGET_LINK_NAME,
        "obs": {"kp3d": [21, 3]},
        "action": {"q": [robot.joints.num_actuated_joints]},
    }
    Server(policy, host=args.host, port=args.port, metadata=metadata).serve()


if __name__ == "__main__":
    main()
