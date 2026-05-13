from pathlib import Path

import cv2
import imageio
import numpy as np
import plotly.graph_objects as go


LINKS_M = np.array([0.084, 0.084, 0.084, 0.190, 0.040], dtype=np.float64)
SERVO_MIN = np.zeros(6, dtype=np.float64)
SERVO_MAX = np.full(6, 180.0, dtype=np.float64)
HOME_DEG = np.array([90, 90, 90, 90, 90, 0], dtype=np.float64)


def clamp_servo_deg(angles_deg: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(angles_deg, dtype=np.float64), SERVO_MIN, SERVO_MAX)


def gripper_open_mm(gripper_deg: float) -> float:
    close_ratio = np.clip(gripper_deg / 180.0, 0.0, 1.0)
    return 100.0 * (1.0 - close_ratio)


def fk_servo(angles_deg: np.ndarray, base_mm: np.ndarray):
    a = clamp_servo_deg(angles_deg)
    base_yaw = np.deg2rad(180.0 - a[0])
    p1 = np.deg2rad(90.0 - a[1])
    p2 = np.deg2rad(90.0 - a[2])
    p3 = np.deg2rad(90.0 - a[3])
    wrist_side = np.deg2rad(a[4] - 90.0)

    origin = base_mm.astype(np.float64) / 1000.0
    theta1 = p1
    theta2 = p1 + p2
    theta3 = p1 + p2 + p3

    pts_local = [np.array([0.0, 0.0, 0.0], dtype=np.float64)]
    x, z = 0.0, LINKS_M[0]
    pts_local.append(np.array([x, 0.0, z], dtype=np.float64))
    for link_length, theta in zip(LINKS_M[1:4], [theta1, theta2, theta3]):
        x += link_length * np.sin(theta)
        z += link_length * np.cos(theta)
        pts_local.append(np.array([x, 0.0, z], dtype=np.float64))

    tool_dir = np.array(
        [np.sin(theta3) * np.cos(wrist_side), np.sin(wrist_side), np.cos(theta3) * np.cos(wrist_side)],
        dtype=np.float64,
    )
    tcp_local = pts_local[-1] + LINKS_M[4] * tool_dir
    pts_local.append(tcp_local)

    c, s = np.cos(base_yaw), np.sin(base_yaw)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    pts_world = np.asarray([origin + rz @ p for p in pts_local], dtype=np.float64)

    grip_width_m = gripper_open_mm(a[5]) / 1000.0
    side_dir_local = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    finger_center = tcp_local - 0.012 * tool_dir
    finger_a = finger_center + 0.5 * grip_width_m * side_dir_local
    finger_b = finger_center - 0.5 * grip_width_m * side_dir_local
    fingers_world = np.asarray([origin + rz @ finger_a, origin + rz @ finger_b], dtype=np.float64)
    return pts_world, fingers_world


def solve_servo_from_target(target_mm: np.ndarray, base_mm: np.ndarray) -> np.ndarray:
    rel = (target_mm - base_mm).astype(np.float64)
    tx, ty, tz = rel
    yaw_deg = 180.0 - np.rad2deg(np.arctan2(ty, tx + 1e-9))
    yaw_deg = float(np.clip(yaw_deg, 0.0, 180.0))

    r = np.hypot(tx, ty)
    z = tz - LINKS_M[0] * 1000.0
    l1 = LINKS_M[1] * 1000.0
    l2 = LINKS_M[2] * 1000.0
    l3 = (LINKS_M[3] + LINKS_M[4]) * 1000.0
    d = np.hypot(r, z)
    max_reach = l1 + l2 + l3 - 1e-6
    if d > max_reach:
        scale = max_reach / (d + 1e-9)
        r *= scale
        z *= scale
        d = max_reach

    phi = np.arctan2(z, r + 1e-9)
    d_eff = max(d - 0.35 * l3, 1e-6)
    cos_elbow = (l1 * l1 + l2 * l2 - d_eff * d_eff) / (2.0 * l1 * l2 + 1e-9)
    cos_elbow = float(np.clip(cos_elbow, -1.0, 1.0))
    elbow_inner = np.arccos(cos_elbow)
    shoulder_aux = np.arctan2(l2 * np.sin(elbow_inner), l1 + l2 * np.cos(elbow_inner))

    shoulder_abs = phi + shoulder_aux
    elbow_abs = shoulder_abs - (np.pi - elbow_inner)
    wrist_abs = elbow_abs - 0.55 * elbow_abs

    motors = np.array(
        [
            yaw_deg,
            90.0 - np.rad2deg(shoulder_abs),
            90.0 - np.rad2deg(elbow_abs - shoulder_abs),
            90.0 - np.rad2deg(wrist_abs - elbow_abs),
            90.0,
            180.0,
        ],
        dtype=np.float64,
    )
    return clamp_servo_deg(motors)


def build_visuals(motor_deg: np.ndarray, base_mm: np.ndarray, gif_path: Path, scene_objects=None, title: str = "Sketch-Matched Servo Simulator"):
    motor_deg = clamp_servo_deg(motor_deg)
    base_mm = np.asarray(base_mm, dtype=np.float64)
    traj_deg = np.linspace(HOME_DEG, motor_deg, 56)
    traj_pose = [fk_servo(angles, base_mm) for angles in traj_deg]
    traj_pts_mm = [pose[0] * 1000.0 for pose in traj_pose]
    traj_fingers_mm = [pose[1] * 1000.0 for pose in traj_pose]

    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    x_min, x_max = -350.0, 350.0
    y_min, y_max = -350.0, 350.0

    def to_px(pxy):
        u = int((pxy[0] - x_min) / (x_max - x_min) * 580 + 30)
        v = int((1.0 - (pxy[1] - y_min) / (y_max - y_min)) * 580 + 30)
        return (u, v)

    for frame_idx, (pts, fingers) in enumerate(zip(traj_pts_mm, traj_fingers_mm), start=1):
        canvas = np.full((640, 640, 3), 245, dtype=np.uint8)
        pix = [to_px(p) for p in pts[:, :2]]
        for joint_idx in range(len(pix) - 1):
            cv2.line(canvas, pix[joint_idx], pix[joint_idx + 1], (60, 130, 255), max(2, 8 - joint_idx), cv2.LINE_AA)
            cv2.circle(canvas, pix[joint_idx], 4, (20, 20, 20), -1, cv2.LINE_AA)
        finger_pix = [to_px(p[:2]) for p in fingers]
        cv2.line(canvas, finger_pix[0], finger_pix[1], (40, 140, 220), 4, cv2.LINE_AA)
        cv2.circle(canvas, pix[-1], 6, (50, 180, 80), -1, cv2.LINE_AA)
        cv2.putText(canvas, f"frame {frame_idx}/{len(traj_deg)}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"m0={traj_deg[frame_idx - 1, 0]:.1f} m1={traj_deg[frame_idx - 1, 1]:.1f} m2={traj_deg[frame_idx - 1, 2]:.1f}", (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"m3={traj_deg[frame_idx - 1, 3]:.1f} m4={traj_deg[frame_idx - 1, 4]:.1f} grip={traj_deg[frame_idx - 1, 5]:.1f}", (16, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
        frames.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    imageio.mimsave(gif_path, frames, duration=0.03)

    final_pts = traj_pts_mm[-1]
    final_fingers = traj_fingers_mm[-1]
    tcp_path = np.asarray([pts[-1] for pts in traj_pts_mm])
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=traj_pts_mm[0][:, 0], y=traj_pts_mm[0][:, 2], z=traj_pts_mm[0][:, 1], mode="lines+markers", name="home", line={"width": 5, "color": "rgb(80,130,255)"}))
    fig.add_trace(go.Scatter3d(x=final_pts[:, 0], y=final_pts[:, 2], z=final_pts[:, 1], mode="lines+markers+text", name="target pose", text=["base", "j1", "j2", "j3", "tool base", "tcp"], textposition="top center", line={"width": 6, "color": "rgb(70,200,100)"}))
    fig.add_trace(go.Scatter3d(x=final_fingers[:, 0], y=final_fingers[:, 2], z=final_fingers[:, 1], mode="lines+markers", name="gripper width", line={"width": 8, "color": "rgb(220,120,40)"}))
    fig.add_trace(go.Scatter3d(x=tcp_path[:, 0], y=tcp_path[:, 2], z=tcp_path[:, 1], mode="lines", name="tcp path", line={"width": 4, "dash": "dash", "color": "rgb(190,60,220)"}))

    if scene_objects:
        for obj in scene_objects:
            center_mm = obj.get("center_xyz_mm")
            if center_mm is None:
                center_mm = obj.get("center_mm")
            if center_mm is None:
                continue
            c = np.asarray(center_mm, dtype=np.float64)
            fig.add_trace(go.Scatter3d(x=[c[0]], y=[c[2]], z=[c[1]], mode="markers+text", name=obj.get("class_name", "obj"), text=[obj.get("class_name", "obj")], textposition="top center", marker={"size": 5}, showlegend=False))

    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "X (mm)",
            "yaxis_title": "Z (mm)",
            "zaxis_title": "Y (mm)",
            "xaxis": {"autorange": "reversed"},
            "aspectmode": "data",
        },
        height=760,
    )
    return fig, gif_path