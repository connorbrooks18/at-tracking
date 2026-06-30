"""Replay.py — project recorded tracker poses onto a live RealSense feed.

The reference AprilTag (ID=2) must be visible in the live feed. Its current
pose is used to project the recorded world-frame positions into image space,
so the replay overlays correctly onto the physical scene.

Usage:
    python Replay.py output.parquet
    python Replay.py output.parquet --speed 2.0
    python Replay.py output.parquet --loop
"""

import argparse
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R

from DataCollector import DataCollector

# ── Config ────────────────────────────────────────────────────────────────────
REFERENCE_ID    = 1
TAG_SIZE_M      = 0.0170
DECISION_MARGIN = 5
TRAIL_LEN       = 60
AXIS_LEN_M      = 0.04

TRACKER_COLORS = {
    "Apple":  (60,  60,  180),
    "Branch": (60,  100,  60),
}
DEFAULT_COLOR = (255, 255, 255)


# ── Camera + detector setup ───────────────────────────────────────────────────

def init_camera():
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile  = pipeline.start(config)

    intr = (
        profile.get_stream(rs.stream.color)
        .as_video_stream_profile()
        .get_intrinsics()
    )
    K = np.array(
        [[intr.fx, 0,       intr.ppx],
         [0,       intr.fy, intr.ppy],
         [0,       0,       1       ]],
        dtype=np.float64,
    )
    dist          = np.zeros(5, dtype=np.float64)
    camera_params = (intr.fx, intr.fy, intr.ppx, intr.ppy)
    return pipeline, K, dist, camera_params


def init_detector():
    return Detector(families="tag36h11", nthreads=4,
                    quad_decimate=1.0, quad_sigma=0.8,
                    refine_edges=1, decode_sharpening=1, debug=0)


# ── Reference tag detection ───────────────────────────────────────────────────

def get_reference_extrinsics(frame, detector, camera_params, K, dist):
    """Detect the reference tag and return (rvec, tvec) for cv2.projectPoints.

    Returns (None, None) if the reference tag is not visible.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=camera_params,
        tag_size=TAG_SIZE_M,
    )
    for tag in tags:
        if tag.tag_id == REFERENCE_ID and tag.decision_margin > DECISION_MARGIN:
            rvec, _ = cv2.Rodrigues(tag.pose_R.astype(np.float64))
            tvec    = tag.pose_t.astype(np.float64).reshape(3, 1)
            return rvec, tvec
    return None, None


# ── Projection helpers ────────────────────────────────────────────────────────

def project_point(pos_ref, rvec, tvec, K, dist):
    """Project a 3D point in the reference frame into image pixels."""
    pt  = np.array(pos_ref, dtype=np.float64).reshape(1, 3)
    px, _ = cv2.projectPoints(pt, rvec, tvec, K, dist)
    return int(px[0, 0, 0]), int(px[0, 0, 1])


def draw_replay_pose(frame, name, x, y, z, qx, qy, qz, qw,
                     rvec, tvec, K, dist, trail):
    """Draw a recorded pose onto the live frame."""
    color    = TRACKER_COLORS.get(name, DEFAULT_COLOR)
    h, w     = frame.shape[:2]
    cx, cy   = project_point([x, y, z], rvec, tvec, K, dist)

    # trail
    trail.append((cx, cy))
    if len(trail) > TRAIL_LEN:
        trail.pop(0)
    for i in range(1, len(trail)):
        alpha = i / len(trail)
        faded = tuple(int(c * alpha) for c in color)
        cv2.line(frame, trail[i - 1], trail[i], faded, 1)

    # orientation axis: project the object's +Z tip
    rot   = R.from_quat([qx, qy, qz, qw]).as_matrix()
    z_tip = np.array([x, y, z]) + rot[:, 2] * AXIS_LEN_M
    ax, ay = project_point(z_tip, rvec, tvec, K, dist)
    clipped, p1, p2 = cv2.clipLine((0, 0, w, h), (cx, cy), (ax, ay))
    if clipped:
        cv2.line(frame, p1, p2, color, 2)

    # origin dot + label
    if 0 <= cx < w and 0 <= cy < h:
        cv2.circle(frame, (cx, cy), 6, color, -1)
        label = f"{name}  x:{x:+.3f} y:{y:+.3f} z:{z:+.3f}"
        cv2.putText(frame, label, (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_rows(filename):
    """Load parquet and return [(milli, {name: (x,y,z,qx,qy,qz,qw)})] sorted by time."""
    dc    = DataCollector()
    table = dc.read(filename)
    df    = table.to_pandas()

    expected = ["milli", "name", "x", "y", "z", "qx", "qy", "qz", "qw"]
    if list(df.columns) != expected and len(df.columns) == len(expected):
        df.columns = expected

    df = df.sort_values("milli")

    grouped = {}
    for _, row in df.iterrows():
        t = row["milli"]
        grouped.setdefault(t, {})[row["name"]] = (
            row["x"], row["y"], row["z"],
            row["qx"], row["qy"], row["qz"], row["qw"],
        )

    return sorted(grouped.items(), key=lambda kv: kv[0])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Replay tracker data projected onto a live camera feed."
    )
    parser.add_argument("filename", help="Path to .parquet file from DataCollector")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (default: 1.0)")
    parser.add_argument("--loop", action="store_true",
                        help="Loop playback when it reaches the end")
    args = parser.parse_args()

    try:
        timeline = load_rows(args.filename)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {args.filename}")
        sys.exit(1)

    if not timeline:
        print("[ERROR] No rows found in file.")
        sys.exit(1)

    print(f"Loaded {len(timeline)} timestamps from {args.filename}")
    print(f"Point reference tag (ID={REFERENCE_ID}) at the camera.")
    print("Controls: space = pause/resume  r = restart  q = quit")

    pipeline, K, dist, camera_params = init_camera()
    detector = init_detector()

    trails  = {}
    paused  = False
    idx     = 0
    t0_data = timeline[0][0]
    t0_wall = time.time()

    try:
        while True:
            # grab live frame
            frames      = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())

            # detect reference tag every frame
            rvec, tvec = get_reference_extrinsics(
                frame, detector, camera_params, K, dist
            )

            if rvec is None:
                cv2.putText(frame,
                            f"Waiting for reference tag (ID={REFERENCE_ID})...",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)
                cv2.imshow("Replay", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # draw reference axes so you can verify alignment
            cv2.drawFrameAxes(frame, K, dist, rvec, tvec, 0.05, 2)

            # handle end of timeline
            if idx >= len(timeline):
                if args.loop:
                    idx     = 0
                    trails.clear()
                    t0_data = timeline[0][0]
                    t0_wall = time.time()
                else:
                    cv2.putText(frame, "Playback finished — q to quit",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)
                    cv2.imshow("Replay", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

            milli, objects = timeline[idx]
            elapsed = milli - t0_data

            # draw each recorded object
            for name, (x, y, z, qx, qy, qz, qw) in objects.items():
                if (x,y,z) == (0, 0, 0):
                    continue
                trail = trails.setdefault(name, [])
                draw_replay_pose(
                    frame, name, x, y, z, qx, qy, qz, qw,
                    rvec, tvec, K, dist, trail,
                )

            # HUD
            cv2.putText(frame,
                        f"t={elapsed:.2f}s  frame {idx+1}/{len(timeline)}  "
                        f"speed={args.speed}x",
                        (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            if paused:
                cv2.putText(frame, "PAUSED", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Replay", frame)

            # pace to match recorded timestamps
            if not paused:
                target_wall = t0_wall + elapsed / args.speed
                wait_ms     = max(1, int((target_wall - time.time()) * 1000))
            else:
                wait_ms = 30

            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                if not paused:
                    t0_wall = time.time() - elapsed / args.speed
            elif key == ord('r'):
                idx     = 0
                trails.clear()
                t0_data = timeline[0][0]
                t0_wall = time.time()

            if not paused:
                idx += 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
