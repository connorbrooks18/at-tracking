"""Browse a unified system-ID Parquet episode over a live RealSense feed.

The unified file stores robot and camera-derived geometry in Franka base O.
For replay, this script reads the reference-tag-to-base calibration saved in
the Parquet metadata, inverts it, and projects the resulting reference-tag
coordinates onto the *current* camera image.  Keep the reference tag visible.

The replay overlay follows the compiler topology directly:
three woody chords are stored in `woody_part_start_pos` / `woody_part_end_pos`
in `junction_names` order, and there is no synthetic `fruiting_base` point
used for display here.

Usage:
    python Replay.py ../pull_unified.parquet
    python Replay.py ../pull_unified.parquet --speed 2
    python Replay.py ../pull_unified.parquet --loop

Controls:
    Space       play / pause
    Right or d   next recorded row (pauses)
    Left or a    previous recorded row (pauses)
    Home or r    first row (pauses)
    End          last row (pauses)
    + / -        double / halve playback speed
    q or Esc     quit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

try:
    import pyrealsense2 as rs
    from pupil_apriltags import Detector
except ImportError:  # Allows schema/calibration helpers to be used headlessly.
    rs = None
    Detector = None

# Keep these values local to the standalone replay tool. The robot package has
# no camera-runtime dependency.
TAG_SIZE_M = 0.0170
REFERENCE_TAG_ID = 1

# Must match the physical reference tag used by the tracker/compiled episode.
REFERENCE_ID = REFERENCE_TAG_ID
DECISION_MARGIN = 5
AXIS_LEN_M = 0.040

COLOR_TCP = (255, 255, 0)       # cyan, BGR
COLOR_APPLE = (255, 0, 255)     # magenta
COLOR_BRANCH = (0, 210, 255)    # yellow
COLOR_SPUR = (0, 180, 0)        # green
COLOR_APPLE_CHORD = (70, 70, 220)
COLOR_TEXT = (245, 245, 245)


@dataclass(frozen=True)
class UnifiedEpisode:
    """Rows and the calibration needed to put base-O points back in tag frame."""

    rows: list[dict]
    base_to_reference_4x4: np.ndarray
    episode_id: str
    junction_names: tuple[str, ...]


def init_camera():
    if rs is None:
        raise RuntimeError("Replay needs pyrealsense2; run it in the RealSense environment.")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    camera_matrix = np.array(
        [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return pipeline, camera_matrix, np.zeros(5, dtype=np.float64), (intr.fx, intr.fy, intr.ppx, intr.ppy)


def init_detector():
    if Detector is None:
        raise RuntimeError("Replay needs pupil_apriltags; run it in the AprilTag environment.")
    return Detector(
        families="tag36h11",
        quad_decimate=1.0,
        nthreads=24,
        refine_edges=1,
        quad_sigma=0.2,
        decode_sharpening=1.0,
    )


def get_reference_extrinsics(frame, detector, camera_params):
    """Return camera<-reference rvec/tvec for the currently detected reference tag."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=TAG_SIZE_M)
    for tag in tags:
        if tag.tag_id == REFERENCE_ID and tag.decision_margin > DECISION_MARGIN:
            rvec, _ = cv2.Rodrigues(np.asarray(tag.pose_R, dtype=np.float64))
            return rvec, np.asarray(tag.pose_t, dtype=np.float64).reshape(3, 1)
    return None, None


def _dataset_metadata(path: Path) -> dict:
    metadata = pq.read_schema(path).metadata or {}
    raw = metadata.get(b"dataset_metadata")
    if raw is None:
        return {}
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Parquet dataset_metadata is not valid JSON") from exc


def load_unified_episode(filename: str) -> UnifiedEpisode:
    """Load a unified episode and invert its saved reference-tag calibration."""
    path = Path(filename)
    table = pq.read_table(path)
    required = {"timestamp", "tcp_pos", "apple_pos", "woody_part_start_pos", "woody_part_end_pos"}
    missing = required - set(table.column_names)
    if missing:
        raise ValueError(
            "Replay expects a unified static system-ID Parquet file; missing fields: "
            + ", ".join(sorted(missing))
        )

    metadata = _dataset_metadata(path)
    topology = metadata.get("topology", {})
    junction_names = tuple(topology.get("junction_names", ("Branch", "Spur", "Apple")))
    tag_to_base = metadata.get("reference_tag_to_base_4x4_used", metadata.get("reference_tag_to_base_4x4"))
    if tag_to_base is None:
        raise ValueError(
            "Unified Parquet metadata has no reference_tag_to_base_4x4_used calibration. "
            "Recompile the episode with compile_static_sysid.py."
        )
    tag_to_base = np.asarray(tag_to_base, dtype=np.float64)
    if tag_to_base.shape != (4, 4) or not np.isfinite(tag_to_base).all():
        raise ValueError("reference_tag_to_base_4x4_used must be a finite 4x4 matrix")
    if abs(float(np.linalg.det(tag_to_base[:3, :3]))) < 1e-10:
        raise ValueError("reference_tag_to_base_4x4_used has a non-invertible rotation block")

    rows = table.to_pylist()
    if not rows:
        raise ValueError("No rows found in unified Parquet")
    rows.sort(key=lambda row: float(row["timestamp"]))
    return UnifiedEpisode(
        rows=rows,
        base_to_reference_4x4=np.linalg.inv(tag_to_base),
        episode_id=str(metadata.get("episode_id", rows[0].get("episode_id", ""))),
        junction_names=junction_names,
    )


def _as_point(value) -> np.ndarray | None:
    if value is None:
        return None
    point = np.asarray(value, dtype=np.float64).reshape(-1)
    if point.size != 3 or not np.isfinite(point).all():
        return None
    return point


def point_base_to_reference(point_base, base_to_reference_4x4: np.ndarray) -> np.ndarray | None:
    """Transform one 3-D base-O point into the reference-tag coordinate frame."""
    point = _as_point(point_base)
    if point is None:
        return None
    homogeneous = base_to_reference_4x4 @ np.append(point, 1.0)
    if abs(homogeneous[3]) < 1e-12:
        return None
    return homogeneous[:3] / homogeneous[3]


def pose_base_to_reference(pose_base, base_to_reference_4x4: np.ndarray) -> np.ndarray | None:
    """Transform a row-major 4x4 base-O pose into reference-tag coordinates."""
    if pose_base is None:
        return None
    pose = np.asarray(pose_base, dtype=np.float64)
    if pose.size != 16 or not np.isfinite(pose).all():
        return None
    return base_to_reference_4x4 @ pose.reshape(4, 4)


def project_point(point_ref, rvec, tvec, camera_matrix, dist_coeffs):
    point = np.asarray(point_ref, dtype=np.float64).reshape(1, 3)
    projected, _ = cv2.projectPoints(point, rvec, tvec, camera_matrix, dist_coeffs)
    return tuple(np.rint(projected[0, 0]).astype(int))


def _in_image(point, frame) -> bool:
    x, y = point
    height, width = frame.shape[:2]
    return 0 <= x < width and 0 <= y < height


def draw_point(frame, point_ref, label, color, rvec, tvec, camera_matrix, dist_coeffs):
    if point_ref is None:
        return None
    pixel = project_point(point_ref, rvec, tvec, camera_matrix, dist_coeffs)
    if _in_image(pixel, frame):
        cv2.circle(frame, pixel, 6, color, -1)
        cv2.putText(frame, label, (pixel[0] + 8, pixel[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
    return pixel


def draw_line(frame, start_ref, end_ref, label, color, rvec, tvec, camera_matrix, dist_coeffs):
    if start_ref is None or end_ref is None:
        return
    start_px = project_point(start_ref, rvec, tvec, camera_matrix, dist_coeffs)
    end_px = project_point(end_ref, rvec, tvec, camera_matrix, dist_coeffs)
    height, width = frame.shape[:2]
    clipped, start_px, end_px = cv2.clipLine((0, 0, width, height), start_px, end_px)
    if clipped:
        cv2.line(frame, start_px, end_px, color, 3)
        midpoint = ((start_px[0] + end_px[0]) // 2, (start_px[1] + end_px[1]) // 2)
        cv2.putText(frame, label, midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 2)


def draw_pose_axes(frame, pose_ref, label, color, rvec, tvec, camera_matrix, dist_coeffs):
    """Draw an origin and RGB local axes from a reference-frame 4x4 pose."""
    if pose_ref is None:
        return
    origin = pose_ref[:3, 3]
    rotation = pose_ref[:3, :3]
    origin_px = draw_point(frame, origin, label, color, rvec, tvec, camera_matrix, dist_coeffs)
    if origin_px is None:
        return
    for axis, axis_color, axis_name in ((0, (0, 0, 255), "x"), (1, (0, 255, 0), "y"), (2, (255, 0, 0), "z")):
        tip_px = project_point(origin + rotation[:, axis] * AXIS_LEN_M, rvec, tvec, camera_matrix, dist_coeffs)
        height, width = frame.shape[:2]
        clipped, line_start, line_end = cv2.clipLine((0, 0, width, height), origin_px, tip_px)
        if clipped:
            cv2.line(frame, line_start, line_end, axis_color, 2)
            if _in_image(tip_px, frame):
                cv2.putText(frame, axis_name, (tip_px[0] + 3, tip_px[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.43, axis_color, 1)


def draw_unified_row(frame, row, base_to_reference_4x4, rvec, tvec, camera_matrix, dist_coeffs):
    """Overlay every spatial unified-data field that can be projected meaningfully."""
    tcp_ref = point_base_to_reference(row.get("tcp_pos"), base_to_reference_4x4)
    apple_ref = point_base_to_reference(row.get("apple_pos"), base_to_reference_4x4)
    draw_point(frame, tcp_ref, "TCP", COLOR_TCP, rvec, tvec, camera_matrix, dist_coeffs)
    draw_point(frame, apple_ref, "Apple", COLOR_APPLE, rvec, tvec, camera_matrix, dist_coeffs)

    # The row stores the woody chord endpoints in `junction_names` order.
    # For replay we only render the two physical segments you care about:
    # Spur, then the Spur-to-Apple connection.
    starts = np.asarray(row.get("woody_part_start_pos", []), dtype=np.float64).reshape(-1)
    ends = np.asarray(row.get("woody_part_end_pos", []), dtype=np.float64).reshape(-1)
    if starts.size == 9 and ends.size == 9:
        spur_start = point_base_to_reference(starts[0:3], base_to_reference_4x4)
        spur_end = point_base_to_reference(ends[0:3], base_to_reference_4x4)
        draw_line(frame, spur_start, spur_end, "Spur", COLOR_SPUR, rvec, tvec, camera_matrix, dist_coeffs)
        draw_point(frame, spur_start, "Spur start", COLOR_SPUR, rvec, tvec, camera_matrix, dist_coeffs)
        draw_point(frame, spur_end, "Spur end", COLOR_SPUR, rvec, tvec, camera_matrix, dist_coeffs)

        draw_line(frame, spur_end, apple_ref, "Apple", COLOR_APPLE_CHORD, rvec, tvec, camera_matrix, dist_coeffs)
        draw_point(frame, apple_ref, "Apple", COLOR_APPLE_CHORD, rvec, tvec, camera_matrix, dist_coeffs)

    # Unlike tcp_pos, apple_pose_4x4 carries an orientation; show it when present.
    apple_pose_ref = pose_base_to_reference(row.get("apple_pose_4x4"), base_to_reference_4x4)
    draw_pose_axes(frame, apple_pose_ref, "Apple pose", COLOR_APPLE, rvec, tvec, camera_matrix, dist_coeffs)


def _timestamp(row) -> float:
    return float(row["timestamp"])


def _key_is_left(key: int) -> bool:
    return key in (81, 2424832, ord("a"))


def _key_is_right(key: int) -> bool:
    return key in (83, 2555904, ord("d"))


def _key_is_home(key: int) -> bool:
    return key in (ord("r"), 2359296)


def _key_is_end(key: int) -> bool:
    return key == 2293760


def main():
    parser = argparse.ArgumentParser(description="Browse a base-frame unified Parquet over a live camera feed.")
    parser.add_argument("filename", help="Path to a unified static system-ID .parquet file")
    parser.add_argument("--speed", type=float, default=1.0, help="Initial playback speed multiplier (default: 1)")
    parser.add_argument("--loop", action="store_true", help="Loop when playback reaches the final row")
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("--speed must be positive")
    if rs is None or Detector is None:
        print("[ERROR] Replay needs both pyrealsense2 and pupil_apriltags in the active Python environment.")
        sys.exit(1)

    try:
        episode = load_unified_episode(args.filename)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    print(f"Loaded {len(episode.rows)} unified rows from {args.filename}")
    print(f"episode_id: {episode.episode_id or '(not recorded)'}")
    print("Using inverse of metadata reference_tag_to_base_4x4_used (base O -> reference tag).")
    print("Keep reference tag ID=1 visible. Controls: Space play/pause, arrows or a/d step, r/Home first, End last, +/- speed, q quit.")

    pipeline, camera_matrix, dist_coeffs, camera_params = init_camera()
    detector = init_detector()
    index = 0
    paused = True
    speed = float(args.speed)
    anchor_index = 0
    anchor_wall = time.monotonic()

    def restart_clock():
        nonlocal anchor_index, anchor_wall
        anchor_index = index
        anchor_wall = time.monotonic()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())

            # Playback is timestamp-driven, so repeated rows at a static hold do not
            # artificially consume one row per camera frame.
            if not paused:
                elapsed_data = (time.monotonic() - anchor_wall) * speed
                base_time = _timestamp(episode.rows[anchor_index])
                while index + 1 < len(episode.rows) and _timestamp(episode.rows[index + 1]) - base_time <= elapsed_data:
                    index += 1
                if index == len(episode.rows) - 1 and elapsed_data >= _timestamp(episode.rows[-1]) - base_time:
                    if args.loop:
                        index = 0
                        restart_clock()
                    else:
                        paused = True

            rvec, tvec = get_reference_extrinsics(frame, detector, camera_params)
            if rvec is None:
                cv2.putText(frame, f"Waiting for reference tag ID={REFERENCE_ID}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)
            else:
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05, 2)
                cv2.putText(frame, "Reference tag frame", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.56, COLOR_TEXT, 2)
                draw_unified_row(frame, episode.rows[index], episode.base_to_reference_4x4, rvec, tvec, camera_matrix, dist_coeffs)

            row = episode.rows[index]
            elapsed = _timestamp(row) - _timestamp(episode.rows[0])
            state = "PAUSED" if paused else "PLAYING"
            hud = f"{state}  row {index + 1}/{len(episode.rows)}  t={elapsed:.3f}s  hold={row.get('hold_index', '?')}  speed={speed:g}x"
            cv2.rectangle(frame, (7, frame.shape[0] - 35), (min(frame.shape[1] - 7, 850), frame.shape[0] - 7), (20, 20, 20), -1)
            cv2.putText(frame, hud, (13, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_TEXT, 1)
            cv2.imshow("Unified Parquet Replay", frame)

            key = cv2.waitKeyEx(1)
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
                restart_clock()
            elif _key_is_right(key):
                index = min(index + 1, len(episode.rows) - 1)
                paused = True
                restart_clock()
            elif _key_is_left(key):
                index = max(index - 1, 0)
                paused = True
                restart_clock()
            elif _key_is_home(key):
                index = 0
                paused = True
                restart_clock()
            elif _key_is_end(key):
                index = len(episode.rows) - 1
                paused = True
                restart_clock()
            elif key in (ord("+"), ord("=")):
                speed = min(speed * 2.0, 64.0)
                restart_clock()
            elif key in (ord("-"), ord("_")):
                speed = max(speed / 2.0, 0.0625)
                restart_clock()
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
