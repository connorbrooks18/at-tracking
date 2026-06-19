"""Replay.py — visualize recorded Tracker/DataCollector output from a parquet file.

Replays logged (x, y, z, qx, qy, qz, qw) poses for each tracked object on a
blank canvas, drawing each object's position and orientation axes over time —
no camera or live tags required.

Usage:
    python Replay.py output.parquet
    python Replay.py output.parquet --speed 2.0
    python Replay.py output.parquet --fps 30
"""

import argparse
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from DataCollector import DataCollector

# ── Visualization config ─────────────────────────────────────────────────────
CANVAS_SIZE   = (800, 800)     # (height, width) in pixels
PIXELS_PER_M  = 800           # scale: pixels per metre, tune to your motion range
ORIGIN_PX     = (400, 400)     # where (0,0) in world coords maps to on screen
AXIS_LEN_M    = 0.04           # length of drawn orientation axis, metres
TRAIL_LEN     = 60             # number of past positions to draw as a fading trail

# Distinct colors per tracker name (BGR), falls back to white if name unseen.
TRACKER_COLORS = {
    "Apple":  (60, 60, 230),   # red-ish
    "Branch": (60, 200, 60),   # green-ish
}
DEFAULT_COLOR = (255, 255, 255)


def world_to_pixel(x, y):
    """Project world-frame (x, y) in metres to pixel coordinates.

    Uses the reference frame's X/Y plane (camera convention: +X right, +Y down)
    so this maps fairly naturally onto a 2D canvas.
    """
    px = int(ORIGIN_PX[0] + x * PIXELS_PER_M)
    py = int(ORIGIN_PX[1] + y * PIXELS_PER_M)
    return px, py


def draw_pose(canvas, name, x, y, z, qx, qy, qz, qw, trail):
    """Draw one object's position, orientation axis, and label on the canvas."""
    color = TRACKER_COLORS.get(name, DEFAULT_COLOR)
    cx, cy = world_to_pixel(x, y)

    # ── trail ──
    trail.append((cx, cy))
    if len(trail) > TRAIL_LEN:
        trail.pop(0)
    for i in range(1, len(trail)):
        alpha = i / len(trail)
        faded = tuple(int(c * alpha) for c in color)
        cv2.line(canvas, trail[i - 1], trail[i], faded, 1)

    # ── orientation axis (project the object's +Z onto the XY canvas) ──
    rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
    z_axis_world = rot[:, 2] * AXIS_LEN_M
    ax, ay = world_to_pixel(x + z_axis_world[0], y + z_axis_world[1])
    cv2.line(canvas, (cx, cy), (ax, ay), color, 2)

    # ── origin marker ──
    cv2.circle(canvas, (cx, cy), 6, color, -1)

    # ── label ──
    label = f"{name}  x:{x:+.3f} y:{y:+.3f} z:{z:+.3f}"
    cv2.putText(canvas, label, (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def draw_origin_marker(canvas):
    """Mark the reference tag's origin on the canvas for orientation."""
    ox, oy = ORIGIN_PX
    cv2.drawMarker(canvas, (ox, oy), (180, 180, 180),
                    markerType=cv2.MARKER_CROSS, markerSize=14, thickness=1)
    cv2.putText(canvas, "Reference origin", (ox + 10, oy + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)


def load_rows(filename):
    """Load parquet data and return rows grouped by timestamp.

    Returns:
        List of (milli, {name: (x,y,z,qx,qy,qz,qw)}) sorted by time.
    """
    dc = DataCollector()
    table = dc.read(filename)
    df = table.to_pandas()

    # DataCollector.update appends rows as plain lists in this column order:
    # ['milli', 'name', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
    # If the parquet file was written with named columns already, this just works;
    # if columns came through unnamed, assign the expected order defensively.
    expected_cols = ["milli", "name", "x", "y", "z", "qx", "qy", "qz", "qw"]
    if list(df.columns) != expected_cols and len(df.columns) == len(expected_cols):
        df.columns = expected_cols

    df = df.sort_values("milli")

    grouped = {}
    for _, row in df.iterrows():
        t = row["milli"]
        grouped.setdefault(t, {})[row["name"]] = (
            row["x"], row["y"], row["z"],
            row["qx"], row["qy"], row["qz"], row["qw"],
        )

    return sorted(grouped.items(), key=lambda kv: kv[0])


def main():
    parser = argparse.ArgumentParser(description="Replay recorded tracker data.")
    parser.add_argument("filename", help="Path to .parquet file from DataCollector")
    parser.add_argument("--speed", type=float, default=1.0,
                         help="Playback speed multiplier (default: 1.0)")
    parser.add_argument("--fps", type=float, default=30.0,
                         help="Target draw rate when timestamps allow (default: 30)")
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
    print("Controls: space = pause/resume, q = quit, r = restart")

    trails = {}  # name -> list of (px, py)
    paused = False
    idx = 0
    t0_data = timeline[0][0]
    t0_wall = time.time()

    while True:
        if idx >= len(timeline):
            if args.loop:
                idx = 0
                trails.clear()
                t0_data = timeline[0][0]
                t0_wall = time.time()
            else:
                print("Playback finished.")
                cv2.waitKey(0)
                break

        canvas = np.zeros((*CANVAS_SIZE, 3), dtype=np.uint8)
        draw_origin_marker(canvas)

        milli, objects = timeline[idx]

        for name, (x, y, z, qx, qy, qz, qw) in objects.items():
            trail = trails.setdefault(name, [])
            draw_pose(canvas, name, x, y, z, qx, qy, qz, qw, trail)

        # HUD
        elapsed = milli - t0_data
        cv2.putText(canvas, f"t={elapsed:.2f}s  frame {idx+1}/{len(timeline)}",
                    (10, CANVAS_SIZE[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        if paused:
            cv2.putText(canvas, "PAUSED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Replay", canvas)

        # pacing: try to match real elapsed time between recorded samples
        if not paused:
            target_wall = t0_wall + elapsed / args.speed
            wait_ms = max(1, int((target_wall - time.time()) * 1000))
        else:
            wait_ms = 30

        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            if not paused:
                # resync wall clock so playback doesn't jump ahead
                t0_wall = time.time() - elapsed / args.speed
        elif key == ord('r'):
            idx = -1
            trails.clear()
            t0_data = timeline[0][0]
            t0_wall = time.time()

        if not paused:
            idx += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
