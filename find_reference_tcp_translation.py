"""Find the translation between a reference tag and a TCP tag.

This is a live calibration helper:
1. Wait until both the reference tag and TCP tag are visible.
2. For the next 5 seconds, read the robot TCP position from pylibfranka
   and the TCP tag pose from the camera.
3. Estimate the translation that maps the tag pose into the Franka base frame.

The rotation is assumed to already be correct. Edit the hardcoded rotation and
translation block near the top of this file if the tag-to-base convention
changes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from real_robot_exps.pro_robot_interface import FrankaInterface  # noqa: E402

TAG_SIZE_M = 0.0170

# Edit this block if the tag-to-base convention changes.
# Current convention:
#   base x = tag x
#   base y = tag z
#   base z = -tag y
TAG_TO_BASE_ROTATION = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
], dtype=np.float64)


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_detector() -> Detector:
    return Detector(
        families="tag36h11",
        quad_decimate=1.0,
        nthreads=24,
        refine_edges=1,
        quad_sigma=0.2,
        decode_sharpening=1.0,
    )


def _init_camera(camera_fps: int, width: int, height: int, exposure: int):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, camera_fps)
    profile = pipeline.start(config)
    sensor = profile.get_device().query_sensors()[1]
    sensor.set_option(rs.option.enable_auto_exposure, 0)
    sensor.set_option(rs.option.exposure, exposure)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    camera_params = (intr.fx, intr.fy, intr.ppx, intr.ppy)
    return pipeline, camera_params


def _detect_tags(detector, frame, camera_params, decision_margin, allowed_ids):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raw_tags = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=camera_params,
        tag_size=TAG_SIZE_M,
    )
    return {
        tag.tag_id: tag
        for tag in raw_tags
        if tag.decision_margin > decision_margin and tag.tag_id in allowed_ids
    }


def _tag_pose_in_reference(tag, ref_tag):
    r_ref_inv = ref_tag.pose_R.T
    t_ref = ref_tag.pose_t
    pos = (r_ref_inv @ (tag.pose_t - t_ref)).reshape(3)
    rot = r_ref_inv @ tag.pose_R
    return pos, rot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("real_robot_exps/config.yaml"))
    parser.add_argument("--reference-id", type=int, default=1)
    parser.add_argument("--tcp-id", type=int, default=9)
    parser.add_argument("--decision-margin", type=float, default=3.0)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-fps", type=int, default=15)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--exposure", type=int, default=100)
    args = parser.parse_args()

    real_config = _load_config(args.config)
    robot = FrankaInterface(real_config, device="cpu")
    detector = _make_detector()
    pipeline, camera_params = _init_camera(
        args.camera_fps, args.camera_width, args.camera_height, args.exposure
    )

    samples: list[dict[str, object]] = []
    collecting = False
    collect_start = None

    try:
        robot.start_torque_mode()
        time.sleep(0.5)

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            tags = _detect_tags(
                detector,
                frame,
                camera_params,
                args.decision_margin,
                allowed_ids=(args.reference_id, args.tcp_id),
            )
            snap = robot.get_state_snapshot()
            now = time.time()

            both_visible = args.reference_id in tags and args.tcp_id in tags
            if both_visible and not collecting:
                collecting = True
                collect_start = now
                samples.clear()
                print("Both tags visible. Starting 5 second capture window.")

            if collecting and collect_start is not None:
                if args.reference_id not in tags or args.tcp_id not in tags:
                    print("Lost one tag during capture; restarting window.")
                    collecting = False
                    collect_start = None
                    samples.clear()
                else:
                    ref_tag = tags[args.reference_id]
                    tcp_tag = tags[args.tcp_id]
                    tcp_pos_ref, tcp_rot_ref = _tag_pose_in_reference(tcp_tag, ref_tag)
                    tcp_pos_base_est = TAG_TO_BASE_ROTATION @ tcp_pos_ref
                    translation_est = snap.ee_pos.detach().cpu().numpy() - tcp_pos_base_est
                    samples.append({
                        "timestamp": now,
                        "robot_tcp_pos": snap.ee_pos.detach().cpu().numpy(),
                        "tcp_pos_ref": tcp_pos_ref,
                        "tcp_pos_base_est": tcp_pos_base_est,
                        "translation_est": translation_est,
                        "tcp_rot_ref_quat_xyzw": R.from_matrix(tcp_rot_ref).as_quat(),
                    })
                    elapsed = now - collect_start
                    if args.show:
                        cv2.putText(
                            frame,
                            f"t={elapsed:.1f}s  trans=[{translation_est[0]:+.3f}, {translation_est[1]:+.3f}, {translation_est[2]:+.3f}] m",
                            (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                        )

                    if elapsed >= args.window_sec and len(samples) >= args.min_samples:
                        break

            if args.show:
                if both_visible:
                    cv2.putText(frame, "Both tags visible", (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Waiting for both tags", (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Reference-TCP translation", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if len(samples) < args.min_samples:
            raise RuntimeError(
                f"Collected only {len(samples)} samples; keep both tags visible for the full window."
            )

        translations = np.asarray([s["translation_est"] for s in samples], dtype=np.float64)
        tcp_ref_positions = np.asarray([s["tcp_pos_ref"] for s in samples], dtype=np.float64)
        robot_tcp_positions = np.asarray([s["robot_tcp_pos"] for s in samples], dtype=np.float64)

        summary = {
            "samples": len(samples),
            "window_sec": args.window_sec,
            "reference_id": args.reference_id,
            "tcp_id": args.tcp_id,
            "translation_mean_m": translations.mean(axis=0).tolist(),
            "translation_median_m": np.median(translations, axis=0).tolist(),
            "translation_std_m": translations.std(axis=0).tolist(),
            "tcp_ref_mean_m": tcp_ref_positions.mean(axis=0).tolist(),
            "robot_tcp_mean_m": robot_tcp_positions.mean(axis=0).tolist(),
            "note": "translation_est = robot_tcp_pos - (TAG_TO_BASE_ROTATION @ tcp_pos_ref)",
        }
        print(json.dumps(summary, indent=2))
    finally:
        try:
            robot.end_control()
        except Exception:
            pass
        robot.shutdown()
        pipeline.stop()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
