"""Estimate an apple transform from a visible AprilTag under a simple assumption.

Assumption:
- the apple frame has no extra rotation relative to the tag frame
- the apple faces the tag's +z direction

This is a lightweight diagnostic helper:
- no parquet
- no robot
- prints the implied 4x4 apple transform from the detected tag pose
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))


TAG_SIZE_M = 0.0170


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


def _make_detector():
    return Detector(
        families="tag36h11",
        quad_decimate=1.0,
        nthreads=24,
        refine_edges=1,
        quad_sigma=0.2,
        decode_sharpening=1.0,
    )


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


def _make_transform(rot, pos):
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = np.asarray(rot, dtype=np.float64)
    tf[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return tf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag-id", type=int, default=6)
    parser.add_argument("--reference-id", type=int, default=1)
    parser.add_argument("--decision-margin", type=float, default=3.0)
    parser.add_argument("--camera-fps", type=int, default=15)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--exposure", type=int, default=100)
    parser.add_argument(
        "--tag-to-apple-z-m",
        type=float,
        default=0.0,
        help="Optional translation from tag origin to apple origin along tag +z.",
    )
    args = parser.parse_args()

    detector = _make_detector()
    pipeline, camera_params = _init_camera(
        args.camera_fps, args.camera_width, args.camera_height, args.exposure
    )

    try:
        print(
            f"Waiting for tag {args.tag_id} and reference tag {args.reference_id}. "
            "Press Ctrl-C to stop."
        )
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
                allowed_ids=(args.reference_id, args.tag_id),
            )
            if args.reference_id not in tags or args.tag_id not in tags:
                continue

            ref_tag = tags[args.reference_id]
            tag = tags[args.tag_id]

            r_ref_inv = ref_tag.pose_R.T
            t_ref = ref_tag.pose_t
            tag_pos_ref = (r_ref_inv @ (tag.pose_t - t_ref)).reshape(3)
            tag_rot_ref = r_ref_inv @ tag.pose_R

            # Assumption: the apple frame is aligned with the tag frame.
            # If you want to express the apple origin offset along tag +z,
            # change --tag-to-apple-z-m.
            apple_rot_ref = tag_rot_ref
            apple_pos_ref = tag_pos_ref + tag_rot_ref @ np.array(
                [0.0, 0.0, args.tag_to_apple_z_m], dtype=np.float64
            )

            apple_tf_ref = _make_transform(apple_rot_ref, apple_pos_ref)
            quat_xyzw = R.from_matrix(apple_rot_ref).as_quat()
            now = time.time()
            print(
                f"t={now:.3f}  apple_pos_ref_m=[{apple_pos_ref[0]:+.4f}, {apple_pos_ref[1]:+.4f}, {apple_pos_ref[2]:+.4f}] "
                f"apple_quat_xyzw=[{quat_xyzw[0]:+.5f}, {quat_xyzw[1]:+.5f}, {quat_xyzw[2]:+.5f}, {quat_xyzw[3]:+.5f}]"
            )
            print("apple_T_ref:")
            print(np.array2string(apple_tf_ref, precision=5, suppress_small=True))
            print()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
