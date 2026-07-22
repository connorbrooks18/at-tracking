"""Headless apple pose reader using the same offsets and tag IDs as Detecting.py.

This script prints the estimated apple pose in the reference frame:
- position in meters
- quaternion in xyzw order

It works when one or both apple tags are visible.
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

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

REFERENCE_TAG_ID = 1
TAG_SIZE_M = 0.0170
APPLE_IDS = (7, 6)
APPLE_OFFSETS = (
    {"pos": [0.0, 0.0, 0.11], "rot": [[-0.7071, 0, -0.7071], [0, 1, 0], [0.7071, 0, -0.7071]]},
    {"pos": [0.085, 0.0, 0.0], "rot": [[0.7071, 0, -0.7071], [0, 1, 0], [0.7071, 0, 0.7071]]},
)

import Tracker  # noqa: E402

TrackedObject = Tracker.Tracker

# Assumed transform from tag frame to apple frame, expressed in tag coordinates.
# This is the part to edit when testing a new physical tag mounting hypothesis.
TAG_TO_APPLE_ROTATION = np.eye(3, dtype=np.float64)
TAG_TO_APPLE_TRANSLATION_M = np.array([0.0, 0.0, 0.0], dtype=np.float64)


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


def _tag_pose_in_reference(tag, ref_tag):
    r_ref_inv = ref_tag.pose_R.T
    t_ref = ref_tag.pose_t
    pos = (r_ref_inv @ (tag.pose_t - t_ref)).reshape(3)
    rot = r_ref_inv @ tag.pose_R
    return pos, rot


def _make_transform(rot, pos):
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = np.asarray(rot, dtype=np.float64)
    tf[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return tf


def _tag_to_apple_transform():
    return _make_transform(TAG_TO_APPLE_ROTATION, TAG_TO_APPLE_TRANSLATION_M)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-id", type=int, default=REFERENCE_TAG_ID)
    parser.add_argument("--decision-margin", type=float, default=3.0)
    parser.add_argument("--camera-fps", type=int, default=15)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--exposure", type=int, default=100)
    parser.add_argument("--print-every", type=float, default=0.0,
                        help="Minimum seconds between printed samples. 0 prints every valid frame.")
    args = parser.parse_args()

    detector = _make_detector()
    pipeline, camera_params = _init_camera(
        args.camera_fps, args.camera_width, args.camera_height, args.exposure
    )
    apple = TrackedObject("Apple", APPLE_IDS, APPLE_OFFSETS)

    fused_pos_samples = []
    fused_quat_samples = []
    tag_samples = {tag_id: {"pos": [], "quat": []} for tag_id in APPLE_IDS}
    both_visible_samples = []
    last_print = 0.0

    try:
        print("Headless apple pose reader running. Press Ctrl-C to stop.")
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
                allowed_ids=(args.reference_id, *APPLE_IDS),
            )
            if args.reference_id not in tags:
                continue

            ref_tag = tags[args.reference_id]
            tags_in_ref = {
                tag_id: {
                    "pos": pos,
                    "rot": rot,
                }
                for tag_id, tag in tags.items()
                if tag_id != args.reference_id
                for pos, rot in [_tag_pose_in_reference(tag, ref_tag)]
            }
            apple.updatePose(tags_in_ref)

            if not tags_in_ref and apple.pose is None:
                continue

            now = time.time()
            if args.print_every > 0.0 and (now - last_print) < args.print_every:
                continue
            last_print = now

            fused_pos = None
            fused_quat = None
            if apple.pose is not None:
                fused_pos = np.asarray(apple.pose["pos"], dtype=np.float64)
                fused_rot = np.asarray(apple.pose["rot"], dtype=np.float64)
                fused_quat = R.from_matrix(fused_rot).as_quat()  # xyzw
                fused_pos_samples.append(fused_pos.copy())
                fused_quat_samples.append(fused_quat.copy())

            per_tag_lines = []
            seen_tag_ids = []
            for tag_id in APPLE_IDS:
                if tag_id not in tags_in_ref:
                    continue
                seen_tag_ids.append(tag_id)
                tag_pos = np.asarray(tags_in_ref[tag_id]["pos"], dtype=np.float64)
                tag_rot = np.asarray(tags_in_ref[tag_id]["rot"], dtype=np.float64)
                tag_quat = R.from_matrix(tag_rot).as_quat()
                tag_to_apple = _tag_to_apple_transform()
                apple_tf_from_tag = _make_transform(tag_rot, tag_pos) @ tag_to_apple
                apple_from_tag_pos = apple_tf_from_tag[:3, 3]
                apple_from_tag_quat = R.from_matrix(apple_tf_from_tag[:3, :3]).as_quat()
                tag_samples[tag_id]["pos"].append(tag_pos.copy())
                tag_samples[tag_id]["quat"].append(tag_quat.copy())
                per_tag_lines.append(
                    f"tag{tag_id}_apple_pos_m=[{tag_pos[0]:+.4f}, {tag_pos[1]:+.4f}, {tag_pos[2]:+.4f}]"
                    f" tag{tag_id}_apple_quat_xyzw=[{tag_quat[0]:+.5f}, {tag_quat[1]:+.5f}, {tag_quat[2]:+.5f}, {tag_quat[3]:+.5f}]"
                    f" tag{tag_id}_assumed_apple_pos_m=[{apple_from_tag_pos[0]:+.4f}, {apple_from_tag_pos[1]:+.4f}, {apple_from_tag_pos[2]:+.4f}]"
                    f" tag{tag_id}_assumed_apple_quat_xyzw=[{apple_from_tag_quat[0]:+.5f}, {apple_from_tag_quat[1]:+.5f}, {apple_from_tag_quat[2]:+.5f}, {apple_from_tag_quat[3]:+.5f}]"
                )
                print(f"tag{tag_id}_ref_to_tag_T:")
                print(np.array2string(_make_transform(tag_rot, tag_pos), precision=5, suppress_small=True))
                print(f"tag{tag_id}_tag_to_apple_T:")
                print(np.array2string(tag_to_apple, precision=5, suppress_small=True))
                print(f"tag{tag_id}_ref_to_apple_T:")
                print(np.array2string(apple_tf_from_tag, precision=5, suppress_small=True))

            if all(tag_id in tags_in_ref for tag_id in APPLE_IDS):
                pos0 = np.asarray(tags_in_ref[APPLE_IDS[0]]["pos"], dtype=np.float64)
                rot0 = np.asarray(tags_in_ref[APPLE_IDS[0]]["rot"], dtype=np.float64)
                quat0 = R.from_matrix(rot0).as_quat()
                pos1 = np.asarray(tags_in_ref[APPLE_IDS[1]]["pos"], dtype=np.float64)
                rot1 = np.asarray(tags_in_ref[APPLE_IDS[1]]["rot"], dtype=np.float64)
                quat1 = R.from_matrix(rot1).as_quat()
                rel_rot = R.from_matrix(rot0).inv() * R.from_matrix(rot1)
                both_visible_samples.append(
                    {
                        "timestamp": now,
                        "pos0": pos0,
                        "quat0": quat0,
                        "pos1": pos1,
                        "quat1": quat1,
                        "pos_delta": pos1 - pos0,
                        "angle_deg": rel_rot.magnitude() * 180.0 / np.pi,
                    }
                )

            print(
                f"t={now:.3f}"
                + (
                    f"  fused_apple_pos_m=[{fused_pos[0]:+.4f}, {fused_pos[1]:+.4f}, {fused_pos[2]:+.4f}]"
                    f"  fused_apple_quat_xyzw=[{fused_quat[0]:+.5f}, {fused_quat[1]:+.5f}, {fused_quat[2]:+.5f}, {fused_quat[3]:+.5f}]"
                    if fused_pos is not None and fused_quat is not None
                    else "  fused_apple=unavailable"
                )
                + "  tags_seen="
                + str(sorted(seen_tag_ids))
                + ("  " + " | ".join(per_tag_lines) if per_tag_lines else "")
            )
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()

        if fused_pos_samples:
            pos_arr = np.asarray(fused_pos_samples, dtype=np.float64)
            quat_arr = np.asarray(fused_quat_samples, dtype=np.float64)
            quat_mean = quat_arr.mean(axis=0)
            quat_mean_norm = np.linalg.norm(quat_mean)
            quat_mean_unit = quat_mean / quat_mean_norm if quat_mean_norm > 0 else quat_mean
            print("\nSummary")
            print(f"fused samples: {len(fused_pos_samples)}")
            print(
                "mean fused apple pos m: "
                f"[{pos_arr[:, 0].mean():+.4f}, {pos_arr[:, 1].mean():+.4f}, {pos_arr[:, 2].mean():+.4f}]"
            )
            print(
                "std fused apple pos m:  "
                f"[{pos_arr[:, 0].std():+.4f}, {pos_arr[:, 1].std():+.4f}, {pos_arr[:, 2].std():+.4f}]"
            )
            print(
                "mean fused apple quat xyzw: "
                f"[{quat_mean_unit[0]:+.5f}, {quat_mean_unit[1]:+.5f}, {quat_mean_unit[2]:+.5f}, {quat_mean_unit[3]:+.5f}]"
            )
            ref_rot = R.from_quat(quat_mean_unit)
            sample_rots = R.from_quat(quat_arr)
            angular_deviation_deg = (ref_rot.inv() * sample_rots).magnitude() * 180.0 / np.pi
            print(
                "fused quat angular deviation deg: "
                f"mean={angular_deviation_deg.mean():.3f}, std={angular_deviation_deg.std():.3f}, "
                f"median={np.median(angular_deviation_deg):.3f}, min/max={angular_deviation_deg.min():.3f}/{angular_deviation_deg.max():.3f}"
            )
            for tag_id in APPLE_IDS:
                samples = tag_samples[tag_id]["pos"]
                quats = tag_samples[tag_id]["quat"]
                if not samples:
                    print(f"tag {tag_id}: no samples")
                    continue
                tag_pos_arr = np.asarray(samples, dtype=np.float64)
                tag_quat_arr = np.asarray(quats, dtype=np.float64)
                tag_quat_mean = tag_quat_arr.mean(axis=0)
                tag_quat_mean_norm = np.linalg.norm(tag_quat_mean)
                tag_quat_mean_unit = (
                    tag_quat_mean / tag_quat_mean_norm if tag_quat_mean_norm > 0 else tag_quat_mean
                )
                tag_ref_rot = R.from_quat(tag_quat_mean_unit)
                tag_sample_rots = R.from_quat(tag_quat_arr)
                tag_ang_dev_deg = (tag_ref_rot.inv() * tag_sample_rots).magnitude() * 180.0 / np.pi
                print(f"tag {tag_id} samples: {len(samples)}")
                print(
                    f"tag {tag_id} mean apple pos m: "
                    f"[{tag_pos_arr[:, 0].mean():+.4f}, {tag_pos_arr[:, 1].mean():+.4f}, {tag_pos_arr[:, 2].mean():+.4f}]"
                )
                print(
                    f"tag {tag_id} std apple pos m:  "
                    f"[{tag_pos_arr[:, 0].std():+.4f}, {tag_pos_arr[:, 1].std():+.4f}, {tag_pos_arr[:, 2].std():+.4f}]"
                )
                print(
                    f"tag {tag_id} mean apple quat xyzw: "
                    f"[{tag_quat_mean_unit[0]:+.5f}, {tag_quat_mean_unit[1]:+.5f}, {tag_quat_mean_unit[2]:+.5f}, {tag_quat_mean_unit[3]:+.5f}]"
                )
                print(
                    f"tag {tag_id} quat angular deviation deg: "
                    f"mean={tag_ang_dev_deg.mean():.3f}, std={tag_ang_dev_deg.std():.3f}, "
                    f"median={np.median(tag_ang_dev_deg):.3f}, min/max={tag_ang_dev_deg.min():.3f}/{tag_ang_dev_deg.max():.3f}"
                )

            if both_visible_samples:
                pos_delta_arr = np.asarray([s["pos_delta"] for s in both_visible_samples], dtype=np.float64)
                angle_arr = np.asarray([s["angle_deg"] for s in both_visible_samples], dtype=np.float64)
                print("\nBoth-tags-visible section")
                print(f"frames with both tags visible: {len(both_visible_samples)}")
                print(
                    "mean tag1-tag2 position delta m: "
                    f"[{pos_delta_arr[:, 0].mean():+.4f}, {pos_delta_arr[:, 1].mean():+.4f}, {pos_delta_arr[:, 2].mean():+.4f}]"
                )
                print(
                    "std tag1-tag2 position delta m:  "
                    f"[{pos_delta_arr[:, 0].std():+.4f}, {pos_delta_arr[:, 1].std():+.4f}, {pos_delta_arr[:, 2].std():+.4f}]"
                )
                print(
                    "mean tag1-tag2 angular disagreement deg: "
                    f"{angle_arr.mean():.3f}"
                )
                print(
                    "std tag1-tag2 angular disagreement deg: "
                    f"{angle_arr.std():.3f}"
                )
                print(
                    "median/min/max tag1-tag2 angular disagreement deg: "
                    f"{np.median(angle_arr):.3f} / {angle_arr.min():.3f} / {angle_arr.max():.3f}"
                )
            else:
                print("\nBoth-tags-visible section")
                print("frames with both tags visible: 0")
        else:
            print("\nNo apple poses were captured.")


if __name__ == "__main__":
    main()
