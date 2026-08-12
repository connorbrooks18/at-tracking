import argparse
import json
import signal
import sys
from threading import Event
import time
from pathlib import Path

import numpy as np
import cv2
import pyrealsense2 as rs
from pupil_apriltags import Detector # type: ignore
from scipy.spatial.transform import Rotation as R

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from real_robot_exps.frame_transforms import (
    make_transform,
    pose_dict_to_transform,
    transform_pose_to_base,
)
from real_robot_exps.static_constants import CAMERA_TO_BASE_4X4_DEFAULT

import annotate
from DataCollector import DataCollector
import Tracker

# tag length in meters
TAG_SIZE_M = 0.0170

class Detecting:
    """RealSense capture, AprilTag detection, and base-frame tracking pipeline."""

    def __init__(
        self,
        allowed_ids,
        reference_id,
        trackers,
        decision_margin=5,
        use_reference_frame: bool = False,
    ):
        """
        Args:
            allowed_ids:     Tag IDs to accept from the detector.
            reference_id:    Tag ID that defines the reference frame.
                             Should be fixed and reliably visible at all times.
            trackers:        Tracker instances to update each frame.
            decision_margin: Minimum detector confidence (higher = stricter).
            use_reference_frame: When True, draw tracker overlays in the
                reference-tag frame if the reference tag is visible. Saved
                parquet rows are still written in the Franka base frame.
        """
        self.allowed_ids         = allowed_ids
        self.reference_id        = reference_id
        self.trackers            = trackers
        self.decision_margin     = decision_margin
        self.use_reference_frame = bool(use_reference_frame)
        self.last_reference_pose = None  # persists across brief occlusions
        self.record_video = False
        self.record_video_path = None
        self.video_writer = None

        self._init_camera()
        self.detector = Detector(families="tag36h11",
                                 quad_decimate=1.0,
                                 nthreads=12,
                                 refine_edges=1,
                                 quad_sigma=0.2,
                                 decode_sharpening=1.0
                        )

    def _init_camera(self):
        """Start the RealSense color stream and cache intrinsics."""
        self.pipeline = rs.pipeline()
        config        = rs.config()
        self.camera_fps = 15 # 6, 15, 30
        self.camera_width = 1280
        self.camera_height = 720
        self.camera_exposure = 100
        config.enable_stream(
            rs.stream.color,
            self.camera_width,
            self.camera_height,
            rs.format.bgr8,
            self.camera_fps,
        )
        profile = self.pipeline.start(config)
        color_sensor = profile.get_device().query_sensors()[1]
        color_sensor.set_option(rs.option.enable_auto_exposure, 0)

        # Set a low manual exposure value (e.g., 70-150 microseconds)
        color_sensor.set_option(rs.option.exposure, self.camera_exposure)

        intrinsics = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self.camera_params = (
            intrinsics.fx, intrinsics.fy,
            intrinsics.ppx, intrinsics.ppy,
        )
        self.K = np.array(
            [[intrinsics.fx, 0,             intrinsics.ppx],
             [0,             intrinsics.fy, intrinsics.ppy],
             [0,             0,             1             ]],
            dtype=np.float32,
        )
        # D435 distortion is negligible at typical working distances.
        self.dist_coeffs = np.zeros(5, dtype=np.float32)

    def _open_video_writer(self, output_path: Path, frame_shape) -> None:
        if self.video_writer is not None:
            return
        height, width = frame_shape[:2]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            float(self.camera_fps),
            (int(width), int(height)),
        )
        if not self.video_writer.isOpened():
            self.video_writer = None
            raise RuntimeError(f"Could not open video writer for {output_path}")
        self.record_video_path = output_path

    def write_recorded_frame(self, frame):
        if not self.record_video:
            return
        if self.record_video_path is None:
            raise RuntimeError("record_video_path must be set before recording frames")
        if self.video_writer is None:
            self._open_video_writer(self.record_video_path, frame.shape)
        self.video_writer.write(frame)

    def close(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def process_frame(self, frame):
        """Detect tags, update reference frame, refresh tracker poses.

        Returns:
            tags_out : dict  tag_id -> {'pos': (3,), 'rot': (3,3)}
            tag_dict    : dict  tag_id -> raw Detection (camera frame)
        """
        tag_dict = self._detect_valid_tags(frame)

        if self.use_reference_frame and self.reference_id in tag_dict:
            self.last_reference_pose = tag_dict[self.reference_id]

        tags_out = self._transform_to_output_frame(tag_dict)
        for tracker in self.trackers:
            tracker.updatePose(tags_out)

        return tags_out, tag_dict

    def annotate_frame(self, frame, tag_dict):
        """Draw debug overlays — implementation lives in annotate.py."""
        annotate.annotate_frame(
            frame,
            tag_dict,
            last_reference_pose=self.last_reference_pose,
            trackers=self.trackers,
            camera_matrix=self.K,
            dist_coeffs=self.dist_coeffs,
            use_reference_frame=self.use_reference_frame,
        )

    def _detect_valid_tags(self, frame):
        """Run AprilTag detection and filter by ID and decision margin."""
        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw_tags = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=TAG_SIZE_M,
        )
        return {
            tag.tag_id: tag
            for tag in raw_tags
            if tag.decision_margin > self.decision_margin
            and tag.tag_id in self.allowed_ids
        }

    def _transform_to_output_frame(self, tag_dict):
        """Return tag poses in either camera frame or reference-tag frame."""
        if not self.use_reference_frame:
            tags_in_camera = {}
            for tag_id, tag in tag_dict.items():
                tags_in_camera[tag_id] = {
                    "pos": np.asarray(tag.pose_t, dtype=np.float32).reshape(3),
                    "rot": np.asarray(tag.pose_R, dtype=np.float32),
                }
            return tags_in_camera

        if self.last_reference_pose is None:
            return {}

        R_ref_inv = self.last_reference_pose.pose_R.T
        t_ref = self.last_reference_pose.pose_t

        tags_in_ref = {}
        for tag_id, tag in tag_dict.items():
            tags_in_ref[tag_id] = {
                "pos": (R_ref_inv @ (tag.pose_t - t_ref)).flatten().astype(np.float32),
                "rot": (R_ref_inv @ tag.pose_R).astype(np.float32),
            }
        return tags_in_ref

    def pose_for_storage(self, tracker):
        """Return a tracker pose as a 4x4 transform in the Franka base frame."""
        if tracker.pose is None:
            return None

        if self.use_reference_frame and self.last_reference_pose is not None:
            cam_T_ref = make_transform(
                np.asarray(self.last_reference_pose.pose_R, dtype=np.float64),
                np.asarray(self.last_reference_pose.pose_t, dtype=np.float64).reshape(3),
            )
            ref_T_obj = pose_dict_to_transform(tracker.pose)
            cam_pose = cam_T_ref @ ref_T_obj
            pose_camera = {"pos": cam_pose[:3, 3], "rot": cam_pose[:3, :3]}
            return transform_pose_to_base(pose_camera)

        return transform_pose_to_base(tracker.pose)



# USAGE BELOW

def main():
    parser = argparse.ArgumentParser(description="Record AprilTag poses with Unix timestamps.")
    parser.add_argument("--output", default="output.parquet", help="Raw tracking Parquet path")
    parser.add_argument(
        "--record",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record the full camera feed to an .mp4 alongside the tracking output",
    )
    parser.add_argument(
        "--record-output",
        default=None,
        help="Optional MP4 path for --record; defaults to the tracking output stem with .mp4",
    )
    parser.add_argument("--snapshot-request", default=None, help="File to poll for one-shot snapshot requests")
    parser.add_argument("--snapshot-output", default=None, help="JSON path for the requested one-shot snapshot")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True,
                        help="Disable OpenCV GUI windows and run headless (default: true)")
    parser.add_argument(
        "--use-reference-frame",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw the live overlay in the reference-tag frame instead of the camera frame",
    )
    args = parser.parse_args()

    stop_requested = Event()

    def _request_stop(signum, frame):  # noqa: ARG001
        stop_requested.set()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    capture_start = time.time()

    #relationship between tags and offsets

    # second apple offset is for tag only 45 degrees from it
    apple_offsets = [
        {"pos": [.02, 0.0, .11], "rot": [[-0.7071, 0, -0.7071], [0, 1, 0], [0.7071, 0, -0.7071]]},
        # Tag 6 uses the same apple convention as read_apple_pose.py: rotate
        # -45 degrees about the tag's y axis so x/y stay consistent and z points
        # the correct way in the right-handed frame.
        {"pos": [.085, 0.00, 0.0], "rot": [[0.7071, 0, -0.7071], [0, 1, 0], [0.7071, 0, 0.7071]]},
    ]
    apple = Tracker.Tracker("Apple", ids=(7,0), id_offsets=apple_offsets)

    spur_down_offset = 0.035 # 0.035 is 'normal' for non-small spurs. 0.025 if going on apple directly (not on spur)
    spur_offsets = [{"pos": [0.0, spur_down_offset, 0.03], "rot": np.eye(3)},{"pos": [0.0, spur_down_offset, 0.03], "rot": [[0, 0, -1], [0, 1,  0], [1, 0,  0]]},{"pos": [0.0, spur_down_offset, 0.03], "rot": [[0, 0, 1], [0, 1,  0], [-1, 0,  0]]}]
    spur = Tracker.Tracker("Spur", ids=(3,4,5,), id_offsets=spur_offsets)

    branch_offsets = [
        {
            "pos": [0, -0.015, 0.03],
            "rot": np.eye(3),
        },
    ]
    branch = Tracker.Tracker("Branch", ids=(2,), id_offsets=branch_offsets)

    trackers = [branch, spur, apple] # , apple


    pipeline = Detecting(
        allowed_ids=(0, 1, 2, 3, 4, 5, 6, 7, 8),
        reference_id=1,
        trackers=trackers,
        decision_margin=3,
        use_reference_frame=bool(args.use_reference_frame),
    )
    pipeline.record_video = bool(args.record)
    if pipeline.record_video:
        record_output = Path(args.record_output) if args.record_output else Path(args.output).with_suffix(".mp4")
        pipeline.record_video_path = record_output
        print(f"Recording camera feed to {record_output}")
    tracking_metadata = {
        "capture_start_timestamp": capture_start,
        "coordinate_frame": "franka_base_o",
        "camera_to_base_4x4_used": CAMERA_TO_BASE_4X4_DEFAULT.tolist(),
        "position_unit": "m",
        "quaternion_order": "xyzw",
        "tag_family": "tag36h11",
        "tag_size_m": TAG_SIZE_M,
        "allowed_tag_ids": list(pipeline.allowed_ids),
        "decision_margin_threshold": pipeline.decision_margin,
        "camera": {
            "stream": "color",
            "width": pipeline.camera_width,
            "height": pipeline.camera_height,
            "fps": pipeline.camera_fps,
            "manual_exposure": pipeline.camera_exposure,
            "intrinsics_matrix": pipeline.K.tolist(),
            "distortion_coefficients": pipeline.dist_coeffs.tolist(),
        },
        "video_recording_enabled": bool(args.record),
        "video_recording_path": str(pipeline.record_video_path) if pipeline.record_video_path else None,
        "tracker_names": [tracker.name for tracker in trackers],
        "tracker_tag_ids": {
            tracker.name: [int(tag_id) for tag_id in tracker.ids]
            for tracker in trackers
        },
        "tracker_tag_to_object_transforms": {
            tracker.name: {
                str(tag_id): transform.tolist()
                for tag_id, transform in tracker.offsets.items()
            }
            for tracker in trackers
        },
        "topology": {
            "node_order": ["Branch", "Spur", "Apple"],
            "woody_part_names": ["Branch", "Spur", "Apple"],
            "start_nodes": ["Branch", "Spur"],
            "end_nodes": ["Branch", "Spur", "Apple"],
        },
    }
    dataCollector = DataCollector(metadata=tracking_metadata)

    def _write_requested_snapshot(poses, timestamp):
        if not args.snapshot_request or not args.snapshot_output or not Path(args.snapshot_request).exists():
            return
        if not all(name in poses for name in ("Branch", "Spur", "Apple")):
            return
        positions = {name: pose[:3, 3].astype(float).tolist() for name, pose in poses.items()}
        starts = np.asarray([positions["Branch"], positions["Branch"], positions["Spur"]], dtype=float)
        ends = np.asarray([positions["Spur"], positions["Apple"], positions["Apple"]], dtype=float)
        snapshot = {
            "timestamp": float(timestamp),
            "camera_frame_count": 1,
            "camera_to_base_4x4": CAMERA_TO_BASE_4X4_DEFAULT.tolist(),
            "apple_pos": positions["Apple"],
            "apple_pose_4x4": poses["Apple"].reshape(-1).astype(float).tolist(),
            "apple_quat_xyzw": R.from_matrix(poses["Apple"][:3, :3]).as_quat().astype(float).tolist(),
            "branch_pos": positions["Branch"],
            "branch_pose_4x4": poses["Branch"].reshape(-1).astype(float).tolist(),
            "spur_pos": positions["Spur"],
            "spur_pose_4x4": poses["Spur"].reshape(-1).astype(float).tolist(),
            "woody_part_start_pos": starts.reshape(-1).tolist(),
            "woody_part_end_pos": ends.reshape(-1).tolist(),
            "woody_bending_angles": [0.0, 0.0, 0.0],
            "source": "running_detector_snapshot_request",
        }
        output_path = Path(args.snapshot_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(snapshot), encoding="utf-8")
        temporary_path.replace(output_path)
        Path(args.snapshot_request).unlink(missing_ok=True)

    try:
        while not stop_requested.is_set():
            frames      = pipeline.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            _, tag_dict = pipeline.process_frame(frame)
            recorded_frame = frame.copy() if pipeline.record_video else None
            pipeline.annotate_frame(frame, tag_dict)
            if recorded_frame is not None:
                pipeline.write_recorded_frame(recorded_frame)
            frame_timestamp = time.time()

            current_poses = {}

            for tracker in trackers:
                pose_base = pipeline.pose_for_storage(tracker)
                if pose_base is None:
                    continue
                current_poses[tracker.name] = pose_base.copy()
                x, y, z = pose_base[:3, 3]
                quat = R.from_matrix(pose_base[:3, :3]).as_quat()
                dataCollector.update(frame_timestamp, tracker.name, x, y, z, quat[0], quat[1], quat[2], quat[3])

            _write_requested_snapshot(current_poses, frame_timestamp)

            if not args.headless:
                cv2.imshow("RealSense Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_requested.set()
    finally:
        #dataCollector.print()
        dataCollector.dump(
            args.output,
            metadata={
                "capture_end_timestamp": time.time(),
                "row_count": len(dataCollector.rows),
                "stop_requested": bool(stop_requested.is_set()),
            },
        )
        print(f"Wrote tracking data to {args.output}")
        pipeline.close()
        pipeline.pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
