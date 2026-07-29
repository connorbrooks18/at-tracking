import argparse
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
    median_pose_4x4,
    pose_dict_to_transform,
    transform_pose_to_base,
)
from real_robot_exps.static_constants import CAMERA_TO_BASE_4X4_DEFAULT

import annotate
from DataCollector import DataCollector
import Tracker

# tag length in meters
TAG_SIZE_M = 0.0170
REFERENCE_TAG_ID = 1

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
    reference_tag_base_samples: list[np.ndarray] = []

    #relationship between tags and offsets

    # second apple offset is for tag only 45 degrees from it
    apple_offsets = [
        {"pos": [0, 0.0, .11], "rot": [[-0.7071, 0, -0.7071], [0, 1, 0], [0.7071, 0, -0.7071]]},
        # Tag 6 uses the same apple convention as read_apple_pose.py: rotate
        # -45 degrees about the tag's y axis so x/y stay consistent and z points
        # the correct way in the right-handed frame.
        {"pos": [.085, 0.00, 0.0], "rot": [[0.7071, 0, -0.7071], [0, 1, 0], [0.7071, 0, 0.7071]]},
    ]
    apple = Tracker.Tracker("Apple", ids=(7,0), id_offsets=apple_offsets)

    spur_offsets = [{"pos": [0.0, 0.035, 0.03], "rot": np.eye(3)},{"pos": [0.0, 0.035, 0.03], "rot": [[0, 0, -1], [0, 1,  0], [1, 0,  0]]},{"pos": [0.0, 0.035, 0.03], "rot": [[0, 0, 1], [0, 1,  0], [-1, 0,  0]]}]
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
    reference_tag_to_base = median_pose_4x4(reference_tag_base_samples)

    tracking_metadata = {
        "capture_start_timestamp": capture_start,
        "reference_tag_id": pipeline.reference_id,
        "reference_tag_enabled": bool(args.use_reference_frame),
        "reference_tag_is_fruiting_base": bool(args.use_reference_frame),
        "coordinate_frame": "franka_base_o",
        "camera_to_base_4x4_used": CAMERA_TO_BASE_4X4_DEFAULT.tolist(),
        "reference_tag_to_base_4x4_used": reference_tag_to_base.tolist() if reference_tag_to_base is not None else None,
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

    try:
        while not stop_requested.is_set():
            frames      = pipeline.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            _, tag_dict = pipeline.process_frame(frame)
            pipeline.annotate_frame(frame, tag_dict)
            frame_timestamp = time.time()

            if REFERENCE_TAG_ID in tag_dict:
                ref_tag = tag_dict[REFERENCE_TAG_ID]
                reference_tag_base_samples.append(
                    transform_pose_to_base(
                        {
                            "pos": np.asarray(ref_tag.pose_t, dtype=np.float64).reshape(3),
                            "rot": np.asarray(ref_tag.pose_R, dtype=np.float64),
                        },
                        camera_to_base=CAMERA_TO_BASE_4X4_DEFAULT,
                    )
                )

            for tracker in trackers:
                pose_base = pipeline.pose_for_storage(tracker)
                if pose_base is None:
                    continue
                x, y, z = pose_base[:3, 3]
                quat = R.from_matrix(pose_base[:3, :3]).as_quat()
                dataCollector.update(frame_timestamp, tracker.name, x, y, z, quat[0], quat[1], quat[2], quat[3])

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
        pipeline.pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
