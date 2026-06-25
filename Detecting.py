from pupil_apriltags import Detector
import annotate
import pyrealsense2 as rs
import cv2
from DataCollector import DataCollector
import Tracker
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

# tag length in meters
TAG_SIZE_M = 0.018

class Detecting:
    """RealSense capture, AprilTag detection, and reference-frame tracking pipeline."""

    def __init__(self, allowed_ids, reference_id, trackers, decision_margin=5):
        """
        Args:
            allowed_ids:     Tag IDs to accept from the detector.
            reference_id:    Tag ID that defines the reference frame.
                             Should be fixed and reliably visible at all times.
            trackers:        Tracker instances to update each frame.
            decision_margin: Minimum detector confidence (higher = stricter).
        """
        self.allowed_ids         = allowed_ids
        self.reference_id        = reference_id
        self.trackers            = trackers
        self.decision_margin     = decision_margin
        self.last_reference_pose = None  # persists across brief occlusions

        self._init_camera()
        self.detector = Detector(families="tag36h11", nthreads=4)

    def _init_camera(self):
        """Start the RealSense color stream and cache intrinsics."""
        self.pipeline = rs.pipeline()
        config        = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)

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
            tags_in_ref : dict  tag_id -> {'pos': (3,), 'rot': (3,3)}
            tag_dict    : dict  tag_id -> raw Detection (camera frame)
        """
        tag_dict = self._detect_valid_tags(frame)

        if self.reference_id in tag_dict:
            self.last_reference_pose = tag_dict[self.reference_id]

        if self.last_reference_pose is None:
            return {}, tag_dict

        tags_in_ref = self._transform_to_reference(tag_dict)
        for tracker in self.trackers:
            tracker.updatePose(tags_in_ref)

        return tags_in_ref, tag_dict

    def annotate_frame(self, frame, tag_dict):
        """Draw debug overlays — implementation lives in annotate.py."""
        annotate.annotate_frame(
            frame,
            tag_dict,
            last_reference_pose=self.last_reference_pose,
            trackers=self.trackers,
            camera_matrix=self.K,
            dist_coeffs=self.dist_coeffs,
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

    def _transform_to_reference(self, tag_dict):
        """Express every detected tag pose in the reference tag's frame.

        For each tag:
            pos_in_ref = R_ref^T @ (pos_in_cam - pos_ref_in_cam)
            rot_in_ref = R_ref^T @ R_tag
        """
        R_ref_inv     = self.last_reference_pose.pose_R.T  # R^T = R^-1 for rotation matrices
        t_ref         = self.last_reference_pose.pose_t

        tags_in_ref = {}
        for tag_id, tag in tag_dict.items():
            tags_in_ref[tag_id] = {
                "pos": (R_ref_inv @ (tag.pose_t - t_ref)).flatten().astype(np.float32),
                "rot": (R_ref_inv @ tag.pose_R).astype(np.float32),
            }
        return tags_in_ref



# USAGE BELOW

def main():
    start = time.time()

    #relationship between tags and offsets
    apple_offsets = [{"pos": [0.0, -0.05, 0.0], "rot": np.eye(3),},]
	        #{"pos": [0, 0.0, 0.05], "rot": [[0, 0, -1], [0, 1,  0], [1, 0,  0]]}]
    apple = Tracker.Tracker("Apple", ids=(3,), id_offsets=apple_offsets)

    spur_offsets = [{"pos": [0.0, 0.01, -0.03], "rot": np.eye(3)},]
    spur = Tracker.Tracker("Spur", ids=(5,), id_offsets=spur_offsets)

    branch_offsets = [
        {
            "pos": [0, -0.01, -0.03],
            "rot": np.eye(3),
        },
    ]
    branch = Tracker.Tracker("Branch", ids=(4,), id_offsets=branch_offsets)

    trackers = [apple, branch, spur]


    pipeline = Detecting(
        allowed_ids=(0, 1, 2, 3, 4, 5),
        reference_id=2,
        trackers=trackers,
    )

    dataCollector = DataCollector()

    try:
        while True:
            frames      = pipeline.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            _, tag_dict = pipeline.process_frame(frame)
            pipeline.annotate_frame(frame, tag_dict)

            for tracker in trackers:
                x, y, z = tracker.pose['pos'] if tracker.pose is not None and tracker.pose['pos'] is not None else (0, 0, 0)
                quat = R.from_matrix(tracker.pose['rot'], assume_valid=False).as_quat() if tracker.pose is not None and tracker.pose['rot'] is not None else (0, 0, 0, 1) # returns [x, y, z, w]
                dataCollector.update(time.time() - start, tracker.name, x, y, z, quat[0], quat[1], quat[2], quat[3])

            cv2.imshow("RealSense Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        #dataCollector.print()
        dataCollector.dump()
        pipeline.pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
