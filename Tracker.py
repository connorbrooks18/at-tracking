"""AprilTag object tracking with an Intel RealSense color camera.

Each Tracker fuses one or more AprilTags into a single object pose, expressed
in a persistent reference frame anchored to a designated reference tag.
"""

import os

import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector

import annotate

# Physical tag edge length (meters) used for pose estimation.
TAG_SIZE_M = 0.0335


def _make_transform(rotation, translation):
    """Build a 4x4 homogeneous transform from a 3x3 rotation and 3-vector translation."""
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float32)
    transform[:3, 3] = np.asarray(translation, dtype=np.float32).reshape(3)
    return transform


def _orthogonalize_rotation(rotation):
    """Project a near-rotation matrix onto SO(3) via SVD."""
    u, _, vt = np.linalg.svd(rotation)
    rotation_ortho = u @ vt

    # Reject reflections so the result stays a proper rotation.
    if np.linalg.det(rotation_ortho) < 0:
        vt[2, :] *= -1
        rotation_ortho = u @ vt

    return rotation_ortho


class Tracker:
    """Tracks a rigid object using one or more AprilTags on its surface."""

    def __init__(self, name, ids, id_offsets):
        """
        Args:
            name: Display label for this object.
            ids: AprilTag IDs attached to this object.
            id_offsets: Per-tag calibration from object frame to tag frame,
                        each entry with keys ``pos`` and ``rot``.
        """
        self.name = name
        self.ids = ids
        self.pose = None  # {'pos': (3,), 'rot': (3, 3)} in reference frame

        # Store T_tag_obj = inv(T_obj_tag) so pose estimation is a single multiply.
        self.offsets = {
            tag_id: np.linalg.inv(_make_transform(offset["rot"], offset["pos"]))
            for tag_id, offset in zip(ids, id_offsets)
        }

    def updatePose(self, tags_in_ref):
        """Estimate object pose by averaging transforms from visible tags."""
        positions, rotations = [], []

        for tag_id in self.ids:
            if tag_id not in tags_in_ref:
                continue

            tag = tags_in_ref[tag_id]
            transform_ref_tag = _make_transform(tag["rot"], tag["pos"])
            transform_ref_obj = transform_ref_tag @ self.offsets[tag_id]

            positions.append(transform_ref_obj[:3, 3])
            rotations.append(transform_ref_obj[:3, :3])

        if not positions:
            self.pose = None
            return None

        self.pose = {
            "pos": np.mean(positions, axis=0),
            "rot": _orthogonalize_rotation(np.mean(rotations, axis=0)),
        }
        return self.pose


class Detecting:
    """RealSense capture, AprilTag detection, and reference-frame tracking pipeline."""

    def __init__(self, allowed_ids, reference_id, trackers, decision_margin=5):
        """
        Args:
            allowed_ids: Tag IDs to accept from the detector.
            reference_id: Tag ID that defines the world/reference frame.
            trackers: Tracker instances updated each frame.
            decision_margin: Minimum detector confidence for a valid tag.
        """
        self.allowed_ids = allowed_ids
        self.reference_id = reference_id
        self.trackers = trackers
        self.decision_margin = decision_margin

        # Last seen pose of the reference tag in the camera frame.
        self.last_reference_pose = None

        self._init_camera()
        self.detector = Detector(families="tag36h11", nthreads=4)

    def _init_camera(self):
        """Start the RealSense color stream and cache intrinsics for projection."""
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)

        intrinsics = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )

        self.camera_params = (
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.ppx,
            intrinsics.ppy,
        )
        self.K = np.array(
            [
                [intrinsics.fx, 0, intrinsics.ppx],
                [0, intrinsics.fy, intrinsics.ppy],
            ],
            dtype=np.float32,
        )
        self.dist_coeffs = np.zeros(5, dtype=np.float32)

    def process_frame(self, frame):
        """Detect tags, update the reference frame, and refresh tracker poses."""
        tag_dict = self._detect_valid_tags(frame)

        # Keep the last reference pose so tracking survives brief occlusions.
        if self.reference_id in tag_dict:
            self.last_reference_pose = tag_dict[self.reference_id]

        if self.last_reference_pose is None:
            return {}, tag_dict

        tags_in_ref = self._transform_to_reference(tag_dict)
        for tracker in self.trackers:
            tracker.updatePose(tags_in_ref)

        return tags_in_ref, tag_dict

    def annotate_frame(self, frame, tag_dict):
        """Draw debug overlays; implementation lives in annotate.py."""
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        """Express every detected tag pose relative to the reference tag frame."""
        rotation_ref_inv = self.last_reference_pose.pose_R.T
        translation_ref = self.last_reference_pose.pose_t

        tags_in_ref = {}
        for tag_id, tag in tag_dict.items():
            tags_in_ref[tag_id] = {
                "pos": (rotation_ref_inv @ (tag.pose_t - translation_ref))
                .flatten()
                .astype(np.float32),
                "rot": (rotation_ref_inv @ tag.pose_R).astype(np.float32),
            }
        return tags_in_ref


def main():
    # Suppress noisy Qt font warnings when OpenCV creates its display window.
    os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false"

    apple_offsets = [
        {"pos": [0.0, 0.0, -0.05], "rot": np.eye(3)},
        {
            "pos": [-0.05, 0.0, 0.0],
            "rot": [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
        },
    ]
    apple = Tracker("Apple", ids=(4, 5), id_offsets=apple_offsets)

    branch_offsets = [{"pos": [0.08, 0.0, -0.015], "rot": np.eye(3)}]
    branch = Tracker("Branch", ids=(3,), id_offsets=branch_offsets)

    trackers = [apple, branch]
    pipeline = Detecting(
        allowed_ids=(0, 1, 2, 3, 4, 5),
        reference_id=2,
        trackers=trackers,
    )

    try:
        while True:
            frames = pipeline.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            _, tag_dict = pipeline.process_frame(frame)
            pipeline.annotate_frame(frame, tag_dict)

            cv2.imshow("RealSense Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipeline.pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
