"""AprilTag object tracking with an Intel RealSense color camera.

Each Tracker fuses one or more AprilTags into a single object pose, expressed
in a persistent reference frame anchored to a designated reference tag.

═══════════════════════════════════════════════════════════════════════════════
OFFSET CONVENTION  (id_offsets in Tracker)
═══════════════════════════════════════════════════════════════════════════════

Each offset describes how to get from a TAG to its OBJECT POINT OF INTEREST.

  pos : translation from tag center to object point of interest,
        expressed in TAG coordinates.
        Measure with calipers: "if I'm standing at the tag center,
        looking out of the tag face (+Z), where is the object?"

  rot : rotation matrix from OBJECT frame to TAG frame.
        i.e.  v_tag = rot @ v_object
        If the tag is mounted flush and upright (tag axes == object axes),
        use np.eye(3).

TAG COORDINATE FRAME (pupil-apriltags / OpenCV convention):
  +X  right  (when facing the tag)
  +Y  down   (when facing the tag)
  +Z  into the tag face, away the camera

REFERENCE FRAME:
  Same convention — +Z into scene, +X right, +Y down —
  anchored to the reference tag's pose.

SIGN CHECK
  With only one tag visible, move the object toward the camera.
  The reported object pos[2] should decrease (getting closer = smaller Z).
  If axes feel swapped, transpose rot.
═══════════════════════════════════════════════════════════════════════════════
"""

import os

import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector

import annotate

# Physical tag edge length (metres): black border outer-edge to outer-edge.
# Do NOT include the white quiet zone.
TAG_SIZE_M = 0.0335


def _make_transform(rotation, translation):
    """Build a 4x4 homogeneous transform from a 3x3 rotation and 3-vector translation."""
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float32)
    transform[:3, 3]  = np.asarray(translation, dtype=np.float32).reshape(3)
    return transform


def _orthogonalize_rotation(rotation):
    """Project a near-rotation matrix onto SO(3) via SVD."""
    u, _, vt = np.linalg.svd(rotation)
    rotation_ortho = u @ vt
    if np.linalg.det(rotation_ortho) < 0:   # reject reflections
        vt[2, :] *= -1
        rotation_ortho = u @ vt
    return rotation_ortho


class Tracker:
    """Tracks a rigid object using one or more AprilTags on its surface."""

    def __init__(self, name, ids, id_offsets):
        """
        Args:
            name:       Display label for this object.
            ids:        AprilTag IDs attached to this object.
            id_offsets: Per-tag offset dicts with keys:
                            'pos' — translation from tag center to object,
                                    in tag coordinates (metres).
                            'rot' — rotation from object frame to tag frame.
                        See module docstring for full convention and examples.
        """
        self.name = name
        self.ids  = ids
        self.pose = None  # {'pos': (3,), 'rot': (3,3)} in reference frame, or None

        # T_tag_obj: takes a point in tag frame to object frame.
        # Built directly from offset — no inversion needed given the convention above.
        self.offsets = {
            tag_id: _make_transform(offset["rot"], offset["pos"])
            for tag_id, offset in zip(ids, id_offsets)
        }

    def updatePose(self, tags_in_ref):
        """Estimate object pose by averaging position estimates from visible tags.

        T_ref_obj = T_ref_tag @ T_tag_obj

        Rotation is taken from the first visible tag only — averaging rotations
        from tags with different orientations is not meaningful.
        """
        positions = []
        first_rot = None

        for tag_id in self.ids:
            if tag_id not in tags_in_ref:
                continue

            tag           = tags_in_ref[tag_id]
            T_ref_tag     = _make_transform(tag["rot"], tag["pos"])
            T_ref_obj     = T_ref_tag @ self.offsets[tag_id]

            positions.append(T_ref_obj[:3, 3])
            if first_rot is None:
                first_rot = T_ref_obj[:3, :3]

        if not positions:
            self.pose = None
            return None

        self.pose = {
            "pos": np.mean(positions, axis=0),
            "rot": first_rot,
        }
        return self.pose


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


def main():
    os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false"

    # ── Tracked objects ───────────────────────────────────────────────────────
    # See module docstring for offset convention.
    #
    apple_offsets = [
        {
            "pos": [0.0, 0.0, 0.05],       # 5 cm behind tag face along tag -Z
            "rot": np.eye(3),               # tag axes == object axes
        },
        {
            "pos": [0, 0.0, 0.05],       # 5 cm to the right in tag coords
            "rot": [[0, 0, -1],             # object +X → tag -Z
                    [0, 1,  0],             # object +Y → tag +Y
                    [1, 0,  0]],            # object +Z → tag +X
        },
    ]
    apple = Tracker("Apple", ids=(4, 5), id_offsets=apple_offsets)

    # BRANCH — single tag, 10 cm along branch and 1.5 cm behind face.
    branch_offsets = [
        {
            "pos": [-0.07, 0.0, 0.015],
            "rot": np.eye(3),
        },
    ]
    branch = Tracker("Branch", ids=(3,), id_offsets=branch_offsets)

    trackers = [apple, branch]
    pipeline = Detecting(
        allowed_ids=(0, 1, 2, 3, 4, 5),
        reference_id=2,
        trackers=trackers,
    )

    try:
        while True:
            frames      = pipeline.pipeline.wait_for_frames()
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
