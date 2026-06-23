# Tracking with realsense camera to april tags.
# Tracker object tracks an object with one or more april tags (averages position)
#
#
import os
import numpy as np
import pyrealsense2 as rs




def _make_transform(rotation, translation):
    """Build a 4x4 homogeneous transform from a 3x3 rotation and 3-vector translation."""
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float32)
    transform[:3, 3]  = np.asarray(translation, dtype=np.float32).reshape(3)
    return transform


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

        self.offsets = {
            tag_id: _make_transform(offset["rot"], offset["pos"])
            for tag_id, offset in zip(ids, id_offsets)
        }

    def updatePose(self, tags_in_ref):
		# tags in ref gives list of tag poses in reference frame

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
            self.pose = self.pose
            return self.pose

        self.pose = {
            "pos": np.mean(positions, axis=0),
            "rot": first_rot,
        }
        return self.pose


