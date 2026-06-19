import cv2
import sys
import os
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector

class Tracker:
    def __init__(self, name, ids, id_offsets):
        self.name = name
        self.ids = ids
        self.pose = None  # {'pos': [x,y,z], 'rot': R} in reference frame
        
        # Directly treating offsets as Tag -> Object to match your measurement intuition
        self.offsets = {}
        for tag_id, offset in zip(ids, id_offsets):
            R_offset = np.array(offset['rot'])
            t_offset = np.array(offset['pos'])
            
            T = np.eye(4)
            T[:3, :3] = R_offset
            T[:3, 3] = t_offset
            self.offsets[tag_id] = T

    def updatePose(self, tags_dict):
        estimated_positions = []
        estimated_rotations = []

        for tag_id in self.ids:
            if tag_id in tags_dict:
                R_tag = tags_dict[tag_id]['rot']
                t_tag = tags_dict[tag_id]['pos']
                
                T_ref_tag = np.eye(4)
                T_ref_tag[:3, :3] = R_tag
                T_ref_tag[:3, 3] = t_tag

                # T_ref_obj = T_ref_tag @ T_tag_to_obj
                T_tag_obj = self.offsets[tag_id]
                T_ref_obj = T_ref_tag @ T_tag_obj
                
                estimated_positions.append(T_ref_obj[:3, 3])
                estimated_rotations.append(T_ref_obj[:3, :3])

        if not estimated_positions:
            self.pose = None
            return None

        avg_pos = np.mean(estimated_positions, axis=0)
        avg_R_approx = np.mean(estimated_rotations, axis=0)
        U, _, Vt = np.linalg.svd(avg_R_approx)
        avg_rot = U @ Vt

        self.pose = {'pos': avg_pos, 'rot': avg_rot}
        return self.pose   


class Detecting:
    def __init__(self, allowed_ids, reference_id, trackers, decision_margin=10):
        self.allowed_ids = allowed_ids
        self.reference_id = reference_id
        self.trackers = trackers
        self.decision_margin = decision_margin
        self.last_reference_pose = None

        # Initialize RealSense Pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)

        # Extract Intrinsics
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.camera_params = (intr.fx, intr.fy, intr.ppx, intr.ppy)
        
        # Build Camera Matrices for cv2.projectPoints
        self.K = np.array([[intr.fx, 0, intr.ppx],
                           [0, intr.fy, intr.ppy],
                          ], dtype=np.float32)
        self.dist_coeffs = np.zeros(5) # RealSense streams are already rectified

        # Initialize AprilTag Detector
        self.detector = Detector(families="tag36h11", nthreads=4)

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect tags with pose estimation
        raw_tags = self.detector.detect(gray, estimate_tag_pose=True, 
                                        camera_params=self.camera_params, tag_size=0.0335)
        
        # Filter detections
        valid_tags = [t for t in raw_tags if t.decision_margin > self.decision_margin and t.tag_id in self.allowed_ids]
        tag_dict = {t.tag_id: t for t in valid_tags}

        # Handle master reference persistence
        if self.reference_id in tag_dict:
            self.last_reference_pose = tag_dict[self.reference_id]

        if self.last_reference_pose is None:
            return {}, tag_dict # Can't build relative poses without a anchor reference

        # Transform all tags to Master Reference Frame
        R_ref_inv = self.last_reference_pose.pose_R.T
        t_ref = self.last_reference_pose.pose_t
        
        tags_in_ref = {}
        for tag_id, tag in tag_dict.items():
            t_rel = R_ref_inv @ (tag.pose_t - t_ref)
            R_rel = R_ref_inv @ tag.pose_R
            tags_in_ref[tag_id] = {'pos': t_rel.flatten(), 'rot': R_rel}

        # Update tracker objects
        for tracker in self.trackers:
            tracker.updatePose(tags_in_ref)

        return tags_in_ref, tag_dict

    def annotate_frame(self, frame, tags_in_ref, tag_dict):
        # 1. Draw detected AprilTag boundaries
        for tag_id, tag in tag_dict.items():
            corners = tag.corners.astype(int)
            cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {tag_id}", tuple(corners), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if self.last_reference_pose is None:
            return

        # 2. Draw Tracker Object Centers (Project 3D Reference space -> 2D Camera space)
        # T_cam_ref matrix brings points from Master Ref frame to Camera frame
        R_ref_cam = self.last_reference_pose.pose_R
        t_ref_cam = self.last_reference_pose.pose_t
        rvec_ref_cam, _ = cv2.Rodrigues(R_ref_cam)

        for tracker in self.trackers:
            if tracker.pose is not None:
                # 3D position of object in Master Reference frame
                obj_pos_ref = tracker.pose['pos'].reshape(1, 3)

                # Project 3D point in master reference frame to 2D image coordinates
                img_pts, _ = cv2.projectPoints(obj_pos_ref, rvec_ref_cam, t_ref_cam, self.K, self.dist_coeffs)
                center_2d = tuple(img_pts.astype(int))

                # Visualizing tracking dot and name
                cv2.circle(frame, center_2d, 7, (0, 0, 255), -1)
                cv2.putText(frame, tracker.name, (center_2d + 10, center_2d), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def main():
    # Setup dummy offsets for example (Tag 0 is right at the center of 'apple')
    apple_offsets = [{'pos': [0.0, 0.0, 0.0], 'rot': np.eye(3)}]
    apple = Tracker("Apple", ids=(0,), id_offsets=apple_offsets)
    
    trackers = [apple]
    detector_pipeline = Detecting(allowed_ids=(0, 1, 2, 3), reference_id=2, trackers=trackers)

    try:
        while True:
            frames = detector_pipeline.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            
            tags_in_ref, tag_dict = detector_pipeline.process_frame(frame)
            detector_pipeline.annotate_frame(frame, tags_in_ref, tag_dict)

            cv2.imshow('RealSense Tracker', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        detector_pipeline.pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
