import cv2
import sys
import os
import numpy as np
from pupil_apriltags import Detector


# suppress apriltag C library stderr
devnull = open(os.devnull, 'w')
old_stderr = os.dup(2)
os.dup2(devnull.fileno(), 2)

def relativePoses(reference, alltags):
    """
    Returns dict of tag_id -> {pos, rot} expressed in reference tag's frame.
    Reference tag itself will be at origin (pos=[0,0,0], rot=I).
    """

    if(reference == None): return None

    R_ref_inv = reference.pose_R.T
    t_ref = reference.pose_t

    result = {}
    for tag in alltags:
        t_rel = R_ref_inv @ (tag.pose_t - t_ref)
        R_rel = R_ref_inv @ tag.pose_R
        result[tag.tag_id] = {
            'pos': t_rel.flatten(),
            'rot': R_rel,
        }
    
    return result





port = 1
capture = cv2.VideoCapture(port)
capture.set(cv2.CAP_PROP_EXPOSURE, -6) # decrease exposure
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # decrease auto exposure
# change resolution
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


detector = Detector(
   families="tag16h5", # wrong family. need to print out actual tags 
   nthreads=2,
   quad_decimate=1.0,
   quad_sigma=0.6,
   refine_edges=1,
   decode_sharpening=1,
   debug=0
)

if not capture.isOpened():
	print("failed to open camera")
	sys.exit()

last_reference = None

while(True):
    # size capture.get(3) x capture.get(4)

	ret, frame = capture.read()
	#params = (1350, 1350, 960, 540) # fx, fy, cx, cy
	params = (900, 900, 640, 360) # 720p
	allowed_ids = (8, 1)

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#tags = detector.detect(gray, estimate_tag_pose=False)
	tags = detector.detect(gray, estimate_tag_pose=True, camera_params=params, tag_size=.055)
 
 
	tags = [t for t in tags if t.decision_margin > 60 and t.tag_id in allowed_ids]


 	#reference tag number
	ref = 8
	tag_dict = {t.tag_id: t for t in tags}

	if ref not in tag_dict:
		pass
	else:
		last_reference = tag_dict[ref]

	
	poses = relativePoses(last_reference, tag_dict.values())

	"""
	for tag in tags:
		print(f"ID: {tag.tag_id}  center: {tag.center}")
		# draw it
		corners = tag.corners.astype(int)
		cv2.polylines(frame, [corners], True, (0,255,0), 2)
		cv2.circle(frame, tuple(tag.center.astype(int)), 5, (0,0,255), -1)
		# pose_t, pose_R
		trans = [str(round(n[0], 2)) for n in tag.pose_t.tolist()]
		text_trans = ",".join(trans)
		cv2.putText(frame, text_trans, tuple(tag.center.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	"""
	if(poses == None):
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		continue
	for tag_id, pose in poses.items():
		tag = tag_dict[tag_id]
		trans = [str(round(n, 2)) for n in pose['pos'].tolist()]
		text_trans = ",".join(trans)
		cv2.putText(frame, text_trans, tuple(tag.center.astype(int)), 
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
		corners = tag.corners.astype(int)
		cv2.polylines(frame, [corners], True, (0,255,0), 2)
		cv2.circle(frame, tuple(tag.center.astype(int)), 5, (0,0,255), -1)
	

	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


capture.release()
cv2.destroyAllWindows()
os.dup2(old_stderr, 2)

