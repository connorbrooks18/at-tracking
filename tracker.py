import cv2
import sys
import os
import numpy as np
from pupil_apriltags import Detector


# suppress apriltag C library stderr
"""
devnull = open(os.devnull, 'w')
old_stderr = os.dup(2)
os.dup2(devnull.fileno(), 2)
#"""

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
# disable auto settings first — order matters
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)      # 1 = manual, 3 = auto (varies by driver)
capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)          # disable autofocus
capture.set(cv2.CAP_PROP_AUTO_WB, 0)            # disable auto white balance

# then set manual values
capture.set(cv2.CAP_PROP_EXPOSURE, -6)          # log scale on most webcams, try -4 to -8
capture.set(cv2.CAP_PROP_BRIGHTNESS, 100)       # 0-255
capture.set(cv2.CAP_PROP_CONTRAST, 150)         # 0-255
capture.set(cv2.CAP_PROP_GAIN, 0)    



detector = Detector(
   families="tag36h11", # wrong family. need to print out actual tags 
   nthreads=4,
   quad_decimate=2.0,
   quad_sigma=.25,
   refine_edges=1,
   decode_sharpening=.25,
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
	#params = (900, 900, 640, 360) # 720p
	# for realsense belo
	params = (921.48, 921.89, 644.41, 358.64)
	allowed_ids = (0,1,2,3)

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#tags = detector.detect(gray, estimate_tag_pose=False)
	tags = detector.detect(gray, estimate_tag_pose=True, camera_params=params, tag_size=.034925)
 
 
	tags = [t for t in tags if t.decision_margin > 10 and t.tag_id in allowed_ids]


 	#reference tag number
	ref = 2
	tag_dict = {t.tag_id: t for t in tags}

	if ref not in tag_dict:
		pass
	else:
		last_reference = tag_dict[ref]

	
	poses = relativePoses(last_reference, tag_dict.values())

	if(poses == None):
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		continue
	for tag_id, pose in poses.items():
		tag = tag_dict[tag_id]
		center = tuple(tag.center.astype(int))

		# mark tag
		corners = tag.corners.astype(int)
		cv2.polylines(frame, [corners], True, (0,255,0), 2)
		cv2.circle(frame, center, 5, (0,0,255), -1)

		# get translation and add it to frame
		trans = [str(round(n, 3)) for n in pose['pos'].tolist()]
		text_trans = ",".join(trans)
		cv2.putText(frame, text_trans, center, 
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

		# add normal vector
		normal = pose['rot'][:, 2]
		norm_text = f"n:{normal[0]:.2f},{normal[1]:.2f},{normal[2]:.2f}"
		cv2.putText(frame, norm_text, (center[0], center[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)

	

	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


capture.release()
cv2.destroyAllWindows()
#os.dup2(old_stderr, 2)

