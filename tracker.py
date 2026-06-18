import cv2
import sys
import numpy as np
from pupil_apriltags import Detector

port = 1
capture = cv2.VideoCapture(port)
capture.set(cv2.CAP_PROP_EXPOSURE, -6) # decrease exposure
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # decrease auto exposure


detector = Detector(
   families="tag16h5", # wrong family. need to print out actual tags 
   nthreads=2,
   quad_decimate=1.0,
   quad_sigma=0.2,
   refine_edges=1,
   decode_sharpening=0.5,
   debug=0
)

if not capture.isOpened():
	print("failed to open camera")
	sys.exit()

while(True):
    # size capture.get(3) x capture.get(4)

	ret, frame = capture.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	tags = detector.detect(gray, estimate_tag_pose=False)
	tags = [t for t in tags if t.decision_margin > 50]
	for tag in tags:
		print(f"ID: {tag.tag_id}  center: {tag.center}")
		# draw it
		corners = tag.corners.astype(int)
		cv2.polylines(frame, [corners], True, (0,255,0), 2)
		cv2.circle(frame, tuple(tag.center.astype(int)), 5, (0,0,255), -1)
	

	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


capture.release()
cv2.destroyAllWindows()





