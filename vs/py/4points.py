import cv2
import numpy as np
cv2.namedWindow("image", cv2.WINDOW_NORMAL)

#img = cv2.imread("opencv_frame_0.png")

img = cv2.imread("opencv_frame_0.png")

#print img.shape
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		xy = "%d,%d" % (x, y)
		print (xy)
		cv2.circle(img, (x, y), 1, (255, 0, 0), thickness = -1)
		cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
					1.0, (0,0,0), thickness = 1)
		cv2.imshow("image", img)


cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", img)

while(True):
	try:
		cv2.waitKey(100)
	except Exception:
		cv2.destroyWindow("image")
		break
		
cv2.waitKey(0)
cv2.destroyAllWindow()