#!/usr/local/bin/python
import os, sys

import cv2
import numpy as np

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    
    # pts1 = np.float32([[175,385], [450,385], [224,210], [402,210]])
    # pts2 = np.float32([[0,480], [450,480], [0,0], [450,0]])
    p1 = [551,322]
    p2 = [1066,329]
    p3 = [1112,764]
    p4 = [385,765]

    pts1 = np.float32([p1, p2, p3, p4])

    pts2 = np.float32([[0,0], [600,0], [600,550], [0,550]])
    # pts2 = np.float32([[100,100], [700,100], [700,650], [100,650]])
    #pts2 = np.float32([[0,0], [560,0], [560,510], [0,510]])

    cv2.circle(frame,(551,322), 3, (0, 0, 255), -1)# g  l d
    cv2.circle(frame,(1066,329), 3, (0, 0, 255), -1)# g  r d

    cv2.circle(frame,(1112,764), 3, (0 ,0, 255), -1)#b
    cv2.circle(frame,(385,765),3,(0,0, 255), -1)#b
    cv2.imshow("test", frame)
    # 计算得到转换矩阵
    M = cv2.getPerspectiveTransform(pts1, pts2)
    print(M)

    np.savez('fname.npz', M = M)
    x = np.load('fname.npz')
    print(x['M'])

    # 实现透视变换转换
    #processed = cv2.warpPerspective(frame,M,(560, 510))
    processed = cv2.warpPerspective(frame,M,(600, 550))
    # processed = cv2.warpPerspective(frame,M,(600, 550))
    cv2.imshow("processed", processed)
    
    cv2.imwrite("processed.png", processed)
    
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    
cam.release()

cv2.destroyAllWindows()
