import cv2
import numpy as np
import matplotlib.pyplot as plt
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

    pts1 = np.float32([[170,207], [340,216], [351,363], [93,358]])

    pts2 = np.float32([[0,0], [600,0], [600,550], [0,550]])
#     pts2 = np.float32([[100,100], [700,100], [700,650], [100,650]])

    cv2.circle(frame,(170,207), 3, (0, 0, 255), -1)# g  l d
    cv2.circle(frame,(340,216), 3, (0, 0, 255), -1)# g  r d

    cv2.circle(frame,(351,363), 3, (0 ,0, 255), -1)#b
    cv2.circle(frame,(93,358),3,(0,0, 255), -1)#b
    cv2.imshow("test", frame)
    # 计算得到转换矩阵
    M = cv2.getPerspectiveTransform(pts1, pts2)
    print(M)

    np.savez('fname.npz', M = M)
    x = np.load('fname.npz')
    print(x['M'])

    # 实现透视变换转换
#     processed = cv2.warpPerspective(frame,M,(800, 750))
    processed = cv2.warpPerspective(frame,M,(600, 550))
    cv2.imshow("processed", processed)
    
    cv2.imwrite("processed.png", processed)
    
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    
cam.release()

cv2.destroyAllWindows()
