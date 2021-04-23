#!/usr/local/bin/python
import os, sys

import time
import serial

import numpy as np
import cv2
import numpy as np
import cmath
import math

def RA(img):
    kernel = np.ones((5,5), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations = 7)
    kernel = np.ones((5,5), np.uint8)
    erosion = cv2.erode(dilation, kernel, iterations = 18)

    r1 = erosion[:,:,2].astype(np.float32) - erosion[:,:,0].astype(np.float32)
    r2 = erosion[:,:,2].astype(np.float32) - erosion[:,:,1].astype(np.float32)
    img1 = cv2.normalize(r1, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
    img2 = cv2.normalize(r2, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)

    bitwiseAnd = cv2.bitwise_and(img1, img2)

    ret, thresh1 = cv2.threshold(bitwiseAnd, 0, 255, cv2.THRESH_BINARY)
    #plt.imshow(cv2.cvtColor(thresh1, cv2.COLOR_BGR2RGB))
    
    return thresh1

def RC(img):
    r3 = img[:,:,2].astype(np.float32)-img[:,:,0].astype(np.float32)
    r4 = img[:,:,2].astype(np.float32)-img[:,:,1].astype(np.float32)    
    img3 = cv2.normalize(r3, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
    img4 = cv2.normalize(r4, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
    
    bitwiseAnd2 = cv2.bitwise_and(img3, img4)
    
    ret, thresh2 = cv2.threshold(bitwiseAnd2, 0, 255, cv2.THRESH_BINARY_INV)
    #plt.imshow(cv2.cvtColor(thresh2, cv2.COLOR_BGR2RGB))
    
    return thresh2

def Merge_Thinning_Edges(img1, img2):
    img = cv2.bitwise_and(img1, img2)
    # thinned = cv2.ximgproc.thinning(bitwiseAnd3)
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)
    ret,img = cv2.threshold(img,127,255,0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        # img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
        else:
            img = eroded.copy()

    edges = cv2.Canny(skel, 50, 150, apertureSize = 3)
    #plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
    
    return edges, img

def HoughLinesP(img, Original):
    lines = cv2.HoughLinesP(img, 1.0, np.pi/180, 100, minLineLength=200, maxLineGap=50)
    lines_data = np.empty([0,9])
    sp_x = np.empty([0,3])
    sp_y = np.empty([0,3])

    check_x = int(0)
    check_y = int(0)

    if lines is not None:       
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y1-y2)/(x1-x2)
            print("(",x1,",",y1,") ",'(',x2,",",y2,')')

            if (slope == 0):
                if (y1 == y2):
                    check_x = int(1)
                    s = np.array([(y1, y2, 90)])
                    sp_x = np.append(sp_x, s, axis=0)
                    #sp_x.append(s)
                    print('h')
                    print(sp_x.shape)
            elif (x1 == x2):
                check_y = int(1)
                s = np.array([(x1, x2, 90)])
                sp_y = np.append(sp_y, s, axis=0)
                #sp_y.append(s)
                print('s')
                print(sp_y.shape)
            elif (x2-x1 != 0):
                check_x = int(0)
                check_y = int(0)
                print("(",x1,",",y1,") ",'(',x2,",",y2,')')
                #print(slope)
                # if (abs(slope) <= 0.1 and abs(slope) >= 0) or (abs(slope) <= 1 and abs(slope) >= 0.9):				
                angle_in_radians = math.atan(abs(slope))
                angle_in_degrees = math.degrees(angle_in_radians)
                print(angle_in_degrees)
                if 0 <= abs(angle_in_degrees) <= 20 or 70 <= abs(angle_in_degrees) <= 90:
                    cv2.line(Original, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(Original, (x1,y1), 5, (0,255,0), -1)
                    cv2.circle(Original, (x2,y2), 5, (0,255,0), -1)
                    Linear_Equation_A = -slope
                    Linear_Equation_B = 1
                    Linear_Equation_C = y1-slope*x1
                    a = np.array([(Linear_Equation_A, Linear_Equation_B, Linear_Equation_C,slope,x1,x2,y1,y2,angle_in_degrees)])
                    lines_data = np.append(lines_data,a, axis=0)

    else:
        print('Nothing!!!')

    print(lines_data)
    
    return lines, Original, lines_data, sp_x, sp_y, check_x, check_y

def Intersection(img, lines_data, Horizontal, vertical, check_x, check_y):
    if check_x == 1 and check_y == 1:
        for k in range(len(Horizontal)):
            for m in range(len(vertical)):  		
                c_x = int(vertical[m][1])
                c_y = int(Horizontal[k][1])
                if  0 < c_x < 1000 and 0 < c_y < 1000:
                    print('X:', c_x)
                    print('Y:', c_y)
                    print('90 degree')

                    deg = int(Horizontal[k][1])

                    m_x = int(280)
                    m_y = int(510)

                    p1=np.array([c_x,c_y])
                    p2=np.array([m_x,m_y])
                    p3=p2-p1
                    p4=math.hypot(p3[0],p3[1])

                    print('Distance:', p4)
                    cv2.line(img,(c_x,c_y),(m_x,m_y),(255,255,0),3)
                    cv2.putText(img, str(p4)[0:5]+' Distance', (m_x-50, m_y-45), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,0),1)
                                        
                    cv2.circle(img, (c_x, c_y), 5, (0,0,255), -1)                    
                    cv2.line(img,(c_x,0),(c_x,c_y),(255,0,0),3)#竖向
                    cv2.line(img,(1000,c_y),(c_x,c_y),(255,0,0),3)#横向
                    
                    cv2.putText(img, '90 degree', (c_x+50, c_y+45), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,0),1)
    else:		
        for i in range(len(lines_data)):
            for j in range(i+1,len(lines_data)):
                A = np.array([[lines_data[i][0], lines_data[i][1]], [lines_data[j][0], lines_data[j][1]]])        
                try:
                    inv_A = np.linalg.inv(A)
                except np.linalg.LinAlgError:
                    print('error') 

                B = np.array([lines_data[i][2], lines_data[j][2]])
                try:
                    ans = np.linalg.inv(A).dot(B)
                    c_x = int(ans[0])
                    c_y = int(ans[1])

                    if  0 < c_x < 1000 and 0 < c_y < 1000:
                        m_x = int(280)
                        m_y = int(510)

                        p1=np.array([c_x,c_y])
                        p2=np.array([m_x,m_y])
                        p3=p2-p1
                        p4=math.hypot(p3[0],p3[1])
                        print('Distance:', p4)

                        cv2.line(img,(c_x,c_y),(m_x,m_y),(255,255,0),3)
                        cv2.putText(img, str(p4)[0:5]+'mm  Distance', (m_x-50, m_y-45), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,0),1)
                                        
                        cv2.circle(img, (c_x, c_y), 5, (0,0,255), -1)                    
                        cv2.line(img,(c_x,0),(c_x,c_y),(255,0,0),3)#竖向
                        cv2.line(img,(1000,c_y),(c_x,c_y),(255,0,0),3)#横向

                        print('X:', c_x)
                        print('Y:', c_y)


                        deg = lines_data[i][8]

                        L_x = int(lines_data[i][4])
                        L_y = int(lines_data[i][6])
                        
                        R_x = int(lines_data[i][5])
                        R_y = int(lines_data[i][7])

                        if deg < 60:
                            cv2.line(img,(L_x, L_y),(c_x,c_y),(0,0,255),3)#Left
                        else:
                            cv2.line(img,(R_x, R_y),(c_x,c_y),(0,0,255),3)#Right

                        cv2.putText(img, str(deg)[0:5]+' degree', (c_x+50, c_y+45), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,0),1)
                        print('degree:',deg)

                except np.linalg.LinAlgError:
                    print('error')
    
    return img, deg, p1, p4

cap = cv2.VideoCapture(0) 

x = np.load('/home/ivetynano001/py/fname.npz')
M = x['M']

serial_port = serial.Serial(
    port="/dev/ttyUSB0",
    baudrate=115200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
)
# Wait a second to let the port initialize
time.sleep(1)

while(1): 
    
    if __name__ == '__main__':
        
        ret, frame = cap.read() 

        if frame.size == 0:
        ##if ret == False:   
            print(f"Fail to read image {filename}")
        else:                       
            (height, width, channels) = frame.shape
            print(f"frame dimension ( {height} , {width}, {channels})\n" )

            if channels != 3:
                print("Image is not a color image ##################")

            if frame.dtype != "uint8":
                print("Image is not of type uint8 #################")

            cv2.imshow('Original frame',frame) 

            processed = cv2.warpPerspective(frame,M,(560, 510))

            scr = processed.copy()

            print(scr.shape)

            area = RA(scr)
            color = RC(scr)
            image = Merge_Thinning_Edges(area, color)
            edges = image[0]
            AND = image[1]
            cv2.imshow('and', AND)
            cv2.imshow('edges', edges)
            HoughLines = HoughLinesP(edges, scr)
            Lines = HoughLines[0]
            if Lines is not None:
                new_img = HoughLines[1]
                data = HoughLines[2]
                Horizontal = HoughLines[3]
                vertical = HoughLines[4]
                check_x = HoughLines[5]
                check_y = HoughLines[6]

                result = Intersection(new_img, data, Horizontal, vertical, check_x, check_y)
                new_img = str(result[0])[0:5]
                angle = str(result[1])[0:5]
                point = str(result[2])[0:5]
                distance = str(result[3])[0:5]

                isItPrinted = false
                try:
                    data = serial_port.readline()
                    if data == 1 and isItPrinted == false:
                        if serial_port.inWaiting() > 0:
                            print(data)
                            serial_port.write(angler+"\\r\n".encode())
                            serial_port.write(point)+"\\r\n".encode()
                            serial_port.write(distance+"\\r\n".encode())

                            isItPrinted = True
                except KeyboardInterrupt:
                    print("Exiting Program")
                except Exception as exception_error:
                    print("Error occurred. Exiting Program")
                    print("Error: " + str(exception_error))
                finally:
                    serial_port.close()
                    pass
                
                cv2.imshow('Result', new_img)
            else:
                print('Nothing')

            k = cv2.waitKey(5) & 0xFF
            if k == 27: 
                break

# Close the window 
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()  