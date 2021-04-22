import numpy as np
import cv2
import numpy as np
import cmath
import math

def RA(img):
	kernel = np.ones((5,5), np.uint8)
	dilation = cv2.dilate(img, kernel, iterations = 7)
	kernel = np.ones((7,7), np.uint8)
	erosion = cv2.erode(dilation, kernel, iterations = 16)

	r1 = erosion[:,:,2].astype(np.float32) - erosion[:,:,0].astype(np.float32)
	r2 = erosion[:,:,2].astype(np.float32) - erosion[:,:,1].astype(np.float32)
	img1 = cv2.normalize(r1, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
	img2 = cv2.normalize(r2, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)

	bitwiseAnd = cv2.bitwise_and(img1, img2)

	ret, thresh1 = cv2.threshold(bitwiseAnd, 0, 255, cv2.THRESH_BINARY)
	
	return thresh1

def RC(img):
	r3 = scr_2[:,:,2].astype(np.float32) - scr_2[:,:,0].astype(np.float32)
	r4 = scr_2[:,:,2].astype(np.float32) - scr_2[:,:,1].astype(np.float32)    
	img3 = cv2.normalize(r3, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
	img4 = cv2.normalize(r4, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
	
	bitwiseAnd2 = cv2.bitwise_and(img3, img4)
	
	ret, thresh2 = cv2.threshold(bitwiseAnd2, 0, 255, cv2.THRESH_BINARY_INV)
	
	return thresh2

def Merge_Thinning_Edges(img1, img2):
	bitwiseAnd3 = cv2.bitwise_and(img1, img2)
	thinned = cv2.ximgproc.thinning(bitwiseAnd3)
	edges = cv2.Canny(thinned, 50, 150, apertureSize = 3)
	
	return edges

def HoughLinesP(img, Original):
	lines = cv2.HoughLinesP(img, 1, np.pi/180, 100, minLineLength=180, maxLineGap=250)
	lines_data_straight = np.empty([0,10])
	lines_data_horizontal = np.empty([0,10]) #横

	
	if lines is not None:
		for line in lines:
			x1, y1, x2, y2 = line[0]
			if(x2-x1 != 0):
				slope = (y1-y2)/(x1-x2)
				Linear_Equation_A = -slope
				Linear_Equation_B = 1
				Linear_Equation_C = y1-slope*x1
				print(slope)
				
				if slope < 0:
					angle_in_radians = math.atan(slope)
					angle_in_degrees = 360 + (math.degrees(angle_in_radians))
					 
					#horizontal识别码设为1
					if 0 < angle_in_degrees < 30 or  330 < angle_in_degrees < 360 or angle_in_degrees == 0:
						
						cv2.line(Original, (x1, y1), (x2, y2), (0, 255, 0), 1)
						print("(",x1,",",y1,") ",'(',x2,",",y2,')')
						cv2.circle(Original, (x1,y1), 5, (0,255,0), -1)
						cv2.circle(Original, (x2,y2), 5, (0,255,0), -1)
						
						a = np.array([(Linear_Equation_A, Linear_Equation_B, Linear_Equation_C,slope,x1,x2,y1,y2,angle_in_degrees,1)])
						lines_data_horizontal = np.append(lines_data_horizontal,a, axis=0)
						#print(lines_data_horizontal)
						print('h')
						
					#straight识别码设为0   
					elif 60 < angle_in_degrees < 90 or  240 < angle_in_degrees < 360:
						
						cv2.line(Original, (x1, y1), (x2, y2), (0, 255, 0), 1)
						print("(",x1,",",y1,") ",'(',x2,",",y2,')')
						cv2.circle(Original, (x1,y1), 5, (0,255,0), -1)
						cv2.circle(Original, (x2,y2), 5, (0,255,0), -1)
						
						b = np.array([(Linear_Equation_A, Linear_Equation_B, Linear_Equation_C,slope,x1,x2,y1,y2,angle_in_degrees,0)])
						lines_data_straight = np.append(lines_data_straight,b, axis=0)  
						print('s')

				elif slope > 0 or slope == 0:
					angle_in_radians = math.atan(slope)
					angle_in_degrees = math.degrees(angle_in_radians)

					#horizontal识别码设为1
					if 0 < angle_in_degrees < 30 or  330 < angle_in_degrees <= 360 or angle_in_degrees == 0:
						
						cv2.line(Original, (x1, y1), (x2, y2), (0, 255, 0), 1)
						print("(",x1,",",y1,") ",'(',x2,",",y2,')')
						cv2.circle(Original, (x1,y1), 5, (0,255,0), -1)
						cv2.circle(Original, (x2,y2), 5, (0,255,0), -1)
						
						a = np.array([(Linear_Equation_A, Linear_Equation_B, Linear_Equation_C,slope,x1,x2,y1,y2,angle_in_degrees,1)], dtype='float')
						lines_data_horizontal = np.append(lines_data_horizontal,a, axis=0)
						#print(lines_data_horizontal)
						print('h')
						
					#straight识别码设为0   
					elif 60 < angle_in_degrees < 90 or  240 < angle_in_degrees < 360:
						
						cv2.line(Original, (x1, y1), (x2, y2), (0, 255, 0), 1)
						print("(",x1,",",y1,") ",'(',x2,",",y2,')')
						cv2.circle(Original, (x1,y1), 5, (0,255,0), -1)
						cv2.circle(Original, (x2,y2), 5, (0,255,0), -1)
						
						b = np.array([(Linear_Equation_A, Linear_Equation_B, Linear_Equation_C,slope,x1,x2,y1,y2,angle_in_degrees,0)], dtype='float')
						lines_data_straight = np.append(lines_data_straight,b, axis=0)  
						print('s')
						#print(lines_data_straight)
									
				print(angle_in_degrees)

				# if (lines_data_horizontal.shape != lines_data_straight.shape):
				# 	A = np.array([[lines_data_straight[:,0], lines_data_straight[:,1]], [lines_data_horizontal[:,0], lines_data_horizontal[:,1]]])
				# 	try:
				# 		inv_A = np.linalg.inv(A)
				# 	except np.linalg.LinAlgError:
				# 		print('error') 

				# 	B = np.array([lines_data_straight[2], lines_data_horizontal[2]])
				# 	try:
				# 		ans = np.linalg.inv(A).dot(B)
				# 		c_x = int(ans[0])
				# 		c_y = int(ans[1])
				# 		if  0 < c_x < 1000 and 0 < c_y < 1000:
				# 			# print((c_x, c_y))                    
				# 			cv2.circle(img, (c_x, c_y), 5, (0,0,255), -1)                    
				# 			cv2.line(img,(c_x,0),(c_x,c_y),(255,0,0),3)#竖向
				# 			cv2.line(img,(1000,c_y),(c_x,c_y),(255,0,0),3)#横向

				# 			print('X:', c_x)
				# 			print('Y:', c_y)
							
				# 			if lines_data_horizontal[9] == 1:
				# 				if lines_data_straight[9] == 0:
				# 					L_x = int(lines_data_horizontal[4])
				# 					L_y = int(lines_data_horizontal[6])
									
				# 					R_x = int(lines_data_horizontal[5])
				# 					R_y = int(lines_data_horizontal[7])
									
				# 					imx = int(lines_data_straight[4])
				# 					imy = int(lines_data_straight[6])

				# 					xm1 = int(imx-imy/lines_data_straight[3])
					
				# 					deg = lines_data_horizontal[8]
				# 					if deg < 90:
				# 						cv2.line(img,(L_x, L_y),(c_x,c_y),(0,0,255),3)#竖向
				# 					else:
				# 						cv2.line(img,(R_x, R_y),(c_x,c_y),(0,0,255),3)#横向
										
				# 					cv2.putText(img, str(deg)[0:5]+' degree', (c_x+50, c_y+45), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,0),1)
				# 					cv2.line(img,(xm1,0),(c_x,c_y),(255,255,0),3)#中心定位
				# 					print(deg)
								
				# 	except np.linalg.LinAlgError:
				# 		print('error')    
				# 	continue;           

	else:
		print('Nothing!!')           
	return Original, lines_data_straight, lines_data_horizontal

def Intersection(img, lines_data_straight, lines_data_horizontal):
    for i in range(len(lines_data_straight)):

        A = np.array([[lines_data_straight[i][:,0], lines_data_straight[i][:,1]], [lines_data_horizontal[i][:,0], lines_data_horizontal[i][:,1]]])        
        try:
            inv_A = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            print('error') 

        B = np.array([lines_data_straight[i][2], lines_data_horizontal[i][2]])
        try:
            ans = np.linalg.inv(A).dot(B)
            c_x = int(ans[0])
            c_y = int(ans[1])
            if  0 < c_x < 1000 and 0 < c_y < 1000:
                # print((c_x, c_y))                    
                cv2.circle(img, (c_x, c_y), 5, (0,0,255), -1)                    
                cv2.line(img,(c_x,0),(c_x,c_y),(255,0,0),3)#竖向
                cv2.line(img,(1000,c_y),(c_x,c_y),(255,0,0),3)#横向

                print('X:', c_x)
                print('Y:', c_y)
				
                if lines_data_horizontal[i][9] == 1:
                    if lines_data_straight[i][9] == 0:
                        L_x = int(lines_data_horizontal[i][4])
                        L_y = int(lines_data_horizontal[i][6])
						
                        R_x = int(lines_data_horizontal[i][5])
                        R_y = int(lines_data_horizontal[i][7])
						
                        imx = int(lines_data_straight[i][4])
                        imy = int(lines_data_straight[i][6])

                        xm1 = int(imx-imy/lines_data_straight[i][3])
		
                        deg = lines_data_horizontal[i][8]
                        if deg < 90:
                            cv2.line(img,(L_x, L_y),(c_x,c_y),(0,0,255),3)#竖向
                        else:
                            cv2.line(img,(R_x, R_y),(c_x,c_y),(0,0,255),3)#横向
							
                        cv2.putText(img, str(deg)[0:5]+' degree', (c_x+50, c_y+45), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,0),1)
                        cv2.line(img,(xm1,0),(c_x,c_y),(255,255,0),3)#中心定位
                        print(deg)
					
        except np.linalg.LinAlgError:
            print('error')   
    return img


cap = cv2.VideoCapture(0)

x = np.load('fname.npz')

M = x['M']

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

			processed = cv2.warpPerspective(frame,M,(550, 550))
			cv2.imshow('New', processed)

			scr_2 = processed.copy()
			area = RA(scr_2)
			color = RC(scr_2)
			edges = Merge_Thinning_Edges(area, color)
			Line = HoughLinesP(edges, scr_2)
			new_img = Line[0]
			straight = Line[1]
			horizontal = Line[2]

			print(straight.shape)
			print(horizontal)

			new_img = Intersection(new_img, straight, horizontal)
			
			cv2.imshow('Result', new_img)
			
			k = cv2.waitKey(5) & 0xFF
			if k == 27: 
				break

# Close the window 
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()  