import numpy as np
import cv2
import numpy as np
import cmath
import math

def RA(img):
    kernel = np.ones((5,5), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations = 7)
    kernel = np.ones((5,5), np.uint8)
    erosion = cv2.erode(dilation, kernel, iterations = 15)

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

def Merge(img1, img2):
    bitwiseAnd3 = cv2.bitwise_and(img1, img2)
    #plt.imshow(cv2.cvtColor(bitwiseAnd3, cv2.COLOR_BGR2RGB))
    
    return bitwiseAnd3

def Thinning(img):
    thinned = cv2.ximgproc.thinning(img)
    #plt.imshow(cv2.cvtColor(thinned, cv2.COLOR_BGR2RGB))
    
    return thinned

def Edges(img):
    edges = cv2.Canny(img, 50, 150, apertureSize = 3)
    #plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
    
    return edges

def HoughLinesP(img, Original):
    lines = cv2.HoughLinesP(img, 1, np.pi/180, 100, minLineLength=180, maxLineGap=250)
    lines_data = np.empty([0,5])
    Angle = np.empty([0,5])

    if lines is not None:       
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(Original, (x1, y1), (x2, y2), (0, 255, 0), 1)
            print("(",x1,",",y1,") ",'(',x2,",",y2,')')

            if(x2-x1 != 0):
        #         slope = abs((y2-y1)/(x2-x1))
                slope = (y1-y2)/(x1-x2)

                num = int(3)
                num_sqrt = (cmath.sqrt(num))/num
                tan30_slope = float(num_sqrt.real)        

                if slope > -tan30_slope:
                    if slope < tan30_slope:
                        x = np.array([(x1, y1, x2, y2, slope)])
                        Angle = np.append(Angle,x, axis=0)


                cv2.circle(Original, (x1,y1), 5, (0,255,0), -1)
                cv2.circle(Original, (x2,y2), 5, (0,255,0), -1)

        #         Linear_Equation_A = abs(y2 - y1)
        #         Linear_Equation_B = abs(x1 - x2)
        #         Linear_Equation_C = abs(x1 * y2 - x2 * y1)
                Linear_Equation_A = -slope
                Linear_Equation_B = 1
                Linear_Equation_C = y1-slope*x1

                ##a = np.array([(x1,x2,y1,y2,Linear_Equation_A,Linear_Equation_B,Linear_Equation_C,0,rho,theta)])
                ##lines_data = np.append(lines_data,a, axis=0)
                print(slope)
                print(Linear_Equation_A)
                print(Linear_Equation_B)
                print(Linear_Equation_C,'\n')

                a = np.array([(Linear_Equation_A, Linear_Equation_B, Linear_Equation_C,slope,0)])
                lines_data = np.append(lines_data,a, axis=0)
                '''
                A = y2 - y1;
                B = x1 - x2;
                C = x1 * y2 - x2 * y1;
                ''' 
            else:
                slope = 0
                #识别码设为1
                a = np.array([(Linear_Equation_A, Linear_Equation_B, Linear_Equation_C,slope,0)])
                lines_data = np.append(lines_data,a, axis=0)

        print(Original.shape)
        #plt.imshow(cv2.cvtColor(Original, cv2.COLOR_BGR2RGB))
    else:
        print('Nothing!!!')
    
    return Original, Angle, lines_data, lines

def Intersection(img, lines_data):
    point = np.empty([0,2])
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
                if  0 < c_x < 1000:
                    if 0 < c_y < 1000:
                        # print((c_x, c_y))                    
                        cv2.circle(img, (c_x, c_y), 5, (0,0,255), -1)                    
                        cv2.line(img,(c_x,0),(c_x,c_y),(255,0,0),3)#竖向
                        cv2.line(img,(1000,c_y),(c_x,c_y),(255,0,0),3)#横向

                        print('X:', c_x)
                        print('Y:', c_y)
                        
                        xy = np.array([(c_x, c_y)])
                        point = np.append(point,xy, axis=0)
            except np.linalg.LinAlgError:
                print('error')

    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    return point

def Angle(img, angle_data, point):
    for i in range(len(angle_data)):
        for j in range(i+1,len(angle_data)):
            
            Ax = int(angle_data[i][2])
            Ay = int(angle_data[i][3])
            
            x = int(point[i][0])
            y = int(point[i][1])
            
            cv2.line(img,(Ax, Ay),(x,y),(0,0,255),3)#横向

            inv = abs(np.arctan(angle_data[i][4])) 
            deg = abs(np.degrees(inv))

            cv2.putText(img, str(deg)[0:5]+' degree', (x+10, y+10), cv2.FONT_HERSHEY_PLAIN, 2,(0,0,255),2)

    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  
    

cap = cv2.VideoCapture(0) 

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

            area = RA(frame)
            color = RC(frame)
            AND = Merge(area, color)

            thin = Thinning(AND)
            edges = Edges(thin)
            HoughLines = HoughLinesP(edges, frame)           
            new_img = HoughLines[0]
            angle = HoughLines[1]
            Lines_data = HoughLines[2]
            Lines = HoughLines[3]

            if Lines is not None:
                points = Intersection(new_img, Lines_data)
                Angle(new_img, angle, points)
            else:
                print('No Line!!!')                

            cv2.imshow('Result', new_img)

            k = cv2.waitKey(5) & 0xFF
            if k == 27: 
                break

# Close the window 
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()  




