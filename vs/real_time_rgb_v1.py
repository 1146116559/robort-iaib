import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


def ClassifyColor( BGR, width, height ): ##分類顏色 （BGR, width, height）
    r_threshold = 20 ##r閾值  before 10
    b_threshold = 20 ##b閾值   before 10
    FortyFive_degree = math.pi / 4 ## 45度
    grey_threshold = 10.0 * math.pi / 180.0 ##色角上下限 before 5
    
    for y in range(0, height, 1):
        cur_row = BGR[y]
        for x in range(0, width, 1):
            [b1, g1, r1] = cur_row[x]  #b1, g1 and r1 are type unsigned integer 8 bits 
            #Convert to 32 bits integer ＃b1，g1和r1是無符號整數8位類型 轉換為32位整數
            b = int(b1)
            g = int(g1)
            r = int(r1)
            
            #Compute color angles 計算色角
            r_on_b = math.atan2( r, b )
            r_on_g = math.atan2( r, g )
            b_on_g = math.atan2( b, g )
            
            
            #grey white and black color 灰色白色和黑色 ##自己調整誤差
            if abs( r_on_b - FortyFive_degree ) < grey_threshold: ##abs不考慮符號的值
                if abs( r_on_g - FortyFive_degree ) < grey_threshold:
                    if abs( b_on_g - FortyFive_degree ) < grey_threshold:
                        #white color
                        if r > 150 and g > 150 and b > 150:  
                            cur_row[x] = [255, 255, 255]
                        #black color
                        if r < 50:
                            cur_row[x] = [0, 0, 0]
                        else:
                            cur_row[x] = [128, 128, 128]
                        continue ##跳过循环
            #Red color
            if r - b > r_threshold:
                if r - g > r_threshold:
                    cur_row[x] = [0, 0, 200]
                    continue ##跳过某些循环
            
            #Blue color
            if b - r > b_threshold:
                if b - g > b_threshold:
                    cur_row[x] = [200, 0, 0]
                    continue
            
            #Other colors
            cur_row[x] = [0, 200, 200]

    return BGR


def RequiredArea( RA_BGR, width, height ): ##分類顏色 （BGR, width, height）
    r_threshold = 10 ##r閾值
    b_threshold = 10 ##b閾值
    
    for y in range(0, height, 1):
        cur_row = RA_BGR[y]
        for x in range(0, width, 1):
            [b1, g1, r1] = cur_row[x]  #b1, g1 and r1 are type unsigned integer 8 bits 
            #Convert to 32 bits integer ＃b1，g1和r1是無符號整數8位類型 轉換為32位整數
            b = int(b1)
            g = int(g1)
            r = int(r1)            

            #Red color
            if r - b > r_threshold:
                if r - g > r_threshold:
                    cur_row[x] = [255, 255, 255]
                    continue ##跳过某些循环
            
            #Other colors
            cur_row[x] = [0, 0, 0]

    return RA_BGR

def RequiredColor( RC_BGR, width, height ): ##分類顏色 （BGR, width, height）
    r_threshold = 10 ##r閾值
    b_threshold = 10 ##b閾值
    
    for y in range(0, height, 1):
        cur_row = RC_BGR[y]
        for x in range(0, width, 1):
            [b1, g1, r1] = cur_row[x]  #b1, g1 and r1 are type unsigned integer 8 bits 
            #Convert to 32 bits integer ＃b1，g1和r1是無符號整數8位類型 轉換為32位整數
            b = int(b1)
            g = int(g1)
            r = int(r1)            

            #Red color
            if r - b > r_threshold:
                if r - g > r_threshold:
                    cur_row[x] = [0, 0, 0]
                    continue ##跳过某些循环
            
            #Other colors
            cur_row[x] = [255, 255, 255]

    return RC_BGR



cap = cv2.VideoCapture(0) 

while(1): 
    
    if __name__ == '__main__':
        
        ret, frame = cap.read() 

        if frame.size == 0:
        ##if ret == False:   
            print(f"Fail to read image {filename}")
        else:
            cv2.imshow('Original frame',frame)        
            ##plt.figure(figsize=(20,10))
            ##plt.subplot(3, 1, 1) ##放大圖
            ##plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            (height, width, channels) = frame.shape
            print(f"frame dimension ( {height} , {width}, {channels})\n" )
            if channels != 3:
                print("Image is not a color image ##################")

            if frame.dtype != "uint8":
                print("Image is not of type uint8 #################")

            #Convert image to NumPy array (Create a new 2D array)
            # Note: The order is in BGR format! 將圖像轉換為NumPy數組（創建新的2D數組）
            BGR_array = np.array( frame )

            #Classify red, blue and grey color
            Result_array = ClassifyColor( BGR_array, width, height )


            ##plt.subplot(3, 1, 2)
            ##plt.imshow(cv2.cvtColor(Result_array, cv2.COLOR_BGR2RGB))
            ##plt.show

            # Display the resulting frame
            #cv2.imshow('Processed frame', Result_array)

            ##name = 'ResultFrame.jpg'
            ##print("Write processed frame : " + name)
            ##cv2.imwrite(name, BGR_array)

            ##cv2.waitKey(0)
            # When everything done, release the capture
            ##cv2.destroyAllWindows()            
            #cv2.imshow('BGR',Result_array)

            RA = Result_array.copy()

            kernel = np.ones((7,7), np.uint8)
            ##dilation = cv2.dilate(test, kernel, iterations = 3)
            RA_dilation = cv2.dilate(RA, kernel, iterations = 6)
            ##plt.imshow(cv2.cvtColor(dilation, cv2.COLOR_BGR2RGB))
            #cv2.imshow('RA_dilation',RA_dilation)

            kernel = np.ones((7,7), np.uint8)
            ##erosion = cv2.erode(dilation, kernel, iterations = 3)
            RA_erosion = cv2.erode(RA_dilation, kernel, iterations = 10)
            ##plt.imshow(cv2.cvtColor(erosion, cv2.COLOR_BGR2RGB))
            ##cv2.imshow('RA_erosion',RA_erosion)

            RA_frame = RA_erosion.copy()
            RA_BGR_array = np.array(RA_frame)
            RA_Result_array = RequiredArea( RA_BGR_array, width, height )    
            ##cv2.imshow('RequiredArea',RA_Result_array)

            RC = Result_array.copy()
            RC_BGR_array = np.array(RC)
            #nzCount = cv.CountNonZero(img3);
            RC_Result_array = RequiredColor( RC_BGR_array, width, height )
            ##cv2.imshow('RequiredColor',RC_Result_array)

            img1 = RA_Result_array.copy()
            img2 = RC_Result_array.copy()
            (height, width, channels) = img1.shape
            #print(f1"frame dimension ( {height} , {width}, {channels})\n" )

            (height, width, channels) = img2.shape
            #print(f2"frame dimension ( {height} , {width}, {channels})\n" )

            img = np.zeros ((480, 640, 3), np.uint8)

            img1_list = img1.tolist()
            img2_list = img2.tolist()
            img_list = img.tolist()
            for i in range(height):
                for j in range(width):
                    if img1_list[i][j] == img2_list[i][j] == [255, 255, 255]:
                        img[i][j][0] = 255
                        img[i][j][1] = 255
                        img[i][j][2] = 255
                    else:
                        img[i][j][0] = 0
                        img[i][j][1] = 0
                        img[i][j][2] = 0

            cv2.imshow('AND',img)  

            ms = img.copy()

            kernel = np.ones((3,3), np.uint8)
            ##dilation = cv2.dilate(test, kernel, iterations = 3)
            dilation = cv2.dilate(ms, kernel, iterations = 3)

            kernel = np.ones((3,3), np.uint8)
            ##erosion = cv2.erode(dilation, kernel, iterations = 3)
            erosion = cv2.erode(dilation, kernel, iterations = 3)

            test1 = erosion.copy()
            thinned = cv2.ximgproc.thinning(cv2.cvtColor(test1, cv2.COLOR_RGB2GRAY))
            cv2.imshow('Thinning',thinned)

            edges = cv2.Canny(thinned, 100, 200)
            cv2.imshow('edges',edges)

            '''
           
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 200, minLineLength=10, maxLineGap=250)

            # Draw lines on the image
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(edges, (x1, y1), (x2, y2), (255, 0, 0), 3)
            # Show result
            cv2.imshow("Result Image", edges) 

            '''
            '''

            for i in range(len(lines_data)):
                if lines_data[i][5] != 1 :
                    print(lines_data[i][4])

            #将每条线撮合成对应平行线

            parallel =np.empty([0,2], dtype = int)
            for i in range(len(lines_data)):
                for j in range(i+1,len(lines_data)):
                    if((abs(lines_data[i][4]-lines_data[j][4])<0.03)and(lines_data[i][5]!=1)):
                        print([i,j])
                        a = np.array([(i,j)])
                        parallel = np.append(parallel,a, axis=0)
                    elif(lines_data[i][5] == 1):
                        print(" the line is Vertical line")
                        break
            if len(parallel)< 2:
                print('7k7k')
                for i in range(len(lines_data)):
                    for j in range(i+1,len(lines_data)):
                        #print(i,lines_data[i][4]-lines_data[j][4])
                        if((abs(lines_data[i][4]-lines_data[j][4])<1.5)and(lines_data[i][5]!=1)and(i!=parallel[0][0])and(i!=parallel[0][1])and(j!=parallel[0][0])and(j!=parallel[0][1])):
                            print([i,j])
                            a = np.array([(i,j)])
                            parallel = np.append(parallel,a, axis=0)
                        elif(lines_data[i][5] == 1):
                            print(" the line is Vertical line")
                            break

            #输出撮合的平行线，各自的斜率
            parallel_slope = np.zeros(2)
            for i in range(len(parallel)):
                slope = (lines_data[parallel[i][0]][4]+lines_data[parallel[i][1]][4])/2
                parallel_slope[i] = slope
                
            parallel_slope#parallel[0][0]]

            habe = []
            for i in range(len(parallel)):
                for j in range(len(parallel[:])):
                    habe.append(parallel[i][j])

            lines_data[habe[3]][7]
            #将平行线数据导入，主要是懒的想

            horLine = []
            verLine = []
            # for i in range(len(habe)):
            #     if((lines_data[habe[i]][7]>(0-0.1))&(lines_data[habe[i]][7]<(0+0.1))):
            #         horLine.append(lines_data[habe[i]])
            #     else:
            #         verLine.append(lines_data[habe[i]])
            horLine.append(lines_data[habe[0]])
            horLine.append(lines_data[habe[1]])
            verLine.append(lines_data[habe[2]])
            verLine.append(lines_data[habe[3]])
            lines_data[habe[3]]

            #计算4条线的相交点

            points = []
            for l1 in horLine:
                for l2 in verLine:
                    a = np.array([
                        [np.cos(l1[7]), np.sin(l1[7])],
                        [np.cos(l2[7]), np.sin(l2[7])]
                    ])
                    b = np.array([l1[6],l2[6]])
                    points.append(np.linalg.solve(a, b))

            #计算中心点和输出中心点与四条线的相交点
            for point in points:
                cv2.circle(tempIamge, (int(point[0]),int(point[1])), 3, (0,0,255))
            midx = np.mean([point[0] for point in points])
            midy = np.mean([point[1] for point in points])
            cv2.circle(tempIamge, (int(midx), int(midy)), 3, (0,0,255))
            rgb = tempIamge[...,::-1]

            #盲猜一个中心点上方的位置，提供数据画线

            #import sympy   # 引入解方程的专业模块sympy #但是很傻逼，所以没用到
            imy = int(midy)
            imx = int(midx)
            #x = sympy.symbols("x")   # 申明未知数"x"
            y = (500-imy)
            k=horLine[0][4] #竖的斜率
            #a = sympy.solve([(y+k*imx)/(k*x)],[x])   # 写入需要解的方程体
            x_wei=(y+k*imx)/(k)
            #x_wei = abs(int(a[x]))
            x_wei = int(x_wei)
            print(x_wei)
            print(y+k*imx)
            #print(x_wei,parallel_slope[1],v,imy,imx,linet.shape)

            imy = int(midy)
            imx = int(midx)
            linet=tempIamge.copy()
            cv2.line(linet,(imx,0),(imx,imy),(0,0,255),3)#竖向

            cv2.line(linet,(1000,imy),(imx,imy),(0,0,255),3)#横向
            xm1 = int(imx-imy/parallel_slope[0])
            cv2.line(linet,(xm1,0),(imx,imy),(255,204,102),3)#中心定位线#66ccff
            print(imx,imy)

            plt.imshow(linet)
            plt.show()

            xm1 = imx-imy/parallel_slope[0]
                       
            linet2=linet.copy()
            angle = 90-math.degrees(math.atan(parallel_slope[0]))
            cv2.putText(linet2,str(angle)[0:5]+' degree',(imx+50,imy-100), cv2.FONT_HERSHEY_PLAIN, 2,(255,255,255),4)
            cv2.putText(linet2,'y=0',(imx+100,imy+90), cv2.FONT_HERSHEY_PLAIN, 2,(255,255,255),4)
            cv2.putText(linet2,'x=0',(imx-100,imy-50), cv2.FONT_HERSHEY_PLAIN, 2,(255,255,255),4)
            rgb = linet2[...,::-1]

            cv2.imshow('X',rgb) #红色通道 
            '''
            
            k = cv2.waitKey(5) & 0xFF
            if k == 27: 
                break

                
# Close the window 
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()             
