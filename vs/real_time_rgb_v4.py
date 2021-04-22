import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


cap = cv2.VideoCapture(0) 

while(1): 
    
    if __name__ == '__main__':
        
        ret, frame = cap.read() 

        if frame.size == 0:
        ##if ret == False:   
            print(f"Fail to read image {filename}")
        else:
            cv2.imshow('Original frame',frame)        

            (height, width, channels) = frame.shape
            print(f"frame dimension ( {height} , {width}, {channels})\n" )
            if channels != 3:
                print("Image is not a color image ##################")

            if frame.dtype != "uint8":
                print("Image is not of type uint8 #################")

            pts1 = np.float32([[83, 478], [534, 478], [219, 172], [404, 172]])
            #pts2 = np.float32([[0,0],[0,308],[450,308],[450,308]])
            pts2 = np.float32([[83, 478], [534, 478], [83,0], [534,0]])

            M = cv2.getPerspectiveTransform(pts1,pts2)

            print(M)
    
            #dst = cv2.warpPerspective(frame,M,(308,450), cv2.INTER_LINEAR)
            dst = cv2.warpPerspective(frame,M,(600,500))

            cv2.imshow('dst', dst) # Transformed Capture

            ms = dst.copy()

            kernel = np.ones((5,5), np.uint8)
            ##dilation = cv2.dilate(test, kernel, iterations = 3)
            dilation = cv2.dilate(ms, kernel, iterations = 7)
            plt.imshow(cv2.cvtColor(dilation, cv2.COLOR_BGR2RGB))

            kernel = np.ones((5,5), np.uint8)
            ##erosion = cv2.erode(dilation, kernel, iterations = 3)
            erosion = cv2.erode(dilation, kernel, iterations = 15)
            plt.imshow(cv2.cvtColor(erosion, cv2.COLOR_BGR2RGB))

            r1 = erosion[:,:,2].astype(np.float32)-erosion[:,:,0].astype(np.float32)
            img1 = cv2.normalize(r1, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)

            r2 = erosion[:,:,2].astype(np.float32)-erosion[:,:,1].astype(np.float32)
            img2 = cv2.normalize(r2, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)

            bitwiseAnd = cv2.bitwise_and(img1, img2)
            
            ret, thresh1 = cv2.threshold(bitwiseAnd, 0, 255, cv2.THRESH_BINARY)
            cv2.imshow('RA', thresh1)


            scr = dst.copy()

            r3 = scr[:,:,2].astype(np.float32)-scr[:,:,0].astype(np.float32)
            img3 = cv2.normalize(r3, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)

            r4 = scr[:,:,2].astype(np.float32)-scr[:,:,1].astype(np.float32)
            img4 = cv2.normalize(r4, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)

            bitwiseAnd2 = cv2.bitwise_and(img3, img4)

            ret, thresh2 = cv2.threshold(bitwiseAnd2, 0, 255, cv2.THRESH_BINARY_INV)

            cv2.imshow('RC', thresh2)

            

            bitwiseAnd3 = cv2.bitwise_and(thresh1, thresh2)
            cv2.imshow('Mix', bitwiseAnd3)
            
            test1 = bitwiseAnd3.copy()
            thinned = cv2.ximgproc.thinning(test1)
            cv2.imshow('Thinned', thinned)

            edges = cv2.Canny(thinned, 50, 150, apertureSize = 3)
            cv2.imshow('edges', edges)

            tempIamge = frame.copy()
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=180, maxLineGap=250)

            if lines is not None:

                lines_data =np.empty([0,5])
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(dst, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    print("(",x1,",",y1,") ",'(',x2,",",y2,')')

                    if(x2-x1 != 0):
                    #         slope = abs((y2-y1)/(x2-x1))
                        slope = (y1-y2)/(x1-x2)
                        cv2.circle(dst, (x1,y1), 5, (255,0,0), -1)
                        cv2.circle(dst, (x2,y2), 5, (255,0,0), -1)
                        
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
                        print(Linear_Equation_C)

                        a = np.array([(Linear_Equation_A, Linear_Equation_B, Linear_Equation_C,slope,0)])
                        lines_data = np.append(lines_data,a, axis=0)
                        
                        #A = y2 - y1;
                        #B = x1 - x2;
                        #C = x1 * y2 - x2 * y1;
                        
                    else:
                        slope = 0
                        #识别码设为1
                        a = np.array([(Linear_Equation_A, Linear_Equation_B, Linear_Equation_C,slope,0)])
                        lines_data = np.append(lines_data,a, axis=0)

            

                for i in range(len(lines_data)):
                    for j in range(i+1,len(lines_data)):
                        point = np.empty([0,2])

                        A = np.array([[lines_data[i][0], lines_data[i][1]], [lines_data[j][0], lines_data[j][1]]])
                        print("A:",A)

                        try:
                            inv_A = np.linalg.inv(A)
                        except np.linalg.LinAlgError:
                            print('error')
                                
                        #print(inv_A)    
                        
                        B = np.array([lines_data[i][2], lines_data[j][2]])
                        print("B:",B)

                        try:
                            ans = np.linalg.inv(A).dot(B)
                            if abs(ans[0]) < 1000 or abs(ans[1]) < 1000:
                                c_x = int(ans[0])
                                c_y = int(ans[1])
                                # print((c_x, c_y))
                                cv2.circle(dst, (c_x, c_y), 5, (0,0,255), -1)
                        except np.linalg.LinAlgError:
                            print('error')


                        #ans = ans.astype(np.float32)
                            
                        #cv2.circle(tempIamge, (int(ans[0]), int(ans[1])), 5, (0,0,255), -1)  #int np.int64 
                                    
                        # print(ans)

            else:
                print('Nothing!!')


            print(tempIamge.shape)
            cv2.imshow('tempIamge', dst)

            

            k = cv2.waitKey(5) & 0xFF
            if k == 27: 
                break

# Close the window 
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()  




