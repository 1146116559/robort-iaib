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
            #Red color
            if r - b > r_threshold:
                if r - g > r_threshold:
                    cur_row[x] = [255, 255, 255]
                    continue ##跳过某些循环
            
            #Blue color
            if b - r > b_threshold:
                if b - g > b_threshold:
                    cur_row[x] = [0, 0, 0]
                    continue
            
            #Other colors
            cur_row[x] = [0, 0, 0]

    return BGR


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

            ms = frame.copy()

            kernel = np.ones((5,5), np.uint8)
            ##dilation = cv2.dilate(test, kernel, iterations = 3)
            dilation = cv2.dilate(ms, kernel, iterations = 7)

            cv2.imshow('dilation', dilation)


            kernel = np.ones((7,7), np.uint8)
            ##erosion = cv2.erode(dilation, kernel, iterations = 3)
            erosion = cv2.erode(dilation, kernel, iterations = 9)
            cv2.imshow('erosion', erosion)
                
            #Convert image to NumPy array (Create a new 2D array)
            # Note: The order is in BGR format! 將圖像轉換為NumPy數組（創建新的2D數組）
            BGR_array = np.array( erosion )
        
            #Classify red, blue and grey color
            Result_array = ClassifyColor( BGR_array, width, height )
                    
            cv2.imshow('BGR',Result_array)

            k = cv2.waitKey(5) & 0xFF
            if k == 27: 
                break

                
# Close the window 
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()  

