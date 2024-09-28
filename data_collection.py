import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap =cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
imgSize = 300           # (300,300)
offset = 20


# labels = ['hello','thanks','yes','no','iloveyou']
images_per_label = 50

label = 'iloveyou'


os.mkdir(f"Data/{label}")
print(f"{label} folder created")

while True:
    success, img =cap.read()
    hands, img = detector.findHands(img)
    flippedImg = cv2.flip(img,1)                                               # flip camera horizontally

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255                # making matrix of ones with the size of the image of the hand / actual size of the area to be scanned 
        imgCrop = img[y-offset : y+ h +offset, x-offset : x+ w +offset]
        flippedImgCrop = cv2.flip(imgCrop,1) 

       # Condition:  if not a square, then strecth  (h>w = stretch w ; w>h = stretch h) 
        aspectRatio = h/w
        
        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(wCal,imgSize))
            wGap =math.ceil((imgSize-wCal)/2)                   
            imgWhite[: , wGap:wCal+wGap] = imgResize                            # put the hand image inside the imgWhite
        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop,(imgSize,hCal))
            hGap =math.ceil((imgSize-hCal)/2)                  
            imgWhite[hGap:hCal+hGap , :] = imgResize

    
        cv2.imshow("ImageWhite",cv2.flip(imgWhite,1))
        cv2.imshow("ImageCrop",flippedImgCrop)
        
        cv2.imwrite(f"Data/{label}/{label}.{time.time()}.jpg", imgWhite)
        print(images_per_label)
        time.sleep(0.1)
        images_per_label -= 1
        if images_per_label == 0:
            break

    cv2.imshow("image",flippedImg)
    cv2.waitKey(1)
    # if key == ord('s'):
print("Done")        