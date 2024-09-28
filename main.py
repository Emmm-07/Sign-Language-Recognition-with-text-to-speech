import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3
# import time

def textToSpeech(text):
    engine = pyttsx3.init()
    engine.setProperty('rate',130)
    
    voices = engine.getProperty('voices')
    engine.setProperty('voice',voices[0].id)

    engine.say(text)
    #engine.say(f'<pitch middle="-100">{text}</pitch>')
    engine.runAndWait()
  

cap =cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
imgSize = 300           # (300,300)
offset = 20
prevPrediction = -1
delayCounter = 10
firstSay = True

labels = ['Thank you','Yes','No','I love you','Hello']
model = Classifier("Model/keras_model.h5","Model/labels.txt")


while True:
    success, img =cap.read()
    imgOutput  =  img.copy()
    flippedImg = cv2.flip(imgOutput,1) 

    hands, img = detector.findHands(img)
                                                  # flip camera horizontally

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

        prediction, index = model.getPrediction(imgWhite)
        
        cv2.rectangle(imgOutput, (x-offset, y-offset-50),(x-offset+90, y-offset),
                      color=(255,0,255),thickness=cv2.FILLED)
        
        cv2.putText(imgOutput, text=labels[index], org=(x,y-20), fontFace= cv2.FONT_HERSHEY_COMPLEX,
                     fontScale=1,color=(255,255,255), thickness=1 )
        
        cv2.rectangle(imgOutput, (x-offset, y-offset),(x+w+offset, y+h+offset),
                      color=(255,0,255),thickness=4)
        
        if delayCounter == 0 or firstSay:
            textToSpeech(labels[index])
            delayCounter = 10
            firstSay = False
        else:
            delayCounter -=1
        print(delayCounter)

        prevPrediction = index    
        
        # print(prediction, index)

        cv2.imshow("ImageWhite",cv2.flip(imgWhite,1))
        cv2.imshow("ImageCrop",flippedImgCrop)

        
     

    cv2.imshow("image",imgOutput)
    cv2.waitKey(1)
    # if key == ord('s'):
    
    
