import cv2
import time
from Extras import difference
from Morphology import morph
import matplotlib.pyplot as plt
import numpy as np
cap = cv2.VideoCapture("nofi069.mp4")
first = True
i = 1
kernel = np.ones((21,21), np.uint8)  # Pu
while True: 
    # Lee un frame del video
    ret, frame = cap.read()
    # Verifica si se llegó al final del video
    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if first:
        first = False
        last = frame
    else:

        if(i % 2 ==0):
            diff= difference.difference_images(frame,last)
            mask = np.uint8(cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel))
            imagen_segmentada = cv2.bitwise_and(frame,frame, mask=mask)
            binary_images = np.hstack((diff,mask))
            gray_images= np.hstack((frame,imagen_segmentada))
            cv2.imshow('middle layers',binary_images)
            cv2.imshow('input output',gray_images)
            last = frame
    i +=1

cap.release()
cv2.destroyAllWindows()