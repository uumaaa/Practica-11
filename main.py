import cv2
import time
from Extras import difference
from Morphology import morph
import matplotlib.pyplot as plt
import numpy as np
cap = cv2.VideoCapture("nofi001.mp4")
# cap = cv2.VideoCapture("nofi006.mp4")
# cap = cv2.VideoCapture("nofi069.mp4")

first = True
i = 1
kernel = np.ones((21,21), np.uint8)  # Pu






cap.release()
cv2.destroyAllWindows()