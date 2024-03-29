from turtle import width
from pandas import wide_to_long
import cv2 as cv
import numpy as np
import imutils
import matplotlib.pyplot as plt
from time import sleep
from tensorflow import keras
from PIL import Image
import numpy as np


def calculate_contour_center(cnt):
    M = cv.moments(cnt)
    if M["m00"] != 0:
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        return center_x, center_y
    else:
        return None, None



x_coords = []
y_coords = []

with open('hsv.conf', 'r') as file:
    values = [line.split('=')[1].strip() for line in file.readlines()]
    camera = int(values[0]) if values[0].isdecimal() else values[0]
    Orange_LB = np.array([int(values[1]) , int(values[2]) , int(values[3])])
    Orange_UB = np.array([int(values[4]) , int(values[5]) , int(values[6])])


cap = cv.VideoCapture(camera)

for it in range(40):
    sleep(0.1)
    while True:

        ret, frame = cap.read() 
        frame = imutils.resize(frame, width=720, height=1280) 
        
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (17,17), 0)


        #cv.imshow("Blurframe", gray)


        HSV_im_1 = cv.cvtColor(frame , cv.COLOR_BGR2HSV)


        mask = cv.inRange(HSV_im_1,Orange_LB,Orange_UB)

        mask = cv.medianBlur(mask, 9)

        #circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1.2, 100, param1=100, param2=30, minRadius=170, maxRadius=300)



        #rows = mask.shape[0]

        cnts = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[-2]

        if cnts != ():
            maxcnt = list(filter(lambda A: cv.contourArea(A) == max([cv.contourArea(cnt) for cnt in cnts]), cnts))[0]
            if 2000 < cv.contourArea(maxcnt):
                center = calculate_contour_center(maxcnt)
                # circle center
                cv.circle(frame, center, 1, (0, 100, 100), 3)
                # circle outline
                if center != (None, None):
                    x_coords.append(-center[0])
                    y_coords.append(-center[1])
                    cv.circle(frame, center, 40, (255, 0, 255), 3)

        frame = cv.flip(frame, 1)
        mask = cv.flip(mask, 1)
        cv.imshow("Maszk", mask)
        cv.imshow("Android", frame)
        if cv.waitKey(1) & 0xFF == ord('q'): break

    x_coords = x_coords[-70:]
    y_coords = y_coords[-70:]

    plt.xlabel('x')
    plt.ylabel('y')


    plt.plot(x_coords, y_coords, linewidth=5)
    plt.axis('off')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.savefig('f_5{it}.png'.format(it=it), bbox_inches='tight')
    plt.show()
    x_coords = []
    y_coords = []


    cv.destroyAllWindows()