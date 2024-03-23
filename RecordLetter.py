from turtle import width
from pandas import wide_to_long
import cv2 as cv
import numpy as np
import imutils
import matplotlib.pyplot as plt
from time import sleep

url = "http://192.168.1.12:8080/video"

from tensorflow import keras
from PIL import Image
import numpy as np


x_coords = []
y_coords = []

with open('hsv.conf', 'r') as file:
    values = [int(line.split('=')[1]) for line in file.readlines()]
    Orange_LB = np.array([values[0] , values[1] , values[2]])
    Orange_UB = np.array([values[3] , values[4] , values[5]])

cap = cv.VideoCapture(url)

for it in range(40):
    sleep(0.1)
    while True:

        ret, frame = cap.read() 
        frame = imutils.resize(frame, width=720, height=1280) 
        
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (17,17), 0)

        HSV_im_1 = cv.cvtColor(frame , cv.COLOR_BGR2HSV)


        mask = cv.inRange(HSV_im_1,Orange_LB,Orange_UB)

        mask = cv.medianBlur(mask, 9)

        rows = mask.shape[0]
        circles = cv.HoughCircles(mask, cv.HOUGH_GRADIENT, 1, rows,
                                    param1=100, param2=10,
                                    minRadius=30, maxRadius=60)


        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv.circle(frame, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                x_coords.append(-center[0])
                y_coords.append(-center[1])
                cv.circle(frame, center, radius, (255, 0, 255), 3)

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
    plt.savefig('z_{it}.png'.format(it=it), bbox_inches='tight')
    plt.show()
    x_coords = []
    y_coords = []


    cv.destroyAllWindows()