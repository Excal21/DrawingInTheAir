import requests
import cv2 as cv
import numpy as np
import imutils

url = "http://192.168.1.12:8080/shot.jpg"


Orange_UB = np.array([55 , 220 , 255])
Orange_LB = np.array([20 , 50 , 140])

while True:
    img_resp = requests.get(url) 
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8) 
    frame = cv.imdecode(img_arr, -1) 
    frame = imutils.resize(frame, width=720, height=1280) 

    cv.imshow("Android", frame)
    if cv.waitKey(1) & 0xFF == ord('q'): break

cv.destroyAllWindows()
