import cv2 as cv
import numpy as np
import imutils

Orange_UB = np.array([55 , 220 , 255])
Orange_LB = np.array([20 , 50 , 140])

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # gray = cv.medianBlur(gray, 5)


    #cv.imshow("Blurframe", gray)


    HSV_im_1 = cv.cvtColor(frame , cv.COLOR_BGR2HSV)


    mask = cv.inRange(HSV_im_1,Orange_LB,Orange_UB)

    mask = cv.medianBlur(mask, 9)

    circles = cv.HoughCircles(mask, cv.HOUGH_GRADIENT, 1.2, 100, param1=100, param2=30, minRadius=10, maxRadius=70)



    rows = mask.shape[0]
    circles = cv.HoughCircles(mask, cv.HOUGH_GRADIENT, 1, rows / 8,
                                param1=50, param2=10,
                                minRadius=3, maxRadius=50)



    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(frame, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(frame, center, radius, (255, 0, 255), 3)





    cv.imshow("Maszk", mask)
    cv.imshow("Android", frame)
    if cv.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv.destroyAllWindows()