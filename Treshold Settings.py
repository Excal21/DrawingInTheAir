import cv2
import numpy as np

def nothing(x):
    pass

with open('hsv.conf', 'r') as file:
    values = file.readline().strip().split('=')
    camera = int(values[1]) if values[1].isdecimal() else values[1]


cap = cv2.VideoCapture(camera)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)

cv2.resizeWindow('image', 1280, 1000)
cv2.createTrackbar('HMin','image',0,179,nothing)
cv2.createTrackbar('SMin','image',0,255,nothing)
cv2.createTrackbar('VMin','image',0,255,nothing)
cv2.createTrackbar('HMax','image',0,179,nothing)
cv2.createTrackbar('SMax','image',0,255,nothing)
cv2.createTrackbar('VMax','image',0,255,nothing)

cv2.setTrackbarPos('HMax', 'image', 179)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 255)

hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

waitTime = 33

while True:

    ret, img = cap.read()
    output = img
    hMin = cv2.getTrackbarPos('HMin','image')
    sMin = cv2.getTrackbarPos('SMin','image')
    vMin = cv2.getTrackbarPos('VMin','image')

    hMax = cv2.getTrackbarPos('HMax','image')
    sMax = cv2.getTrackbarPos('SMax','image')
    vMax = cv2.getTrackbarPos('VMax','image')

    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(img,img, mask= mask)

    if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    cv2.imshow('image',output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        with open('hsv.conf', 'w') as file:
            file.write('Camera=' + str(camera) + '\n')
            file.write('Hue minimum=' + str(hMin) + '\n')
            file.write('Saturation minimum=' + str(sMin) + '\n')
            file.write('Value minimum=' + str(vMin) + '\n')
            file.write('Hue maximum=' + str(hMax) + '\n')
            file.write('Saturation maximum=' + str(sMax) + '\n')
            file.write('Value maximum=' + str(vMax) + '\n')
        break

cap.release()
cv2.destroyAllWindows()


print('Orange_UB = np.array([{} , {} , {}])'.format(hMax, sMax, vMax))
print('Orange_LB = np.array([{} , {} , {}])'.format(hMin, sMin, vMin))