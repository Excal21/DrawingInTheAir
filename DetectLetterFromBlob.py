import cv2 as cv
import numpy as np
import imutils
import matplotlib.pyplot as plt
from PIL import Image
from pynput.keyboard import Controller
from tensorflow import keras
from PIL import Image

# Betanított modell betöltése
model = keras.models.load_model('model/A-Z98.h5', compile=False)

keyboard = Controller()

x_coords = []
y_coords = []

with open('hsv.conf', 'r') as file:
    values = [line.split('=')[1].strip() for line in file.readlines()]
    camera = int(values[0]) if values[0].isdecimal() else values[0]
    Orange_LB = np.array([int(values[1]) , int(values[2]) , int(values[3])])
    Orange_UB = np.array([int(values[4]) , int(values[5]) , int(values[6])])

emptycnt = 0
just_resetted = True


def calculate_contour_center(cnt):
    M = cv.moments(cnt)
    if M["m00"] != 0:
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        return center_x, center_y
    else:
        return None, None

cap = cv.VideoCapture(camera)

stop = False

try:
    while not stop:
        detectednum = 0
        x_coords = []
        y_coords = []

        while True:
            ret, frame = cap.read() 
            frame = imutils.resize(frame, width=720, height=1280) 
            HSV_img = cv.cvtColor(frame , cv.COLOR_BGR2HSV)


            mask = cv.inRange(HSV_img, Orange_LB,Orange_UB)
            mask = cv.medianBlur(mask, 9)

            cnts = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[-2]

            if cnts != ():
                maxcnt = list(filter(lambda A: cv.contourArea(A) == max([cv.contourArea(cnt) for cnt in cnts]), cnts))[0]
                if 2500 < cv.contourArea(maxcnt):
                    center = calculate_contour_center(maxcnt)
                    x_coords.append(-center[0])
                    y_coords.append(-center[1])
                    cv.circle(frame, center, 1, (0, 100, 100), 3)
                    cv.circle(frame, center, 40, (255, 0, 255), 3)
                    if detectednum > 30:
                        emptycnt = 0
                        just_resetted = False
                    else:
                        detectednum += 1
                else:
                    emptycnt += 1
            else:
                emptycnt += 1

            if len(x_coords) > 70:
                drawcords = zip(x_coords[-70:], y_coords[-70:])
            else:
                drawcords = zip(x_coords, y_coords)
            
            for x, y in drawcords:
                cv.circle(frame, (-x,-y), 10, (255,255,255), 3)



            frame = cv.flip(frame, 1)
            mask = cv.flip(mask, 1)
            cv.imshow("Maszk", mask)
            cv.imshow("Android", frame)
            if emptycnt > 30 and not just_resetted:
                emptycnt = 0
                just_resetted = True
                break
            elif cv.waitKey(1) & 0xFF == 27:
                stop = True
                break

        if not stop:
            x_coords = x_coords[-70:]
            y_coords = y_coords[-70:]

            plt.plot(x_coords, y_coords, linewidth=5)
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')
            plt.axis('off')

            plt.savefig('current_drawing.png', bbox_inches='tight')
            plt.clf()
            #Kép betöltése és átalakítása a modell bemenetére
            img = Image.open('current_drawing.png').convert('L')  #Szürkeárnyalatos konverzió

            img = img.resize((28, 28))  #Méret átalakítása a modell bemenetének megfelelően
            img_array = np.array(img) / 255.0  #Normalizálás

            #Betű felismerése a betanított modelllel
            predictions = model.predict(np.expand_dims(img_array, axis=0))
            predicted_class = np.argmax(predictions[0])
            predicted_letter = chr(ord('A') + predicted_class)
    
    
            #Eredmény megjelenítése a konzolon
            print("A rajzolt betű: ", predicted_letter)

            probabilities = predictions[0]
            keyboard.press(predicted_letter)
            print("Valószínűségek:")
            for i, prob in enumerate(probabilities):
                print(f"{chr(ord('A') + i)}: {prob}")


    cv.destroyAllWindows()
except KeyboardInterrupt:
    cv.destroyAllWindows()
    cap.release()