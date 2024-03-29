import cv2 as cv
import numpy as np
import imutils
import matplotlib.pyplot as plt
from time import sleep
import io
from PIL import Image
from pynput.keyboard import Key, Controller

from tensorflow import keras
from PIL import Image
import numpy as npqq

# Betanított modell betöltése
model = keras.models.load_model('model/A-Z99.h5', compile=False)

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

cap = cv.VideoCapture(camera)
try:
    for it in range(80):
        detectednum = 0
        sleep(0.01)
        x_coords = []
        y_coords = []

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
                                        minRadius=30, maxRadius=50)



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
                
                if detectednum > 30:
                    emptycnt = 0
                    just_resetted = False
                else:
                    detectednum += 1
            else:
                emptycnt += 1

            frame = cv.flip(frame, 1)
            mask = cv.flip(mask, 1)
            cv.imshow("Maszk", mask)
            cv.imshow("Android", frame)
            if emptycnt > 25 and not just_resetted:
                emptycnt = 0
                just_resetted = True
                break
            elif cv.waitKey(1) & 0xFF == ord('q'):
                emptycnt = 0
                just_resetted = True
                break




        x_coords = x_coords[-80:]
        y_coords = y_coords[-80:]

        # Plot létrehozása
        plt.plot(x_coords, y_coords, linewidth=5)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.axis('off')
        #plt.show()
        plt.savefig('current_drawing.png', bbox_inches='tight')
        plt.clf()

        img = Image.open('current_drawing.png').convert('L')  # Szürkeárnyalatos konverzió
        img = img.resize((28, 28))  # Méret átalakítása a modell bemenetének megfelelően
        img_array = np.array(img) / 255.0  # Normalizálás

        #Betű megtippelése a betanított modelllel
        predictions = model.predict(np.expand_dims(img_array, axis=0))
        predicted_class = np.argmax(predictions[0])
        predicted_letter = chr(ord('A') + predicted_class)
   
   
        # Eredmény megjelenítése a képernyőn
        print("A rajzolt betű: ", predicted_letter)

        keyboard.press(predicted_letter)

        probabilities = predictions[0]
        # print("Valószínűségek:")
        # for i, prob in enumerate(probabilities):
        #     print(f"{chr(ord('A') + i)}: {prob}")

    cv.destroyAllWindows()
except KeyboardInterrupt:
    cv.destroyAllWindows()
    cap.release()