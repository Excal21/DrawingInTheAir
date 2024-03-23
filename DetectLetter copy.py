import cv2 as cv
import numpy as np
import imutils
import matplotlib.pyplot as plt
from time import sleep
import io
from PIL import Image


url = "http://192.168.1.12:8080/video"

from tensorflow import keras
from PIL import Image
import numpy as npqq

# Betanított modell betöltése
model = keras.models.load_model('best_model.h5', compile=False)




x_coords = []
y_coords = []

with open('hsv.conf', 'r') as file:
    values = [int(line.split('=')[1]) for line in file.readlines()]
    Orange_LB = np.array([values[0] , values[1] , values[2]])
    Orange_UB = np.array([values[3] , values[4] , values[5]])

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


cap = cv.VideoCapture(url)
try:
    for it in range(40):
        detectednum = 0
        sleep(0.01)
        x_coords = []
        y_coords = []

        while True:
            # img_resp = requests.get(url) 
            # img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8) 
            # frame = cv.imdecode(img_arr, -1)
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


            if cnts is not None:
                for cnt in cnts:
                    if 4000 < cv.contourArea(cnt):
                        center = calculate_contour_center(cnt)
                        # circle center
                        cv.circle(frame, center, 1, (0, 100, 100), 3)
                        # circle outline
                        if center != (None, None):
                            x_coords.append(-center[0])
                            y_coords.append(-center[1])
                            cv.circle(frame, center, 10, (255, 0, 255), 3)
                
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

        # plt.plot(x_coords, y_coords, 'ro-')
        # plt.show()
        #plt.xlabel('x')
        #plt.ylabel('y')


        plt.plot(x_coords, y_coords, linewidth=5)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.axis('off')
        #plt.savefig('d_{it}.png'.format(it=it), bbox_inches='tight')

        plt.savefig('current_drawing.png', bbox_inches='tight')

        # # Kép betöltése és átalakítása a modell bemenetére
        img = Image.open('current_drawing.png').convert('L')  # Szürkeárnyalatos konverzió



        img = img.resize((28, 28))  # Méret átalakítása a modell bemenetének megfelelően
        img_array = np.array(img) / 255.0  # Normalizálás

        # Kép előrejelzése a betanított modelllel
        predictions = model.predict(np.expand_dims(img_array, axis=0))
        predicted_class = np.argmax(predictions[0])
        predicted_letter = chr(ord('A') + predicted_class)
   
   
        # Eredmény megjelenítése a képernyőn
        print("A rajzolt betű: ", predicted_letter)

        probabilities = predictions[0]
        # print("Valószínűségek:")
        # for i, prob in enumerate(probabilities):
        #     print(f"{chr(ord('A') + i)}: {prob}")

        plt.show()






        cv.destroyAllWindows()
except KeyboardInterrupt:
    cap.release()