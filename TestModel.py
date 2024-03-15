from tensorflow import keras
from PIL import Image
import numpy as np

# Betanított modell betöltése
model = keras.models.load_model('best_model.h5')

# Kép betöltése és előkészítése
image_path = 'd_0.png'
img = Image.open(image_path)
img = img.resize((28, 28))
img = img.convert('L')
img_array = np.array(img) / 255.0

# Kép előrejelzése
predictions = model.predict(np.expand_dims(img_array, axis=0))
predicted_class = np.argmax(predictions[0])
predicted_letter = chr(ord('A') + predicted_class)
print(f"A képen található betű: {predicted_letter}")