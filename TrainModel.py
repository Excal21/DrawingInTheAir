import os
import numpy as np
from tensorflow import keras
from keras.callbacks import EarlyStopping
from PIL import Image

# Adathalmaz beállításai
img_height = 28
img_width = 28
batch_size = 20


filenames = []
for filename in os.listdir('./valid'):
    if os.path.isfile(os.path.join('./valid', filename)):
        filenames.append(filename)

test_letters = [str(filename).strip().split('_')[0].upper() for filename in filenames]


imgarrays = []

for path in filenames:
    img = Image.open('valid/' + path).convert('L')  # Szürkeárnyalatos konverzió
    img = img.resize((28, 28))  # Méret átalakítása a modell bemenetének megfelelően
    img_array = np.array(img) / 255.0  # Normalizálás
    imgarrays.append(img_array)
    img.close()




smart = False
while not smart:


    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dense(128),
        keras.layers.LeakyReLU(alpha = 0.63),
        keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.028)),
        keras.layers.Dense(26, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(26, activation='softmax')
    ])


    # Modell összeállítása
    model.compile(optimizer='adam',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])

    # Adathalmaz betöltése és előkészítése
    ds_train = keras.preprocessing.image_dataset_from_directory(
        'letters/',
        labels='inferred',
        label_mode='int',
        class_names=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=True,
    )

    # Adathalmaz bővítése
    data_augmentation = keras.Sequential([
        keras.layers.experimental.preprocessing.RandomZoom(0.1)
    ])

    ds_train = ds_train.map(lambda x, y: (data_augmentation(x, training=True), y))

    # Modell tanítása
    earlystop = EarlyStopping(monitor='accuracy', patience=2)

    history = model.fit(ds_train, epochs=6, verbose=2, callbacks=[earlystop])
    predicted_letters = []
    
    #Validáció
    if history.history['accuracy'][-1] > 0.95:
        for img_array in imgarrays:
            #Betű felismerése a betanított modelllel
            predictions = model.predict(np.expand_dims(img_array, axis=0), verbose=0)
            predicted_class = np.argmax(predictions[0])
            predicted_letter = chr(ord('A') + predicted_class)
            predicted_letters.append(predicted_letter)

        total_characters = len(test_letters)
        same_characters = sum(1 for char1, char2 in zip(predicted_letters, test_letters) if char1 == char2)

        similarity_percentage = (same_characters / total_characters) * 100
        print("Egyezés aránya: ", similarity_percentage)
        if similarity_percentage >= 97: smart = True


model.save('best_model.h5')