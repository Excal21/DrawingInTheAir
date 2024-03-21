import os
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import random

# Adathalmaz beállításai
img_height = 28
img_width = 28
batch_size = 17


filenames = []
for filename in os.listdir('./test'):
    if os.path.isfile(os.path.join('./test', filename)):
        filenames.append(filename)

test_letters = [str(filename).strip().split('_')[0].upper() for filename in filenames]

smart = False
while not smart:


    model = keras.Sequential([
        keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.06)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(12, activation='softmax')
    ])


    checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath='best_model.h5',
                                        save_best_only=True,  # Csak a legjobb epochot menti
                                        monitor='accuracy',   # A figyelendő metrika (pl. validációs veszteség)
                                        mode='max',           # A metrika minimalizálására törekszik
                                        verbose=1)


    # Modell összeállítása
    model.compile(optimizer='adam',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    # Adathalmaz betöltése és előkészítése
    ds_train = keras.preprocessing.image_dataset_from_directory(
        'letters/',
        labels='inferred',
        label_mode='int',
        class_names=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'],
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=True,
        #validation_split=0.1,
        #subset="training",
    )

    # Adathalmaz bővítése
    data_augmentation = keras.Sequential([
        #keras.layers.experimental.preprocessing.RandomContrast(0.2),
        #keras.layers.experimental.preprocessing.RandomRotation(0.01),
        keras.layers.experimental.preprocessing.RandomZoom(0.1)
    ])

    ds_train = ds_train.map(lambda x, y: (data_augmentation(x, training=True), y))





    # Modell tanítása
    #model.fit(ds_train, epochs=20, callbacks=[checkpoint_callback], verbose=2)
    #model.fit(ds_train, epochs=20, verbose=2)

    #model.save('best_model.h5')



    model.fit(ds_train, epochs=15, verbose=2)
    predicted_letters = []
    #Validáció
    for path in filenames:
        img = Image.open('test/' + path).convert('L')  # Szürkeárnyalatos konverzió
        img = img.resize((28, 28))  # Méret átalakítása a modell bemenetének megfelelően
        img_array = np.array(img) / 255.0  # Normalizálás

        # Kép előrejelzése a betanított modelllel
        predictions = model.predict(np.expand_dims(img_array, axis=0), verbose=0)
        predicted_class = np.argmax(predictions[0])
        predicted_letter = chr(ord('A') + predicted_class)
        predicted_letters.append(predicted_letter)
        
        #print("A rajzolt betű: ", predicted_letter)

    total_characters = len(test_letters)
    same_characters = sum(1 for char1, char2 in zip(predicted_letters, test_letters) if char1 == char2)

    similarity_percentage = (same_characters / total_characters) * 100
    print("Egyezés aránya: ", similarity_percentage)
    if similarity_percentage >= 85: smart = True


model.save('best_model.h5')