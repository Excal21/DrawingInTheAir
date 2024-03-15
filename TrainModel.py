import os
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# Adathalmaz beállításai
img_height = 28
img_width = 28
batch_size = 15

# Modell létrehozása
# model = keras.Sequential([
#     keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     keras.layers.MaxPooling2D((2, 2)),
#     keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     keras.layers.Flatten(),
#     keras.layers.Dense(64, activation='relu'),
#     keras.layers.Dense(10)
# ])

model = keras.Sequential([
    keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(4, activation='softmax')
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
    class_names=['A', 'B', 'C', 'D'],
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True,
    validation_split=0.1,
    subset="training",
    seed=100
)

# Adathalmaz bővítése
data_augmentation = keras.Sequential([
    #keras.layers.experimental.preprocessing.RandomContrast(0.2),
    keras.layers.experimental.preprocessing.RandomRotation(0.1),
    keras.layers.experimental.preprocessing.RandomZoom(0.1)
])

ds_train = ds_train.map(lambda x, y: (data_augmentation(x, training=True), y))





# Modell tanítása
#model.fit(ds_train, epochs=300, callbacks=[checkpoint_callback], verbose=2)
model.fit(ds_train, epochs=400, verbose=2)

model.save('best_model.h5')