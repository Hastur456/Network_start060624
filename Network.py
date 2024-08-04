import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from keras.preprocessing import image_dataset_from_directory
import os

INPUT_SHAPE = 100
BATCH_SIZE = 32

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
main_dir = keras.utils.get_file('cats_and_dogs_filterted.zip', origin=_URL, extract=True)

base_dir = os.path.join(os.path.dirname(main_dir), 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_data = image_dataset_from_directory(train_dir, image_size=(INPUT_SHAPE, INPUT_SHAPE), color_mode="rgb")
val_data = image_dataset_from_directory(validation_dir, image_size=(INPUT_SHAPE, INPUT_SHAPE), color_mode="rgb")

model = keras.models.Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(INPUT_SHAPE, INPUT_SHAPE, 3)))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(GlobalAveragePooling2D())

model.add((Dropout(0.5)))

model.add(Dense(512, activation="relu"))
model.add(Dense(2, activation="softmax"))

model.compile(optimizer="adam", metrics=["accuracy"], loss='sparse_categorical_crossentropy')
model.summary()
model.fit(x=train_data, validation_data=val_data, epochs=10)
print(model.evaluate(x=val_data))

model.save("NN_MODEL4.h5")


