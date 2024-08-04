import numpy as np
import keras

INPUT_SHAPE = (100, 100)

model = keras.models.load_model("NN_MODEL4.h5")
model.compile(optimizer="adam", metrics=["accuracy"], loss='sparse_categorical_crossentropy')
model.summary()


classes = ["cat", "dog"]
image_path = ["images.jpg", "1_4_4.jpg"]


def get_predict_classificator(image_paths, classes):
    predicts = []
    for image_path in image_paths:
        img = keras.utils.load_img(image_path, target_size=INPUT_SHAPE)
        img_array = keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        predict_ = model.predict(img_array)
        predicts.append(classes[np.argmax(predict_)])
    return predicts


print(get_predict_classificator(image_path, classes))