import tensorflow as tf
import keras


def custom_image_dataset_from_directory(image_paths, image_size, batch_size):
    def generator():
        for img_path in image_paths:
            img = keras.preprocessing.image.load_img(img_path, target_size=image_size)
            img_array = keras.preprocessing.image.img_to_array(img)
            yield img_array

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=tf.TensorSpec(shape=(image_size[0], image_size[1], 3), dtype=tf.float32)
    )
    dataset = dataset.batch(batch_size)
    return dataset




