import os, sys, warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential([
    layers.Conv2D(32, 3, padding='same'),
    layers.ReLU(),
    layers.Conv2D(32, 3, padding='same'),
    layers.ReLU(),
    layers.Conv2DTranspose(32, 2, strides=2, padding='same'),
    layers.Conv2D(16, 3, padding='same'),
    layers.ReLU(),
    layers.Conv2D(16, 3, padding='same'),
    layers.ReLU(),
    layers.Conv2DTranspose(16, 2, strides=2, padding='same'),
    layers.Conv2D(3, 3, strides=2, padding='same'),
    layers.ReLU()
])


def prepare_ds(lr_folder,
               hr_folder,
               train_test_split=0.8,
               batch_size=32,
               seed=42):

    def load_and_preprocess_image(x_path, y_path):

        def load_image(file_path):
            img = tf.io.read_file(file_path)
            img = tf.image.decode_png(img, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float16)
            return img

        return load_image(x_path), load_image(y_path)

    lr_files = sorted(os.listdir(lr_folder))
    hr_files = sorted(os.listdir(hr_folder))

    assert lr_files == hr_files, "Mismatch in training files"

    lr_files = [os.path.join(lr_folder, _) for _ in lr_files]
    hr_files = [os.path.join(hr_folder, _) for _ in hr_files]

    split_idx = int(len(hr_files) * train_test_split)
    train_lr_files, train_hr_files = lr_files[:split_idx], hr_files[:split_idx]
    test_lr_files, test_hr_files = lr_files[split_idx:], hr_files[split_idx:]

    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_lr_files, train_hr_files))
    test_ds = tf.data.Dataset.from_tensor_slices(
        (test_lr_files, test_hr_files))

    train_ds = train_ds.map(load_and_preprocess_image,
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.map(load_and_preprocess_image,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    tf.random.set_seed(seed)
    train_ds = train_ds.shuffle(buffer_size=1024).batch(batch_size).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    return train_ds, test_ds


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python train.py <lr_dir> <hr_dir>")
        sys.exit(1)

    INPUT_SHAPE = (None, 50, 50, 3)
    NUM_EPOCHS = 5

    train_ds, test_ds = prepare_ds(sys.argv[1], sys.argv[2])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.build(input_shape=INPUT_SHAPE)
    model.fit(train_ds, epochs=NUM_EPOCHS, validation_data=test_ds)
    print(model.summary())

    model.save('xlss.keras')
