import numpy as np
from to_patch import ImageProcessor
from to_image import ImageStitcher
import tempfile, os, warnings, sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import tensorflow as tf


def load_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python xlss.py <input_dir> <output_dir>")
        sys.exit(1)
    ipath = sys.argv[1]
    opath = sys.argv[2]
    temp_dir = tempfile.mkdtemp()
    model = tf.keras.models.load_model('xlss.keras')

    ImageProcessor(ipath, temp_dir, 25, None)()
    test_patches = os.listdir(temp_dir)
    test_patches = [os.path.join(temp_dir, file) for file in test_patches]

    test_ds = tf.data.Dataset.from_tensor_slices(test_patches)
    test_ds = test_ds.map(load_image,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(128)
    pred = model.predict(test_ds)
    pred = np.clip(pred, 0, 1)

    # Overwrite the test_patches
    for file_path, img in zip(test_patches, pred):
        img = tf.image.convert_image_dtype(img, tf.uint8)
        img = tf.image.encode_png(img)
        tf.io.write_file(file_path, img)

    ImageStitcher(temp_dir, opath)()

    os.system(f"rm -rf {temp_dir}")
    print("Done!")
