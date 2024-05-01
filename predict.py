import os, sys
from multiprocessing import Pool
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt



def load_model(model_path):
    """
    Load a saved Keras model from the specified path.
    """
    return tf.keras.models.load_model(model_path)

def load_images(image_idx, directory):
    """
    Load all images from the specified directory that match the given image index.
    """
    prefix = f'{image_idx}' # SIX DIGITS e.g. 000114
    
    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter files that start with the correct prefix
    matched_files = [file for file in files if file.startswith(prefix)]

    
    # Load images
    images = [(os.path.join(directory, file)) for file in matched_files]
    
    return images

def hr_to_lr(hr_name):
    """
    000001_patch_0000.png to 000001x2_patch_0000.png
    """
    hr_name = hr_name.split('_')
    hr_name[0] += 'x2'
    return '_'.join(hr_name)

def load_and_preprocess_image(x_path, y_path):
    def load_image(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img
    return load_image(x_path), load_image(y_path)


def save_predictions(predictions, file_paths, output_directory):
    """
    Save the model's predictions to disk with modified filenames.
    """
    out_paths = []
    for pred, path in zip(predictions, file_paths):
        filename = os.path.basename(path)
        new_filename = f"{filename.split('.')[0]}_pred.png"  
        save_path = os.path.join(output_directory, new_filename)
        out_paths.append(save_path)
        Image.fromarray((pred * 255).astype(np.uint8)).save(save_path)

    return out_paths

def predict(data_path, model_path, out_path):
    """
    Load the model from the specified path, load the data from the specified path, and save the predictions to the specified path.
    """
    HR_path = os.path.join(data_path, 'HR')
    LR_path = os.path.join(data_path, 'LR')

    image_idx = 0 # SIX_DIGITS
    hr_images = load_images(image_idx, HR_path) # CHANGE TO CORRECT PATH
    lr_images = load_images(image_idx, LR_path) # CHANGE TO CORRECT PATH

    ds = tf.data.Dataset.from_tensor_slices((lr_images, hr_images))
    ds_processed = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)


    # load model
    model = load_model(model_path)
    
    # make predictions
    predictions = model.predict(ds_processed)

    # save predictions on disk
    out_paths = save_predictions(predictions, lr_images, out_path)

    return out_paths

if __name__ == "__main__":

    data_path = './data/minibatch'
    model_path = 'MODEL_PATH'
    output_directory = 'OUTPUT_DIRECTORY'
    
    out_paths = predict(data_path, model_path, output_directory)

    
    