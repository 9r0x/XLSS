import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, Activation, Input, BatchNormalization, MaxPool2D, UpSampling2D, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l1
import os
import shutil


class SuperResolutionAutoencoder(Model):
    def __init__(self, input_shape=(50, 50, 3)):
        super(SuperResolutionAutoencoder, self).__init__()
        # Encoder layers
        
        self.conv1 = Conv2D(64, (3, 3), kernel_initializer='he_uniform', activation='relu', activity_regularizer=tf.keras.regularizers.l1(10e-10), padding='same')
        self.conv2 = Conv2D(64, (3, 3), kernel_initializer='he_uniform', activity_regularizer=tf.keras.regularizers.l1(10e-10), padding='same', activation='relu')
        self.pool1 = MaxPool2D(padding='same')
        
        self.conv3 = Conv2D(128, (3, 3), kernel_initializer='he_uniform', activity_regularizer=tf.keras.regularizers.l1(10e-10), padding='same', activation='relu')
        self.conv4 = Conv2D(128, (3, 3), kernel_initializer='he_uniform', activity_regularizer=tf.keras.regularizers.l1(10e-10), padding='same', activation='relu')
        self.pool2 = MaxPool2D(padding='same')
        
        self.conv5 = Conv2D(256, (3, 3), kernel_initializer='he_uniform', activity_regularizer=tf.keras.regularizers.l1(10e-10), padding='same', activation='relu')

        # Decoder layers
        self.upsample1 = UpSampling2D()
        self.conv6 = Conv2D(128, (3, 3), kernel_initializer='he_uniform', activity_regularizer=tf.keras.regularizers.l1(10e-10), padding='same', activation='relu')
        self.conv7 = Conv2D(128, (2, 2), kernel_initializer='he_uniform', activity_regularizer=tf.keras.regularizers.l1(10e-10), padding='valid', activation='relu')
        
        self.upsample2 = UpSampling2D()
        self.conv8 = Conv2D(64, (3, 3), kernel_initializer='he_uniform', activity_regularizer=tf.keras.regularizers.l1(10e-10), padding='same', activation='relu')
        self.conv9 = Conv2D(64, (3, 3), kernel_initializer='he_uniform', activity_regularizer=tf.keras.regularizers.l1(10e-10), padding='same', activation='relu')

        self.conv10 = Conv2D(3, (3, 3), kernel_initializer='he_uniform', activity_regularizer=tf.keras.regularizers.l1(10e-10), padding='same', activation='relu')

    def call(self, inputs):
        # Encoder
        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        x3 = self.pool1(x2)
        
        x4 = self.conv3(x3)
        x5 = self.conv4(x4)
        x6 = self.pool2(x5)
        
        x7 = self.conv5(x6)
        
        # Decoder
        x8 = self.upsample1(x7)
        x9 = self.conv6(x8)
        x10 = self.conv7(x9)
        
        x11 = Add()([x10, x5])
        
        x12 = self.upsample2(x11)
        x13 = self.conv8(x12)
        x14 = self.conv9(x13)
        
        x15 = Add()([x14, x2])
        
        decoded = self.conv10(x15)
        return decoded

# Load and preprocess data
def load_and_preprocess_data(path):
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        path,
        target_size=(50, 50),  # Low-resolution size
        batch_size=32,
        color_mode='rgb',
        shuffle=False
    )
    return generator


def compile_and_train(model, train_generator, validation_generator, save_path='super_resolution_autoencoder'):
    train_pairs = zip(train_generator, validation_generator) 

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(
        x=train_pairs, 
        epochs=5,
        steps_per_epoch=100,  
        validation_data=train_pairs,  
        validation_steps=50 
    )

    model.save(save_path, save_format='tf')

if __name__ == '__main__':
    model = SuperResolutionAutoencoder()
    train_generator = load_and_preprocess_data('Flickr2K/Flickr2K_LR_bicubic/X2_patches')
    validation_generator = load_and_preprocess_data('Flickr2K/Flickr2K_HR_patches/')
    compile_and_train(model, train_generator, validation_generator)
