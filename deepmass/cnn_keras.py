

import numpy as np
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, add
from keras.models import Model

from keras.optimizers import Adam


class simple_model:
    """
    A CNN class that creates a denoising autoencoder
    """

    def __init__(self, map_size):
        """
        Initialisation
        :param map_size: size of square image (there are map_size**2 pixels)
        """
        self.map_size = map_size


    def model(self):
        input_img = Input(shape=(self.map_size, self.map_size, 1))

        filters = 32

        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(input_img)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', kernel_initializer='he_normal')(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.summary()
        autoencoder.compile(optimizer='adadelta', loss='mse')

        return autoencoder

class autoencoder_model:
    """
    A CNN class that creates a denoising autoencoder
    """

    def __init__(self, map_size):
        """
        Initialisation
        :param map_size: size of square image (there are map_size**2 pixels)
        """
        self.map_size = map_size


    def model(self):
        input_img = Input(shape=(self.map_size, self.map_size, 1))

        filters = 32

        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', kernel_initializer='he_normal')(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.summary()
        autoencoder.compile(optimizer='adadelta', loss='mse')

        return autoencoder





class residual_autoencoder_model:
    """
    A CNN class that creates a denoising autoencoder
    """

    def __init__(self, map_size, learn_rate=1e-4):
        """
        Initialisation
        :param map_size: size of square image (there are map_size**2 pixels)
        """
        self.map_size = map_size
        self.learn_rate = learn_rate


    def model(self):
        input_img = Input(shape=(self.map_size, self.map_size, 1))

        filters = 32

        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = UpSampling2D((2, 2))(x)
        
        final = add([x, input_img])
        
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', kernel_initializer='he_normal')(final)


        autoencoder = Model(input_img, decoded)
        autoencoder.summary()
        autoencoder.compile(optimizer='adadelta', loss='mse')

        return autoencoder

