

import numpy as np
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model


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
