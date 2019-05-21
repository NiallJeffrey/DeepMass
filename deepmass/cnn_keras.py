

import numpy as np
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, add
from keras.models import Model
from keras.callbacks import Callback
from keras.optimizers import Adam

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))



class simple_model:
    """
    A CNN class that creates a simple denoiser
    """

    def __init__(self, map_size, learning_rate):
        """
        Initialisation
        :param map_size: size of square image (there are map_size**2 pixels)
        :param learning_rate: learning rate for the optimizer
        """
        self.map_size = map_size
        self.learning_rate = learning_rate


    def model(self):
        input_img = Input(shape=(self.map_size, self.map_size, 1))

        filters = 32

        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(input_img)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
#        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        final = Conv2D(1, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)

        simple = Model(input_img, final)
        simple.summary()
        simple.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')

        return simple

    
class simple_model_residual:
    """
    A CNN class that creates a residual CNN
    """

    def __init__(self, map_size, learning_rate):
        """
        Initialisation
        :param map_size: size of square image (there are map_size**2 pixels)
        :param learning_rate: learning rate for the optimizer
        """
        self.map_size = map_size
        self.learning_rate = learning_rate


    def model(self):
        input_img = Input(shape=(self.map_size, self.map_size, 1))

        filters = 32

        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(input_img)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        #x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)

        x = Conv2D(1, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = add([x, input_img])

        simple = Model(input_img, x)
        simple.summary()
        simple.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')

        return simple
    

class autoencoder_model:
    """
    A CNN class that creates a denoising autoencoder
    """


    def __init__(self, map_size, learning_rate):
        """
        Initialisation
        :param map_size: size of square image (there are map_size**2 pixels)
        :param learning_rate: learning rate for the optimizer
        """
        self.map_size = map_size
        self.learning_rate = learning_rate


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
        decoded = Conv2D(1, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.summary()
        autoencoder.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')

        return autoencoder





class residual_autoencoder_model:
    """
    A CNN class that creates a denoising autoencoder
    """

    def __init__(self, map_size, learning_rate):
        """
        Initialisation
        :param map_size: size of square image (there are map_size**2 pixels)
        :param learning_rate: learning rate for the optimizer
        """
        self.map_size = map_size
        self.learning_rate = learning_rate


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
        
        
        decoded = Conv2D(1, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        decoded = add([decoded, input_img])


        autoencoder = Model(input_img, decoded)
        autoencoder.summary()
        autoencoder.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')

        return autoencoder

