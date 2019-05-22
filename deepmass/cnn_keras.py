

import numpy as np
from keras import backend as K
from keras.layers import Input, Dropout, Conv2D, MaxPooling2D, UpSampling2D, add, BatchNormalization, Conv2DTranspose
from keras.layers import concatenate
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
        x = BatchNormalization()(x)
        # x = Dropout(0.2)(x)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        # x = Dropout(0.2)(x)
        final = Conv2D(1, (3, 3), activation='sigmoid', padding='same', kernel_initializer='he_normal')(x)

        simple = Model(input_img, final)
        simple.summary()

        if self.learning_rate is None:
            simple.compile(optimizer='adam', loss='mse')
        else:
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

        filters = [32, 64]

        x = Conv2D(filters[0], (3, 3), strides=2,
                   activation='relu', padding='same', kernel_initializer='he_normal')(input_img)
        # x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(filters[1], (3, 3), strides=2,
                   activation='relu', padding='same', kernel_initializer='he_normal')(x)
        encoded = BatchNormalization()(x)

        # encoded = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2DTranspose(filters[1], (3, 3), strides=2,
                            activation='relu', padding='same', kernel_initializer='he_normal')(encoded)
        # x = UpSampling2D((2, 2))(x)
        x = Conv2DTranspose(filters[0], (3, 3), strides=2,
                            activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        # x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', kernel_initializer='he_normal')(x)


        autoencoder = Model(input_img, decoded)
        autoencoder.summary()

        if self.learning_rate is None:
            autoencoder.compile(optimizer='adam', loss='mse')
        else:
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


class unet_model:
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


        x1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_img)
        x1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(x1)
        x2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        x2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(x2)
        x3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        x3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x3)
        drop3 = Dropout(0.2)(x3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)


        x4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        x4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x4)
        drop4 = Dropout(0.5)(x4)

        up5 = Conv2D(128, 2, activation='relu', padding='same',
                     kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop4))
        merge5 = concatenate([drop3, up5], axis=3)
        x5 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
        x5 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x5)


        up6 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(x5))
        merge6 = concatenate([x2, up6], axis=3)
        x6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        x6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x6)

        up7 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(x6))
        merge7 = concatenate([x1, up7], axis=3)
        x7 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        x7 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x7)
        x7 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x7)
        output = Conv2D(1, 1, activation='sigmoid')(x7)

        unet = Model(input_img, output)
        unet.summary()

        if self.learning_rate is None:
            unet.compile(optimizer='adam', loss='mse')
        else:
            unet.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')

        return unet
