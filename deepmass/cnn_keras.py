

import numpy as np
from keras import backend as K
from keras.layers import Input, Dropout, Conv2D, MaxPooling2D, UpSampling2D, add, BatchNormalization, Conv2DTranspose
from keras.layers import concatenate, AveragePooling2D
from keras.models import Model
from keras.callbacks import Callback
from keras.optimizers import Adam



# Make batch generator
def batch_generator(noisy_array, clean_array, gen_batch_size=32):
    while True:
        index = np.random.randint(0, noisy_array.shape[0], gen_batch_size)
        yield (noisy_array[index], clean_array[index])


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

        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)

        final = Conv2D(1, (3, 3), activation='sigmoid', padding='same', kernel_initializer='he_normal')(x)

        simple = Model(input_img, final)
        simple.summary()

        if self.learning_rate is None:
            simple.compile(optimizer='adam', loss='mse')
        else:
            simple.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')

        return simple



class autoencoder_model:
    """
    A CNN class that creates a denoising autoencoder
    No batch norm on the decoder: http://cs230.stanford.edu/files_winter_2018/projects/6937642.pdf
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


        x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(input_img)
        x = BatchNormalization()(x)
        x = AveragePooling2D((2, 2), padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = AveragePooling2D((2, 2), padding='same')(x)

        encoded = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)

        x = UpSampling2D((2,2))(encoded)
        x = Dropout(0.2)(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = UpSampling2D((2,2))(x)
        x = Dropout(0.2)(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', kernel_initializer='he_normal')(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.summary()

        if self.learning_rate is None:
            autoencoder.compile(optimizer='adam', loss='mse')
        else:
            autoencoder.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')

        return autoencoder



class unet:
    """
    A CNN class that creates a denoising U-NET
    Dropout (not batch norm) on the decoder: http://cs230.stanford.edu/files_winter_2018/projects/6937642.pdf
    """


    def __init__(self, map_size, learning_rate, dropout_val=None):
        """
        Initialisation
        :param map_size: size of square image (there are map_size**2 pixels)
        :param learning_rate: learning rate for the optimizer
        """
        self.map_size = map_size
        self.learning_rate = learning_rate
        self.dropout_val = dropout_val


    def model(self):
        input_img = Input(shape=(self.map_size, self.map_size, 1))


        x1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_img)
        x1 = BatchNormalization()(x1)
        x1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x1)
        x1 = BatchNormalization()(x1)

        pool1 = AveragePooling2D(pool_size=(2, 2))(x1)
        x2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        x2 = BatchNormalization()(x2)
        x2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x2)
        x2 = BatchNormalization()(x2)

        pool2 = AveragePooling2D(pool_size=(2, 2))(x2)
        x3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        x3 = BatchNormalization()(x3)
        x3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x3)
        x3 = BatchNormalization()(x3)

        up4 = UpSampling2D((2,2))(x3)
        merge4 = concatenate([x2, up4], axis=3)

        if self.dropout_val is None:
            merge4 = BatchNormalization()(merge4)
        else:
            merge4 = Dropout(self.dropout_val)(merge4)

        x4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge4)
        x4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x4)


        up5 = UpSampling2D((2,2))(x4)
        merge5 = concatenate([x1, up5], axis=3)


        if self.dropout_val is None:
            merge4 = BatchNormalization()(merge5)
        else:
            merge4 = Dropout(self.dropout_val)(merge5)

        x5 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
        x5 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x5)
        output = Conv2D(1, 1, activation='sigmoid')(x5)

        unet = Model(input_img, output)
        unet.summary()

        if self.learning_rate is None:
            unet.compile(optimizer='adam', loss='mse')
        else:
            unet.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')

        return unet

class unet_simple_deep:
    """
    A CNN class that creates a denoising U-NET
    Dropout (not batch norm) on the decoder: http://cs230.stanford.edu/files_winter_2018/projects/6937642.pdf
    """

    def __init__(self, map_size, learning_rate, dropout_val=None):
        """
        Initialisation
        :param map_size: size of square image (there are map_size**2 pixels)
        :param learning_rate: learning rate for the optimizer
        """
        self.map_size = map_size
        self.learning_rate = learning_rate
        self.dropout_val = dropout_val


    def model(self):
        input_img = Input(shape=(self.map_size, self.map_size, 1))

        x1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_img)
        x1 = BatchNormalization()(x1)

        pool1 = AveragePooling2D(pool_size=(2, 2))(x1)
        x2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        x2 = BatchNormalization()(x2)

        pool2 = AveragePooling2D(pool_size=(2, 2))(x2)
        x3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        x3 = BatchNormalization()(x3)

        pool3 = AveragePooling2D(pool_size=(2, 2))(x3)
        x4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        x4 = BatchNormalization()(x4)

        up5 = UpSampling2D((2, 2))(x4)
        merge5 = concatenate([x3, up5], axis=3)

        if self.dropout_val is None:
            merge5 = BatchNormalization()(merge5)
        else:
            merge5 = Dropout(self.dropout_val)(merge5)

        x5 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)

        up6 = UpSampling2D((2, 2))(x5)
        merge6 = concatenate([x2, up6], axis=3)

        if self.dropout_val is None:
            merge6 = BatchNormalization()(merge6)
        else:
            merge6 = Dropout(self.dropout_val)(merge6)

        x6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)

        up7 = UpSampling2D((2, 2))(x6)
        merge7 = concatenate([x1, up7], axis=3)

        if self.dropout_val is None:
            merge7 = BatchNormalization()(merge7)
        else:
            merge7 = Dropout(self.dropout_val)(merge7)

        x7 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        output = Conv2D(1, 1, activation='sigmoid')(x7)

        unet = Model(input_img, output)
        unet.summary()

        if self.learning_rate is None:
            unet.compile(optimizer='adam', loss='mse')
        else:
            unet.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')

        return unet


class unet_simple:
    """
    A CNN class that creates a denoising U-NET
    Dropout (not batch norm) on the decoder: http://cs230.stanford.edu/files_winter_2018/projects/6937642.pdf
    """

    def __init__(self, map_size, learning_rate, dropout_val=None):
        """
        Initialisation
        :param map_size: size of square image (there are map_size**2 pixels)
        :param learning_rate: learning rate for the optimizer
        """
        self.map_size = map_size
        self.learning_rate = learning_rate
        self.dropout_val = dropout_val

    def model(self):
        input_img = Input(shape=(self.map_size, self.map_size, 1))

        x1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_img)
        x1 = BatchNormalization()(x1)

        pool1 = AveragePooling2D(pool_size=(2, 2))(x1)
        x2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        x2 = BatchNormalization()(x2)

        pool2 = AveragePooling2D(pool_size=(2, 2))(x2)
        x3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        x3 = BatchNormalization()(x3)

        up4 = UpSampling2D((2, 2))(x3)
        merge4 = concatenate([x2, up4], axis=3)
        if self.dropout_val is None:
            merge4 = BatchNormalization()(merge4)
        else:
            merge4 = Dropout(self.dropout_val)(merge4)
        x4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge4)

        up5 = UpSampling2D((2, 2))(x4)
        merge5 = concatenate([x1, up5], axis=3)
        if self.dropout_val is None:
            merge5 = BatchNormalization()(merge5)
        else:
            merge5 = Dropout(self.dropout_val)(merge5)
        x5 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
        output = Conv2D(1, 1, activation='sigmoid')(x5)

        unet = Model(input_img, output)
        unet.summary()

        if self.learning_rate is None:
            unet.compile(optimizer='adam', loss='mse')
        else:
            unet.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')

        return unet

