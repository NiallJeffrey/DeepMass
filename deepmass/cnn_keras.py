

import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dropout, Conv2D, MaxPooling2D, UpSampling2D, add, BatchNormalization, Conv2DTranspose
from tensorflow.keras.layers import concatenate, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam



# Make batch generator
def BatchGenerator(noisy_array, clean_array, gen_batch_size=32, sample_weights=False):

    if sample_weights==False:
        while True:
            index = np.random.randint(0, noisy_array.shape[0], gen_batch_size)
            yield (noisy_array[index], clean_array[index])
    else:
        while True:
            index = np.random.randint(0, noisy_array.shape[0], gen_batch_size)
            yield (noisy_array[index], clean_array[index], np.array([1./np.var(clean_array[i]) for i in index]))


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


class SimpleModel:
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


class UnetlikeBaseline:
    """
    A CNN class that creates a denoising Unet
    """

    def __init__(self, map_size, learning_rate, channels=[1,1], dropout_val=None):
        """
        Initialisation
        :param map_size: size of square image (there are map_size**2 pixels)
        :param learning_rate: learning rate for the optimizer
        """
        self.map_size = map_size
        self.learning_rate = learning_rate
        self.dropout_val = dropout_val
        self.channels = channels

        if dropout_val is not None:
            print('using dropout: ' + str(dropout_val))

    def model(self):
        input_img = Input(shape=(self.map_size, self.map_size, self.channels[0]))

        x1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_img)
        x1 = BatchNormalization()(x1)

        pool1 = AveragePooling2D(pool_size=(2, 2))(x1)
        x2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        x2 = BatchNormalization()(x2)

        pool2 = AveragePooling2D(pool_size=(2, 2))(x2)
        x3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        x3 = BatchNormalization()(x3)

        pool3 = AveragePooling2D(pool_size=(2, 2))(x3)
        x4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        x4 = BatchNormalization()(x4)

        pool_deep = AveragePooling2D(pool_size=(2, 2))(x4)
        xdeep = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool_deep)
        xdeep = BatchNormalization()(xdeep)

        updeep = UpSampling2D((2, 2))(xdeep)
        mergedeep = concatenate([x4, updeep], axis=3)

        xdeep2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mergedeep)
        xdeep2 = BatchNormalization()(xdeep2)

        up5 = UpSampling2D((2, 2))(xdeep2)
        merge5 = concatenate([x3, up5], axis=3)
        merge5 = BatchNormalization()(merge5)

        x5 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)

        up6 = UpSampling2D((2, 2))(x5)
        merge6 = concatenate([x2, up6], axis=3)
        merge6 = BatchNormalization()(merge6)

        x6 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)

        up7 = UpSampling2D((2, 2))(x6)
        merge7 = concatenate([x1, up7], axis=3)
        merge7 = BatchNormalization()(merge7)

        x7 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        output = Conv2D(self.channels[1], 1, activation='sigmoid')(x7)

        unet = Model(input_img, output)
        unet.summary()

        if self.learning_rate is None:
            unet.compile(optimizer='adam', loss='mse')
        else:
            unet.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')

        return unet
