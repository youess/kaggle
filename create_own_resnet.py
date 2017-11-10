# -*- coding: utf-8 -*-

from keras.layers import Dense, Activation, BatchNormalization, \
    Conv2D, MaxPooling2D, Dropout, Input, GlobalMaxPooling2D, Add
from keras.optimizers import Adam


def create_net():

    x = Input(shape=(75, 75, 2))
    x = BatchNormalization()(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='valid')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(filters=128, kernel_size=(5, 5), padding='same')(x)
    x = MaxPooling2D()(x)
    x = Activation('relu')(x)

    x1 = BatchNormalization()(x)
    x1 = Conv2D(64, kernel_size=(3, 3), padding='valid')(x1)
    x1 = MaxPooling2D()(x1)
    x1 = Conv2D(64, kernel_size=(5, 5), padding='same')(x1)
    x1 = MaxPooling2D()(x1)
    x1 = Activation('relu')(x1)
    x1 = Add([x, x1])

    x2 = BatchNormalization()(x1)
    x2 = Conv2D(32, kernel_size=(3, 3), padding='valid')(x2)
    x2 = MaxPooling2D()(x2)
    x2 = Conv2D(32, kernel_size=(5, 5), padding='same')(x2)

