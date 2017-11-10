# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import keras
# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import train_test_split
# from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, \
    Conv2D, MaxPooling2D, Dropout, Input, GlobalMaxPooling2D
from keras.optimizers import Adam


def read_jason(file='', loc='./data'):

    file_path = os.path.join(loc, file)
    df = pd.read_json(file_path)
    # print(df.dtypes)
    # df['inc_angle'] = df['inc_angle'].replace('na', -1).astype(float)
    # if df['inc_angle'].dtype == 'object':
    #     ind = df['inc_angle'] == 'na'
    #     # print(ind.sum())
    #     m = df.loc[~ind, 'inc_angle'].astype(np.float32).mean()
    #     df.loc[ind, 'inc_angle'] = m
    #     df['inc_angle'] = df['inc_angle'].astype(np.float32)
    # else:
    df['inc_angle'] = pd.to_numeric(df['inc_angle'], errors='coerce')
    df['inc_angle'] = df['inc_angle'].fillna(df['inc_angle'].mean())

    band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]])
    band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]])
    df = df.drop(['band_1', 'band_2'], axis=1)

    bands = np.stack((band1, band2,  0.5 * (band1 + band2)), axis=-1)
    del band1, band2

    return df, bands


def create_resnet():
    # construct cnn input
    ks = [5, 5]
    image_input = Input(shape=(75, 75, 3))
    cnn_input = BatchNormalization()(image_input)
    cnn_input = Conv2D(filters=32, kernel_size=ks, padding='same')(cnn_input)
    cnn_input = BatchNormalization()(cnn_input)
    cnn_input = Activation('relu')(cnn_input)
    cnn_input = MaxPooling2D()(cnn_input)
    cnn_input = Dropout(rate=.25)(cnn_input)
    cnn_input = Conv2D(filters=64, kernel_size=ks, padding='same')(cnn_input)
    cnn_input = BatchNormalization()(cnn_input)
    cnn_input = Activation('relu')(cnn_input)
    cnn_input = MaxPooling2D()(cnn_input)
    cnn_input = Dropout(.25)(cnn_input)

    # first residual
    cnn_residual = BatchNormalization()(cnn_input)
    cnn_residual = Conv2D(128, kernel_size=ks, padding='same')(cnn_residual)
    cnn_residual = BatchNormalization()(cnn_residual)
    cnn_residual = Activation('relu')(cnn_residual)
    cnn_residual = Dropout(.25)(cnn_residual)
    cnn_residual = Conv2D(64, kernel_size=ks, padding='same')(cnn_residual)
    cnn_residual = BatchNormalization()(cnn_residual)
    cnn_residual = Activation('relu')(cnn_residual)

    # input residual cnn
    cnn_input_residual_model = keras.layers.Add()([cnn_input, cnn_residual])

    # try to add more residual layer
    cnn_residual = BatchNormalization()(cnn_input_residual_model)
    cnn_residual = Conv2D(128, kernel_size=ks, padding='same')(cnn_residual)
    cnn_residual = BatchNormalization()(cnn_residual)
    cnn_residual = Activation('relu')(cnn_residual)
    cnn_residual = Dropout(.25)(cnn_residual)
    cnn_residual = Conv2D(64, kernel_size=ks, padding='same')(cnn_residual)
    cnn_residual = BatchNormalization()(cnn_residual)
    cnn_residual = Activation('relu')(cnn_residual)

    # input residual cnn
    cnn_input_residual_model = keras.layers.Add()([cnn_residual, cnn_input])

    # final CNN
    top_cnn = Conv2D(128, kernel_size=ks, padding='same')(cnn_input_residual_model)
    top_cnn = BatchNormalization()(top_cnn)
    top_cnn = Activation('relu')(top_cnn)
    top_cnn = MaxPooling2D()(top_cnn)
    top_cnn = Conv2D(256, kernel_size=ks, padding='same')(top_cnn)
    top_cnn = BatchNormalization()(top_cnn)
    top_cnn = Activation('relu')(top_cnn)
    top_cnn = Dropout(.25)(top_cnn)
    top_cnn = MaxPooling2D()(top_cnn)
    top_cnn = Conv2D(512, kernel_size=ks, padding='same')(top_cnn)
    top_cnn = BatchNormalization()(top_cnn)
    top_cnn = Activation('relu')(top_cnn)
    top_cnn = Dropout(.25)(top_cnn)
    top_cnn = MaxPooling2D()(top_cnn)
    top_cnn = GlobalMaxPooling2D()(top_cnn)

    # output model
    cnn_output = Dense(512)(top_cnn)
    cnn_output = BatchNormalization()(cnn_output)
    cnn_output = Activation('relu')(cnn_output)
    cnn_output = Dropout(.5)(cnn_output)
    cnn_output = Dense(256)(cnn_output)
    cnn_output = BatchNormalization()(cnn_output)
    cnn_output = Activation('relu')(cnn_output)
    cnn_output = Dropout(.5)(cnn_output)
    cnn_output = Dense(2, activation='softmax')(cnn_output)

    cnn_model = keras.models.Model(inputs=[image_input], outputs=[cnn_output])

    learn_rate = 1e-3
    cnn_model.compile(optimizer=Adam(learn_rate), loss='binary_crossentropy', metrics=['accuracy'])
    # learning_rate = 1e-2
    # opt = Adam(lr=learning_rate, decay=1)
    # cnn_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return cnn_model
