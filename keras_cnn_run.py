# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import keras
# from sklearn.model_selection import StratifiedKFold
# from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, \
    Conv2D, MaxPooling2D, Dropout, Input, GlobalMaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import log_loss, accuracy_score
from datetime import datetime


data_dir = './data'
model_file = 'weights.hdf5'


def read_jason(file='', loc='./data'):

    file_path = os.path.join(loc, file)
    df = pd.read_json(file_path)
    # print(df.dtypes)
    # df['inc_angle'] = df['inc_angle'].replace('na', -1).astype(float)
    if df['inc_angle'].dtype == 'object':
        ind = df['inc_angle'] == 'na'
        # print(ind.sum())
        m = df.loc[~ind, 'inc_angle'].astype(np.float32).mean()
        df.loc[ind, 'inc_angle'] = m
        df['inc_angle'] = df['inc_angle'].astype(np.float32)
    else:
        df['inc_angle'] = df['inc_angle'].fillna(df['inc_angle'].mean())

    band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_1"]])
    band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in df["band_2"]])
    df = df.drop(['band_1', 'band_2'], axis=1)

    bands = np.stack((band1, band2,  0.5 * (band1 + band2)), axis=-1)
    del band1, band2

    return df, bands


df, bands = read_jason('train.json')

# build cnn deep model
# cnn_input_model = Sequential()

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

# train the model
batch_size = 64
epoch_num = 80        # 5
step_per_epoch = 2**14 / batch_size
early_stop = EarlyStopping(monitor='val_loss', patience=10)
check_point = ModelCheckpoint(model_file, monitor='val_loss', verbose=0, save_best_only=True)

gen_images = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=.3,
    height_shift_range=.3,
    zoom_range=.1,
    rotation_range=20
)


# train_test_split()
m = df.shape[0]
train_index = np.random.choice([True, False], size=m, replace=True, p=[.8, .2])
train_x, val_x = bands[train_index, :, :], bands[~train_index, :, :]
train_angle, val_angle = df.loc[train_index, 'inc_angle'], df.loc[~train_index, 'inc_angle']
label = pd.get_dummies(df['is_iceberg'])
train_y, val_y = label.loc[train_index, :].values, label.loc[~train_index, :].values


history = cnn_model.fit_generator(
    gen_images.flow(x=train_x, y=train_y, batch_size=batch_size, seed=123),
    step_per_epoch,
    epochs=epoch_num,
    validation_data=(val_x, val_y),
    callbacks=[check_point, early_stop]
)

model = load_model(model_file)
prediction = model.predict(val_x)
print('Validation accuracy score: {:.2f}'.format(accuracy_score(val_y[:, 1], prediction[:, 1] > .5)))
print('Validation log loss: {:.2f}'.format(log_loss(val_y[:, 1], prediction[:, 1])))

# predict the test data
test_df, test_bands = read_jason('test.json')
y_pred_test = model.predict(test_bands)

sub = pd.concat([test_df['id'], pd.Series(y_pred_test[:, 1])], axis=1)
sub.columns = ['id', 'is_iceberg']
tmp = 'sub_{}_cnn_{}.csv'.format(datetime.now().strftime('%Y%m%d'), 'default')
sub.to_csv(tmp, index=False)
