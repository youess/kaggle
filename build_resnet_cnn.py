# -*- coding: utf-8 -*-


# import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from datetime import datetime
from util import read_jason, create_resnet
import time

# set the cpu num
from multiprocessing import cpu_count
from keras import backend as K


safe_cpu_cnt = int(cpu_count() * .8)
K.set_session(K.tf.Session(
    config=K.tf.ConfigProto(
        intra_op_parallelism_threads=safe_cpu_cnt, inter_op_parallelism_threads=safe_cpu_cnt
)))

data_dir = './data'

df, bands = read_jason('train.json', data_dir)
# predict the test data
test_df, test_bands = read_jason('test.json')


# m = df.shape[0]
# train_index = np.random.choice([True, False], size=m, replace=True, p=[.8, .2])
# train_x, val_x = bands[train_index, :, :], bands[~train_index, :, :]
# train_angle, val_angle = df.loc[train_index, 'inc_angle'], df.loc[~train_index, 'inc_angle']
# label = pd.get_dummies(df['is_iceberg']).values
# train_y, val_y = label.loc[train_index, :].values, label.loc[~train_index, :].values
label = pd.get_dummies(df['is_iceberg']).values

test_ratio = 0.2
nr_runs = 3
split_seed = 25
kf = StratifiedShuffleSplit(n_splits=nr_runs, test_size=test_ratio, train_size=None, random_state=split_seed)
model_file = 'weights_v2.hdf5'
cnn_model = create_resnet()

# train the model
batch_size = 64
epoch_num = 80        # 5
step_per_epoch = 2**14 / batch_size
early_stop = EarlyStopping(monitor='val_loss', patience=10)

gen_images = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=.3,
    height_shift_range=.3,
    zoom_range=.1,
    rotation_range=20
)

for r, (train_index, val_index) in enumerate(kf.split(bands, label)):
    print('\nround {:04d} of {:04d}, seed={}'.format(r + 1, nr_runs, split_seed))

    tmp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    x1, x2 = bands[train_index, :, :], bands[val_index, :, :]
    y1, y2 = label[train_index, :], label[val_index, :]

    model_file = 'weights_{}_v{}.hdf5'.format(tmp, r+1)
    check_point = ModelCheckpoint(model_file, monitor='val_loss', verbose=0, save_best_only=True)

    tic = time.time()
    history = cnn_model.fit_generator(
        gen_images.flow(x=x1, y=y1, batch_size=batch_size, seed=123),
        step_per_epoch,
        epochs=epoch_num,
        validation_data=(x2, y2),
        callbacks=[check_point, early_stop]
    )
    toc = time.time()
    print('This round model cost time: {:.2f} minutes'.format((toc - tic) / 60))
    model = load_model(model_file)
    prediction = model.predict(x2)
    print('Validation accuracy score: {:.2f}'.format(accuracy_score(y2[:, 1], prediction[:, 1] > .5)))
    print('Validation log loss: {:.2f}'.format(log_loss(y2[:, 1], prediction[:, 1])))

    # prediction the test dataset
    y_pred_test = model.predict(test_bands)
    sub = pd.concat([test_df['id'], pd.Series(y_pred_test[:, 1])], axis=1)
    sub.columns = ['id', 'is_iceberg']
    tmp = 'sub_{}_cnn_v{}.csv'.format(tmp, r+1)
    sub.to_csv(tmp, index=False)
