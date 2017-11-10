# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from functools import reduce
from datetime import datetime


def reduce_func(x, y):
    return pd.merge(x, y)


blend_files = [
    "sub_20171108_cnn_default.csv",
    "subm_blend009_2017-11-04-13-32.csv"
]

# data = [pd.read_csv(f).rename(columns={'is_iceberg': f}) for f in blend_files]
#
# data = reduce(reduce_func, data)
# for col in data.columns.tolist()[1:]:
#     data[col] = data[col].clip(lower=0.005, upper=0.995)
# data['is_iceberg'] = data.iloc[:, 1:].mean(axis=1)
#
# t = datetime.today().strftime('%Y%m%d')
# filename = 'sub_{}_xgb_cnn_blend_with_clip.csv'.format(t)
# data[['id', 'is_iceberg']].to_csv(filename, index=False)


blend_files.extend([
    "sub_2017-11-08-12-22_cnn_v1.csv",
    "sub_2017-11-08-17-55_cnn_v2.csv",
    "sub_2017-11-09-02-02_cnn_v3.csv"
])
print(blend_files)

blend_files = [
    "sub_20171108_xgb_cnn_blend_with_clip.csv",
    "sub_2017-11-08-12-22_cnn_v1.csv",
    "sub_2017-11-08-17-55_cnn_v2.csv",
    "sub_2017-11-09-02-02_cnn_v3.csv"
]


data = [pd.read_csv(f).rename(columns={'is_iceberg': f}) for f in blend_files]
data = reduce(reduce_func, data)
# for col in data.columns.tolist()[1:]:
#     data[col] = data[col].clip(lower=0.05, upper=0.95)
# data['is_iceberg'] = data.iloc[:, 1:].mean(axis=1)

data['cnn_shuffle'] = np.dot(data.iloc[:, 2:], [.3, .3, .4])

data['is_iceberg'] = np.dot(data.iloc[:, [1, -1]], [.75, .25])
data['is_iceberg'] = data['is_iceberg'].clip(0.05, 0.95)

t = datetime.today().strftime('%Y%m%d%H%M%S')
filename = 'sub_{}_xgb_cnn_4_blend_with_clip2.csv'.format(t)
data[['id', 'is_iceberg']].to_csv(filename, index=False)
