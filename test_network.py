import pickle

import numpy as np
import pandas as pd
from tensorflow import keras
from tqdm import tqdm
import tensorflow as tf

df = pickle.load(open('outputs/for_test.pkl', 'rb'))

features_considered = ['label', 'P_tr', 'P_zatr', 'T_izm', 'v_P_tr', 'v_P_zatr', 'v_T_izm', 'a_P_tr', 'a_P_zatr',
                       'a_T_izm',
                       'RSI_function_tr', 'RSI_function_zatr', 'RSI_function_temp']

model = keras.models.load_model('outputs/model_ts.h5')

features = df[features_considered]
features.index = df['time']

dataset = features.values


def multivariate_data(dataset, start_index, history_size,
                      target_size, step):
    data = []

    start_index = start_index + history_size
    end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])

    return np.array(data)


to_predict = multivariate_data(dataset, 0, 200, 2, 1)
predictions = []

for row in tqdm(to_predict, total=to_predict.shape[0]):
    predictions.append(model.predict(np.expand_dims(row, axis=0))[0])


# test_predictions = model.predict(df).flatten()
predictions_df = pd.DataFrame(predictions)
predictions_df.to_excel('outputs/predictions.xlsx')
