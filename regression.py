import pickle

import pandas as pd
from tensorflow import keras

df = pickle.load(open('outputs/for_train.pkl', 'rb'))

df = df[
    ['R_P_tr', 'R_P_zatr', 'R_T_izm', 'C1_v_P_tr', 'C1_v_P_zatr', 'C1_v_T_izm', 'C2_v_P_tr',
     'C2_v_P_zatr', 'C2_v_T_izm',
     'C1_a_P_tr', 'C1_a_P_zatr', 'C1_a_T_izm', 'C2_a_P_tr', 'C2_a_P_zatr', 'C2_a_T_izm', 'RSI_function_tr',
     'RSI_function_zatr', 'RSI_function_temp', 'label', 'history_tr', 'history_zatr', 'history_temp']]

# train_dataset = df.sample(frac=0.8, random_state=0)
# test_dataset = df.drop(train_dataset.index)

labels = df.pop('label')
# test_labels = test_dataset.pop('label')


def build_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(50, activation='tanh', input_shape=[len(df.keys())]))
    model.add(keras.layers.Dense(100, activation='tanh'))
    model.add(keras.layers.Dense(150, activation='tanh'))
    model.add(keras.layers.Dense(120, activation='tanh'))
    model.add(keras.layers.Dense(70, activation='tanh'))
    model.add(keras.layers.Dense(35, activation='tanh'))
    model.add(keras.layers.Dense(1))

    optimizer = keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse', 'accuracy'])
    return model


model = build_model()
model.summary()

history = model.fit(df, labels,
                    epochs=1000, validation_split=0.2)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print('\n')
print(hist.tail())

model.save('outputs/model.h5')
