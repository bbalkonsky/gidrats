import pickle

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

df = pickle.load(open('outputs/for_train.pkl', 'rb'))

df = df[
    ['R_P_tr', 'R_P_zatr', 'R_T_izm', 'C1_v_P_tr', 'C1_v_P_zatr', 'C1_v_T_izm', 'C2_v_P_tr',
     'C2_v_P_zatr', 'C2_v_T_izm',
     'C1_a_P_tr', 'C1_a_P_zatr', 'C1_a_T_izm', 'C2_a_P_tr', 'C2_a_P_zatr', 'C2_a_T_izm', 'RSI_function_tr',
     'RSI_function_zatr', 'RSI_function_temp', 'label', 'history_tr', 'history_zatr', 'history_temp']]


train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index)

train_labels = train_dataset.pop('label')
test_labels = test_dataset.pop('label')


# сюда вставить нормализацию (если нужна)

def build_model():
    model = keras.Sequential([
        layers.Dense(50, activation='tanh', input_shape=[len(train_dataset.keys())]),
        layers.Dense(100, activation='tanh'),
        layers.Dense(150, activation='tanh'),
        layers.Dense(120, activation='tanh'),
        layers.Dense(70, activation='tanh'),
        layers.Dense(35, activation='tanh'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


model = build_model()
model.summary()


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


EPOCHS = 1000

history = model.fit(
    train_dataset, train_labels,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print('\n')
print(hist.tail())

model.save('outputs/model.h5')
