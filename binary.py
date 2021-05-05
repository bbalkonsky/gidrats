import numpy
import pandas as pd
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

df = pickle.load(open('outputs/for_train.pkl', 'rb'))
# df = df[:10000]
seed = 5
numpy.random.seed(seed)

prediction_var = ['R_P_tr', 'R_P_zatr', 'R_T_izm', 'C1_v_P_tr', 'C1_v_P_zatr', 'C1_v_T_izm', 'C2_v_P_tr',
     'C2_v_P_zatr', 'C2_v_T_izm',
     'C1_a_P_tr', 'C1_a_P_zatr', 'C1_a_T_izm', 'C2_a_P_tr', 'C2_a_P_zatr', 'C2_a_T_izm', 'RSI_function_tr',
     'RSI_function_zatr', 'RSI_function_temp', 'history_tr', 'history_zatr', 'history_temp']


X = df[prediction_var].values
Y = df['label'].values

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)


model = Sequential()
model.add(Dense(40, activation='relu', kernel_initializer='random_normal', input_dim=len(prediction_var)))
model.add(Dense(20, activation='relu', kernel_initializer='random_normal'))
model.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

#Compiling the neural network
model.compile(optimizer ='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fitting the data to the training dataset
model.fit(X, Y, batch_size=10, epochs=50)

eval_model = model.evaluate(X, encoded_Y)
print(eval_model)
model.save('outputs/model.h5')
