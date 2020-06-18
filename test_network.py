import pickle

from tensorflow import keras
from tqdm import tqdm

df = pickle.load(open('outputs/for_test.pkl', 'rb'))
df = df[
    ['R_P_tr', 'R_P_zatr', 'R_T_izm', 'C1_v_P_tr', 'C1_v_P_zatr', 'C1_v_T_izm', 'C2_v_P_tr',
     'C2_v_P_zatr', 'C2_v_T_izm',
     'C1_a_P_tr', 'C1_a_P_zatr', 'C1_a_T_izm', 'C2_a_P_tr', 'C2_a_P_zatr', 'C2_a_T_izm', 'RSI_function_tr',
     'RSI_function_zatr', 'RSI_function_temp', 'history_tr', 'history_zatr', 'history_temp']]

model = keras.models.load_model('outputs/model.h5')


for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    model.predict(row.to_frame().T)
#     вот тут делай чо хочешь


# test_predictions = model.predict(df).flatten()
# predictions_df = pd.DataFrame(test_predictions)
# predictions_df.to_excel('outputs/predictions.xlsx')
