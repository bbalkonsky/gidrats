#!/usr/bin/env python
# coding: utf-8

import pickle

import pandas as pd
import numpy as np
from tqdm import tqdm

# file = r'sources/910.xlsm'
file = r'sources/11.xlsm'
df = pd.read_excel(file)

df['C_P_tr'] = .0001
df['C_P_zatr'] = .0001
df['C_T_izm'] = .0001
df['R_P_tr'] = .0001
df['R_P_zatr'] = .0001
df['R_T_izm'] = .0001
df['v_P_tr'] = .0001
df['v_P_zatr'] = .0001
df['v_T_izm'] = .0001
df['C1_v_P_tr'] = .0001
df['C1_v_P_zatr'] = .0001
df['C1_v_T_izm'] = .0001
df['C2_v_P_tr'] = .0001
df['C2_v_P_zatr'] = .0001
df['C2_v_T_izm'] = .0001
df['a_P_tr'] = .0001
df['a_P_zatr'] = .0001
df['a_T_izm'] = .0001
df['C1_a_P_tr'] = .0001
df['C1_a_P_zatr'] = .0001
df['C1_a_T_izm'] = .0001
df['C2_a_P_tr'] = .0001
df['C2_a_P_zatr'] = .0001
df['C2_a_T_izm'] = .0001
df['RSI_function_tr'] = .0001
df['RSI_function_zatr'] = .0001
df['RSI_function_temp'] = .0001
df['history_tr'] = .0001
df['history_zatr'] = .0001
df['history_temp'] = .0001

df['drop_for_train'] = 0

#df['label'] = .0
df['label1'] = .0
df['label2'] = .0
df['label3'] = .0
df['label4'] = .0
df['label5'] = .0
df['label6'] = .0
df['label7'] = .0

User_C = 100
User_C1_v = 10
User_C2_v = 15
User_C1_a = 10
User_C2_a = 15
User_w_v_P_tr = 30
User_w_v_P_zatr = 100
User_w_v_T_izm = 50 * 10
User_w_a_P_tr = 50
User_w_a_P_zatr = 100
User_w_a_T_izm = 100
time_reg = 10  # 10 or 1
period_RSI = 200
d1 = -1
d2 = 1
period_history_model = 200

time_out_i = 0
close_well = 0

slice_length = max(User_C, User_C1_v, User_C2_v, User_C1_a, User_C2_a, User_w_v_P_tr, User_w_v_P_zatr, User_w_v_T_izm,
                   User_w_a_P_tr, User_w_a_P_zatr, User_w_a_T_izm, time_reg,
                   period_RSI, d1, d2, period_history_model)

df_sliced = None

# df = df[:501]


def slice_by_period(period):
    if period >= len(df_sliced):
        return df_sliced
    else:
        return df_sliced[len(df_sliced) - period:]


def f_param(param, param_old, param_w):
    return ((param - param_old) / time_reg) * param_w


def f_percent(param_1, param_2):
    return (param_1 / (param_2 + 0.00001)) - 1


def normalization(param):
    return 1 if param > 0 else -1 if param < 0 else .0


C1_v_P_tr = .0
C1_v_P_zatr = .0
C1_v_T_izm = .0

for idx, row in tqdm(df.iterrows(), total=df.shape[0]):  # progress bar
    if row['P_tr'] - row['P_dikt1'] < 5:
        close_well = 0
        time_out_i = time_out_i + 1
        if time_out_i > 120:
            if idx >= slice_length:
                df_sliced = df[idx - slice_length + 1: idx + 1]
            else:
                df_sliced = df[:idx + 1]

            # average
            mean = slice_by_period(User_C)
            df.at[idx, 'C_P_tr'] = mean['P_tr'].mean()
            df.at[idx, 'C_P_zatr'] = mean['P_zatr'].mean()
            df.at[idx, 'C_T_izm'] = mean['T_izm'].mean()

            # param - average of param
            df.at[idx, 'R_P_tr'] = normalization(f_percent(df.at[idx, 'C_P_tr'], df.at[idx, 'P_tr']))
            df.at[idx, 'R_P_zatr'] = normalization(f_percent(df.at[idx, 'C_P_zatr'], df.at[idx, 'P_zatr']))
            df.at[idx, 'R_T_izm'] = normalization(f_percent(df.at[idx, 'C_T_izm'], df.at[idx, 'T_izm']))

            # first derivative
            if idx > 0:
                df.at[idx, 'v_P_tr'] = f_param(df.loc[idx]['P_tr'], df.loc[idx - 1]['P_tr'], User_w_v_P_tr)
                df.at[idx, 'v_P_zatr'] = f_param(df.loc[idx]['P_zatr'], df.loc[idx - 1]['P_zatr'], User_w_v_P_zatr)
                df.at[idx, 'v_T_izm'] = f_param(df.loc[idx]['T_izm'], df.loc[idx - 1]['T_izm'], User_w_v_T_izm)

            # average С1 for first derivative
            C1_v_P_tr_old = C1_v_P_tr
            C1_v_P_zatr_old = C1_v_P_zatr
            C1_v_T_izm_old = C1_v_T_izm
            der_mean = slice_by_period(User_C1_v)
            C1_v_P_tr = der_mean['v_P_tr'].mean()
            C1_v_P_zatr = der_mean['v_P_zatr'].mean()
            C1_v_T_izm = der_mean['v_T_izm'].mean()
            df.at[idx, 'C1_v_P_tr'] = normalization(C1_v_P_tr)
            df.at[idx, 'C1_v_P_zatr'] = normalization(C1_v_P_zatr)
            df.at[idx, 'C1_v_T_izm'] = normalization(C1_v_T_izm)

            # average С2 for first derivative
            der_mean = slice_by_period(User_C2_v)
            df.at[idx, 'C2_v_P_tr'] = normalization(der_mean['v_P_tr'].mean())
            df.at[idx, 'C2_v_P_zatr'] = normalization(der_mean['v_P_zatr'].mean())
            df.at[idx, 'C2_v_T_izm'] = normalization(der_mean['v_T_izm'].mean())

            # second derivative of C1 average
            if idx > 0:
                df.at[idx, 'a_P_tr'] = f_param(C1_v_P_tr, C1_v_P_tr_old, User_w_a_P_tr)
                df.at[idx, 'a_P_zatr'] = f_param(C1_v_P_zatr, C1_v_P_zatr_old, User_w_a_P_zatr)
                df.at[idx, 'a_T_izm'] = f_param(C1_v_T_izm, C1_v_T_izm_old, User_w_a_T_izm)

            # average С1 for second derivative
            der_mean = slice_by_period(User_C1_a)
            df.at[idx, 'C1_a_P_tr'] = normalization(der_mean['a_P_tr'].mean())
            df.at[idx, 'C1_a_P_zatr'] = normalization(der_mean['a_P_zatr'].mean())
            df.at[idx, 'C1_a_T_izm'] = normalization(der_mean['a_T_izm'].mean())

            # average С2 for second derivative
            der_mean = slice_by_period(User_C2_a)
            df.at[idx, 'C2_a_P_tr'] = normalization(der_mean['a_P_tr'].mean())
            df.at[idx, 'C2_a_P_zatr'] = normalization(der_mean['a_P_zatr'].mean())
            df.at[idx, 'C2_a_T_izm'] = normalization(der_mean['a_T_izm'].mean())

            # RSI
            if idx < period_RSI:
                x_max_in_period = df[['P_tr', 'P_zatr', 'T_izm']][0:idx + 1].max()
                x_min_in_period = df[['P_tr', 'P_zatr', 'T_izm']][0:idx + 1].min()
            else:
                x_max_in_period = df[['P_tr', 'P_zatr', 'T_izm']][idx - period_RSI:idx + 1].max()
                x_min_in_period = df[['P_tr', 'P_zatr', 'T_izm']][idx - period_RSI:idx + 1].min()
            df.at[idx, 'RSI_function_tr'] = (((row['P_tr'] - x_min_in_period['P_tr']) * (d2 - d1)) / (
                    x_max_in_period['P_tr'] - x_min_in_period['P_tr'] + 0.00001)) + d1
            df.at[idx, 'RSI_function_zatr'] = (((row['P_zatr'] - x_min_in_period['P_zatr']) * (d2 - d1)) / (
                    x_max_in_period['P_zatr'] - x_min_in_period['P_zatr'] + 0.00001)) + d1
            df.at[idx, 'RSI_function_temp'] = (((row['T_izm'] - x_min_in_period['T_izm']) * (d2 - d1)) / (
                    x_max_in_period['T_izm'] - x_min_in_period['T_izm'] + 0.00001)) + d1

            # history model
            sum_history_tr = 0
            sum_history_zatr = 0
            sum_history_temp = 0

            if idx > period_history_model:
                df_history_slice = df_sliced[len(df_sliced) - period_history_model: -1]
                p_tr = row['P_tr']
                p_tr_sum = []
                p_zatr = row['P_zatr']
                p_zatr_sum = []
                t_izm = row['T_izm']
                t_izm_sum = []
                for index, history_row in df_history_slice.iterrows():
                    p_tr_sum.append(f_percent(p_tr, history_row['P_tr']))
                    p_zatr_sum.append(f_percent(p_zatr, history_row['P_zatr']))
                    t_izm_sum.append(f_percent(t_izm, history_row['T_izm']))

                df.at[idx, 'history_tr'] = normalization(np.mean(p_tr_sum))
                df.at[idx, 'history_zatr'] = normalization(np.mean(p_zatr_sum))
                df.at[idx, 'history_temp'] = normalization(np.mean(t_izm_sum))

            # ------------ label -------------
            
            if (df.at[idx, 'C1_v_P_tr'] < 0) and (df.at[idx, 'C1_v_P_zatr'] > 0) and (df.at[idx, 'C1_v_T_izm'] < 0):
                df.at[idx, 'label1'] = 1
            
            if (df.at[idx, 'C2_v_P_tr'] < 0) and (df.at[idx, 'C2_v_P_zatr'] > 0) and (df.at[idx, 'C2_v_T_izm'] < 0):
                df.at[idx, 'label2'] = 1
            
            if (df.at[idx, 'C1_a_P_tr'] < 0) and (df.at[idx, 'C1_a_P_zatr'] > 0) and (df.at[idx, 'C1_a_T_izm'] < 0):
                df.at[idx, 'label3'] = 1
            
            if (df.at[idx, 'C2_a_P_tr'] < 0) and (df.at[idx, 'C2_a_P_zatr'] > 0) and (df.at[idx, 'C2_a_T_izm'] < 0):
                df.at[idx, 'label4'] = 1
                
            if (df.at[idx, 'RSI_function_tr'] == -1) and (df.at[idx, 'RSI_function_zatr'] == 1) and (df.at[idx, 'RSI_function_temp'] == -1):
                df.at[idx, 'label5'] = 1
                
            if (df.at[idx, 'history_tr'] < 0) and (df.at[idx, 'history_zatr'] > 0) and (df.at[idx, 'history_temp'] < 0):
                df.at[idx, 'label6'] = 1
                
            if (df.at[idx, 'R_P_tr'] > 0) and (df.at[idx, 'R_P_zatr'] < 0) and (df.at[idx, 'R_T_izm'] > 0):
                df.at[idx, 'label7'] = 1
                
        else:
            df.at[idx, 'drop_for_train'] = 1
    else:
        df.at[idx, 'drop_for_train'] = 1
    if row['P_tr'] - row['P_dikt1'] > 5:
        close_well = close_well + 1
        if close_well > 60:
            time_out_i = 0

# df = df[df['drop_for_train'] == 0]

# df.to_excel('outputs/for_train.xlsx')
# pickle.dump(df, open('outputs/for_train.pkl', 'wb'))
df.to_excel('outputs/for_test.xlsx')
pickle.dump(df, open('outputs/for_test.pkl', 'wb'))