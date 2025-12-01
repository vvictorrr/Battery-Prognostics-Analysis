import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from scipy.optimize import curve_fit

import preprocessor as pp

import importlib
importlib.reload(pp)

path = 'cleaned_dataset/'
df = pp.read_clean_file(path)
df_dis = pp.get_discharges_phyiscs(path, df)
df_dis = pp.merge_impedance_with_discharges(df, df_dis, path)

BATTERIES = sorted(df['battery_id'].value_counts().index.tolist())
print(len(BATTERIES))
bat_tr, bat_te = train_test_split(BATTERIES, test_size=0.28, random_state=42)
print(bat_tr)
print(bat_te)
df_tr = df[df['battery_id'].isin(bat_tr)]
df_te = df[df['battery_id'].isin(bat_te)]


dis_tr = pp.get_discharges_phyiscs(path, df_tr)
dis_te = pp.get_discharges_phyiscs(path, df_te)
bat_tr, bat_val = train_test_split(BATTERIES,test_size=0.2,random_state=21)
print(bat_tr)
print(bat_val)

df_tr = df[df['battery_id'].isin(bat_tr)]
df_val = df[df['battery_id'].isin(bat_val)]
dis_tr = pp.get_discharges_phyiscs(path, df_tr)
dis_val = pp.get_discharges_phyiscs(path, df_val)
dis_tr = pp.merge_impedance_with_discharges(df_tr, dis_tr, path)
dis_val = pp.merge_impedance_with_discharges(df_val, dis_val, path)

features = ['ambient_temperature', 
            'cycle_number',
            'mean_voltage',
            'max_voltage',
            'min_voltage',
            'mean_current',
            'max_current',
            'mean_temperature',
            'max_temperature',
            'discharge_time']
target = 'Capacity'
disx_tr = dis_tr[features].values
disy_tr = dis_tr[target].values
disx_val = dis_val[features].values
disy_val = dis_val[target].values

features_physics = ['ambient_temperature', 
            'cycle_number',
            'mean_voltage',
            'max_voltage',
            'min_voltage',
            'mean_current',
            'max_current',
            'mean_temperature',
            'max_temperature',
            'discharge_time',
            'r_internal', 
            'mean_dvdt', 
            'voltage_drop']
target = 'Capacity'

disx_tr = dis_tr[features_physics].values
disy_tr = dis_tr[target].values
disx_val = dis_val[features_physics].values
disy_val = dis_val[target].values

target = 'RUL'
rulx_tr = dis_tr[features_physics].values
ruly_tr = dis_tr[target].values
rulx_val = dis_val[features_physics].values
ruly_val = dis_val[target].values
linreg = LinearRegression()
linreg.fit(rulx_tr, ruly_tr)
rul_pred = linreg.predict(rulx_val)

print("Linear Regression MAE:", mean_absolute_error(ruly_val, rul_pred))
print("Linear Regression MSE:", mean_squared_error(ruly_val, rul_pred))
print("Linear Regression R2:", r2_score(ruly_val, rul_pred))

