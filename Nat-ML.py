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

linreg = LinearRegression()
linreg.fit(disx_tr, disy_tr)
cap_pred = linreg.predict(disx_val)
print("Linear Regression MAE:", mean_absolute_error(disy_val, cap_pred))
print("Linear Regression MSE:", mean_squared_error(disy_val, cap_pred)) 
print("Linear Regression R2:", r2_score(disy_val, cap_pred))
dis_val['pred_lr_cap'] = cap_pred

def lr_capacity_for_battery(df, battery_id):
    d = df[df["battery_id"] == battery_id]

    if d.empty:
        print(f"No data for battery_id {battery_id}")
        return

    plt.figure(figsize=(12,5))

    x = np.arange(len(d))

    plt.plot(x, d["Capacity"], label="True Capacity", linewidth=3)
    plt.plot(x, d["pred_lr_cap"], label="RF Prediction", linestyle="--", color='pink', linewidth=3)

    plt.title(f"Capacity Prediction – Battery {battery_id}")
    plt.xlabel("Cycle (validation subset)")
    plt.ylabel("Capacity (Ah)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print(bat_val)
lr_capacity_for_battery(dis_val, 'B0049')

target = 'RUL'
rulx_tr = dis_tr[features_physics].values
ruly_tr = dis_tr[target].values
rulx_val = dis_val[features_physics].values
ruly_val = dis_val[target].values

linreg.fit(rulx_tr, ruly_tr)
rul_pred = linreg.predict(rulx_val)
print("Linear Regression MAE:", mean_absolute_error(ruly_val, rul_pred))
print("Linear Regression MSE:", mean_squared_error(ruly_val, rul_pred))
print("Linear Regression R2:", r2_score(ruly_val, rul_pred))
dis_val['pred_lr_rul'] = rul_pred

def lr_rul_for_battery(df, battery_id):
    d = df[df["battery_id"] == battery_id]

    if d.empty:
        print(f"No data for battery_id {battery_id}")
        return

    plt.figure(figsize=(12,5))

    x = np.arange(len(d))

    plt.plot(x, d["RUL"], label="True RUL", linewidth=3)
    plt.plot(x, d["pred_lr_rul"], label="RF Prediction", linestyle="--", color='pink', linewidth=3)

    plt.title(f"RUL Prediction – Battery {battery_id}")
    plt.xlabel("Cycle (validation subset)")
    plt.ylabel("RUL (Cycles)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
lr_rul_for_battery(dis_val, 'B0049')
