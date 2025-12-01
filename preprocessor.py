#importable python file

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def parse_datetime(array_str):
    vals = np.fromstring(array_str.strip('[]'), sep=' ')
    return datetime(
        int(vals[0]), int(vals[1]), int(vals[2]),
        int(vals[3]), int(vals[4]),
        int(vals[5]), int((vals[5] % 1) * 1_000_000)
    )

def read_clean_file(filepath):
    """
    input: filepath to the 'cleaned_dataset' folder
    ex: 'user/downloads/cleaned_dataset'
    """
    filepath += '/metadata.csv'
    df = pd.read_csv(filepath)
    df['start_time'] = df['start_time'].apply(parse_datetime)

    if 'Capacity' in df.columns:
        df['Capacity'] = pd.to_numeric(df['Capacity'], errors='coerce')
    elif 'Re' in df.columns:
        df['Re'] = pd.to_numeric(df['Re'], errors='coerce')
    elif 'Rct' in df.columns:
        df['Rct'] = pd.to_numeric(df['Rct'], errors='coerce') 
        
    return df

def extract_cycle_features(filepath, cycle):
    """
    Input: 
        filepath to the 'cleaned_dataset' folder
        cycle = a single discharge CSV filename (time series)
    Output: dictionary of aggregated features for ML
    """
    filepath += '/data/' + cycle
    df_cycle = pd.read_csv(filepath)
    
    features = {}
    features['mean_voltage'] = df_cycle['Voltage_measured'].mean()
    features['max_voltage'] = df_cycle['Voltage_measured'].max()
    features['min_voltage'] = df_cycle['Voltage_measured'].min()
    
    features['mean_current'] = df_cycle['Current_measured'].mean()
    features['max_current'] = df_cycle['Current_measured'].max()
    
    features['mean_temperature'] = df_cycle['Temperature_measured'].mean()
    features['max_temperature'] = df_cycle['Temperature_measured'].max()
    
    features['discharge_time'] = df_cycle['Time'].max() - df_cycle['Time'].min()

    #Internal resistance estimate R = (VOC - V_load) / I

    #dataset has a tendency for discharges to start with really low current
    threshold = 0.1 
    valid_rows = df_cycle[np.abs(df_cycle['Current_load']) > threshold]
    
    if len(valid_rows) > 0:
        i_start = valid_rows.index[0]
    else:
        # fallback if dataset is weird
        i_start = df_cycle.index[5]
    
    I0 = abs(df_cycle.loc[i_start, 'Current_measured'])
    V0 = df_cycle.loc[i_start, 'Voltage_measured']
    
    # NASA method assumption for open-circuit voltage
    VOC = V0 + 0.04  #empirical offset used in PHM08 papers
    #internal resistance
    features['r_internal'] = (VOC - V0) / I0 if I0 > 0 else np.nan

    #mean slope dV/dt
    dVdt = np.gradient(df_cycle['Voltage_measured'], df_cycle['Time'])
    features['mean_dvdt'] = np.mean(dVdt)

    #Delivered capacity Ah = integral(I / 3600)dt
    capacity_As = np.trapz(
                abs(df_cycle['Current_measured']),
                df_cycle['Time']
            )
    features['capacity_Ah'] = capacity_As / 3600

    voltage_drop = (
                df_cycle['Voltage_measured'].iloc[0] -
                df_cycle['Voltage_measured'].iloc[-1]
            )
    features['voltage_drop'] = voltage_drop
    
    
    return features

def get_discharges(filepath, df):
    """
    input: 
        filepath to the 'cleaned_dataset' folder
        main df
    output: df with only discharges and added cycles and cycle features
    """
    df_discharges = df[df['type'] == 'discharge'][['start_time', 'ambient_temperature', 'battery_id', 'uid', 'filename', 'Capacity']].copy()
    df_discharges['cycle_number'] = df_discharges.groupby('battery_id').cumcount() + 1
    
    mean_voltage = []
    max_voltage = []
    min_voltage = []
    
    mean_current = []
    max_current = []
    
    mean_temp = []
    max_temp = []
    discharge_time = []
    
    for _, row in df_discharges.iterrows():
        try:
            file = row['filename']
            feats = extract_cycle_features(filepath, file)
            mean_voltage.append(feats['mean_voltage'])
            max_voltage.append(feats['max_voltage'])
            min_voltage.append(feats['min_voltage'])
            mean_current.append(feats['mean_current'])
            max_current.append(feats['max_current'])
            mean_temp.append(feats['mean_temperature'])
            max_temp.append(feats['max_temperature'])
            discharge_time.append(feats['discharge_time'])
        except Exception as e:
            print(f'error: {e}')
            mean_voltage.append(None)
            max_voltage.append(None)
            min_voltage.append(None)
            mean_current.append(None)
            max_current.append(None)
            mean_temp.append(None)
            max_temp.append(None)
            discharge_time.append(None)
    df_discharges['mean_voltage'] = mean_voltage
    df_discharges['max_voltage'] = max_voltage
    df_discharges['min_voltage'] = min_voltage
    df_discharges['mean_current'] = mean_current
    df_discharges['max_current'] = max_current
    df_discharges['mean_temperature'] = mean_temp
    df_discharges['max_temperature'] = max_temp
    df_discharges['discharge_time'] = discharge_time

    df_discharges = df_discharges.dropna()
    return df_discharges


def get_discharges_phyiscs(filepath, df, nominal_capacity=2.0, eol_capacity=1.4):
    """
    input: 
        filepath to the 'cleaned_dataset' folder
        main df
    output: df with only discharges and added cycles and cycle features
    nominal capacity: stated in datased readme
    eol_capacity: end of life capacity, also stated in dataset readme. When capacity reaches 70% of nominal capacity
    """
    df_discharges = df[df['type'] == 'discharge'][['start_time', 'ambient_temperature', 'battery_id', 'uid', 'filename', 'Capacity']].copy()
    df_discharges['cycle_number'] = df_discharges.groupby('battery_id').cumcount() + 1

    df_discharges["SOH"] = df_discharges["Capacity"] / nominal_capacity

    df_discharges["EOL_cycle"] = df_discharges.groupby("battery_id")["cycle_number"].transform("max")
    df_discharges["RUL"] = df_discharges["EOL_cycle"] - df_discharges["cycle_number"]
    
    mean_voltage = []
    max_voltage = []
    min_voltage = []
    
    mean_current = []
    max_current = []
    
    mean_temp = []
    max_temp = []
    discharge_time = []

    R_internal = []
    mean_dVdt = []
    capacity_Ah = []
    capacity_ratio = []
    voltage_drop = []
    
    for _, row in df_discharges.iterrows():
        try:
            file = row['filename']
            feats = extract_cycle_features(filepath, file)
            mean_voltage.append(feats['mean_voltage'])
            max_voltage.append(feats['max_voltage'])
            min_voltage.append(feats['min_voltage'])
            mean_current.append(feats['mean_current'])
            max_current.append(feats['max_current'])
            mean_temp.append(feats['mean_temperature'])
            max_temp.append(feats['max_temperature'])
            discharge_time.append(feats['discharge_time'])
            R_internal.append(feats['r_internal'])
            mean_dVdt.append(feats['mean_dvdt'])
            capacity_Ah.append(feats['capacity_Ah'])
            voltage_drop.append(feats['voltage_drop'])

            if pd.notna(row['Capacity']) and row['Capacity'] > 0:
                capacity_ratio.append(feats['capacity_Ah'] / row['Capacity'])
            else:
                capacity_ratio.append(np.nan)

        except Exception as e:
            print(f'error: {e}')
            mean_voltage.append(np.nan)
            max_voltage.append(np.nan)
            min_voltage.append(np.nan)
            mean_current.append(np.nan)
            max_current.append(np.nan)
            mean_temp.append(np.nan)
            max_temp.append(np.nan)
            discharge_time.append(np.nan)
            R_internal.append(np.nan)
            mean_dVdt.append(np.nan)
            capacity_Ah.append(np.nan)
            capacity_ratio.append(np.nan)
            voltage_drop.append(np.nan)
    df_discharges['mean_voltage'] = mean_voltage
    df_discharges['max_voltage'] = max_voltage
    df_discharges['min_voltage'] = min_voltage
    df_discharges['mean_current'] = mean_current
    df_discharges['max_current'] = max_current
    df_discharges['mean_temperature'] = mean_temp
    df_discharges['max_temperature'] = max_temp
    df_discharges['discharge_time'] = discharge_time
    df_discharges['r_internal'] = R_internal
    df_discharges['mean_dvdt'] = mean_dVdt
    df_discharges['capacity_Ah'] = capacity_Ah
    df_discharges['capacity_ratio'] = capacity_ratio
    df_discharges['voltage_drop'] = voltage_drop

    df_discharges = df_discharges.dropna()

    return df_discharges

def impedance_features(filepath, cycle):
    """
    Extract usable impedance features from complex-valued EIS CSV
    """

    filepath += '/data/' + cycle
    df_cycle = pd.read_csv(filepath)

    '''
    complex_cols = [
        "Sense_current",
        "Battery_current",
        "Current_ratio",
        "Battery_impedance",
        "Rectified_Impedance"
    ]'''
    complex_cols = [
        "Battery_impedance",
        "Rectified_Impedance"
    ]

    # Convert "(a+bj)" strings which are complex numbers to something workable
    for col in complex_cols:
        df_cycle[col] = df_cycle[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else np.nan)

    # Extract magnitudes, real, imag, phases
    feats = {}

    Z = df_cycle["Battery_impedance"].dropna()

    if len(Z) > 0:
        feats["Battery_impedance_real_mean"] = np.real(Z).mean()
        feats["Battery_impedance_mag_mean"]  = np.abs(Z).mean()
    else:
        feats["Battery_impedance_real_mean"] = np.nan
        feats["Battery_impedance_mag_mean"]  = np.nan

    Zr = df_cycle["Rectified_Impedance"].dropna()

    if len(Zr) > 0:
        feats["Rectified_impedance_real_mean"] = np.real(Zr).mean()
        feats["Rectified_impedance_phase_mean"] = np.angle(Zr).mean()
    else:
        feats["Rectified_impedance_real_mean"] = np.nan
        feats["Rectified_impedance_phase_mean"] = np.nan

    return feats

def merge_impedance_with_discharges(df, df_dis, filepath):
    """
    Adds impedance features to each discharge event.
    Matches the nearest impedance **before** the discharge start_time.
    """

    # Separate impedance rows
    df_imp = df[df["type"] == "impedance"].copy()

    # Extract impedance features for each impedance file
    imp_feat_list = []

    for _, row in df_imp.iterrows():
        feats = impedance_features(filepath, row["filename"])
        feats["start_time"] = row["start_time"]
        feats["battery_id"] = row["battery_id"]
        imp_feat_list.append(feats)

    df_imp_feat = pd.DataFrame(imp_feat_list)

    # Sort both by time, should already be done but just to catch
    df_imp_feat = df_imp_feat.sort_values("start_time")
    df_dis = df_dis.sort_values("start_time")

    #primary merge, first impedence before a discharge
    merged_back = pd.merge_asof(
        df_dis,
        df_imp_feat,
        on="start_time",
        by="battery_id",
        direction="backward"
    )

    #secondary merge, impedence after a discharge
    merged_forward = pd.merge_asof(
        df_dis,
        df_imp_feat,
        on="start_time",
        by="battery_id",
        direction="forward"
    )

    imp_cols = [
        "Battery_impedance_real_mean",
        "Battery_impedance_mag_mean",
        "Rectified_impedance_real_mean",
        "Rectified_impedance_phase_mean"
    ]

    #if one didn't have a impedence before it (first discharge), it fills with the forwards
    for col in imp_cols:
        merged_back[col] = merged_back[col].fillna(merged_forward[col])

    return merged_back