#importable python file

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
    return df

def extract_cycle_features(filepath, cycle):
    """
    Input: 
        filepath to the 'cleaned_dataset' folder
        df_cycle = a single discharge CSV filename (time series)
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

    return df_discharges