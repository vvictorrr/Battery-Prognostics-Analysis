import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import preprocessor as pp 
path = 'cleaned_dataset/'
df = pp.read_clean_file(path)
df_discharges = pp.get_discharges_phyiscs(path, df)
df_discharges = pp.merge_impedance_with_discharges(df, df_discharges, path)


df_discharges = df_discharges.copy()
df_discharges['first_cycle_time'] = df_discharges.groupby('battery_id')['start_time'].transform('min')
df_discharges['elapsed_days'] = (df_discharges['start_time'] - df_discharges['first_cycle_time']).dt.total_seconds() / (3600*24)
# df_discharges['relative_age'] = df_discharges.groupby('battery_id')['elapsed_days'].transform(lambda x: x / x.max())

print(df_discharges.describe())

groups = {
    'A': ['B0025', 'B0026', 'B0027', 'B0028'],
    'B': ['B0029', 'B0030', 'B0031', 'B0032'],
    'C': ['B0033', 'B0034', 'B0036'],
    'D': ['B0038', 'B0039', 'B0040'],
    'E': ['B0041', 'B0042', 'B0043', 'B0044'],
    'F': ['B0045', 'B0046', 'B0047', 'B0048'],
    'G': ['B0049', 'B0050', 'B0051', 'B0052'],
    'H': ['B0053', 'B0054', 'B0055', 'B0056'],
    'I': ['B0005', 'B0006', 'B0007', 'B0018']}

# finding basic correlations 
numeric_cols = df_discharges.select_dtypes(include=[np.number])
corr_matrix = numeric_cols.corr()
print('\nStrong correlations (|r| >= 0.5):\n')

passed = set()

for col1 in corr_matrix.columns:
    for col2 in corr_matrix.columns:
        if col1 == col2:
            continue 
        pair = tuple(sorted([col1, col2]))
        if pair in passed:
            continue 
        r = corr_matrix.loc[col1, col2]
        if abs(r) >= 0.5:
            print(f'{col1} vs {col2}: r = {r:.2f}')
            passed.add(pair)

plt.figure(figsize=(8,6))
plt.imshow(corr_matrix, cmap='plasma', interpolation='nearest')
plt.colorbar(label='Correlation')
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# def normalize(series):
#     'min-max to 0-1'
#     if series.max() == series.min():
#         return series * 0  
#     return (series - series.min()) / (series.max() - series.min())

def plot_group_subplots(group_name, group_df):
    # normalize stats per battery
    df_norm = group_df.copy()
    for battery, bdf in df_norm.groupby('battery_id'):
        idx = bdf.index

        df_norm.loc[idx, 'Capacity'] = (bdf['Capacity'])
        df_norm.loc[idx, 'mean_voltage'] = (bdf['mean_voltage'])
        df_norm.loc[idx, 'mean_temperature'] = (bdf['mean_temperature'])
        df_norm.loc[idx, 'mean_current'] = (bdf['mean_current'])
        df_norm.loc[idx, 'cycle_number'] = (bdf['cycle_number'])
        df_norm['cycle_number_norm'] = 0.0  
        df_norm.loc[idx, 'cycle_number_norm'] = (bdf['cycle_number'].astype(float))
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'{group_name} Battery Relationships', fontsize=16)
    

    # c vs cycles
    ax = axes[0,0]
    for battery, bdf in df_norm.groupby('battery_id'):
        ax.plot(bdf['cycle_number'], bdf['Capacity'], alpha=0.7, label=battery)
    ax.set_title('Capacity vs Cycle Number')
    ax.set_xlabel('Cycle Number')
    ax.set_ylabel('Capacity')
    ax.grid(alpha=0.3)
    ax.legend(title="Battery ID", fontsize=8)

    # v vs cycles
    ax = axes[0,1]
    for battery, bdf in df_norm.groupby('battery_id'):
        ax.plot(bdf['cycle_number'], bdf['mean_voltage'], alpha=0.7, label=battery)
    ax.set_title('Voltage vs Cycle Number')
    ax.set_xlabel('Cycle Number')
    ax.set_ylabel('Voltage')
    ax.grid(alpha=0.3)
    ax.legend(title="Battery ID", fontsize=8)

    # t vs cycles
    ax = axes[0,2]
    for battery, bdf in df_norm.groupby('battery_id'):
        ax.plot(bdf['cycle_number'], bdf['mean_temperature'], alpha=0.7, label=battery)
    ax.set_title('Temperature vs Cycle Number')
    ax.set_xlabel('Cycle Number')
    ax.set_ylabel('Temperature')
    ax.grid(alpha=0.3)
    ax.legend(title="Battery ID", fontsize=8)

    # c vs t
    ax = axes[1,0]
    ax.scatter(df_norm['mean_temperature'], df_norm['Capacity'], alpha=0.5, color='purple')
    ax.set_title('Capacity vs Temperature')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Capacity')
    ax.grid(alpha=0.3)


    # i vs t
    ax = axes[1,1]
    ax.scatter(df_norm['mean_temperature'], df_norm['mean_current'], alpha=0.5, color='green')
    ax.set_title('Current vs Temperature')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Current')
    ax.grid(alpha=0.3)

    ax = axes[1,2]
    sc = ax.scatter(df_norm['cycle_number'], df_norm['Capacity'], alpha=0.5, c=df_norm['mean_temperature'])
    fig.colorbar(sc, ax=ax, label='mean temperature (C)')
    ax.set_title('cycle vs capacity')
    ax.set_xlabel('cycle number')
    ax.set_ylabel('capacity')
    ax.grid(alpha=0.3)


  

    #axes[1,2].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

for group_name, battery_list in groups.items():
    group_df = df_discharges[df_discharges['battery_id'].isin(battery_list)]
    if len(group_df) > 0:
        plot_group_subplots(group_name, group_df)
