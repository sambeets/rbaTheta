
"""
Comparison of size of the original array and the events representation
"""

import pandas as pd

input_path = r'input_data/new_8_wind_turbine_data.xlsx'
events_path = r'test_results/multiple_T/major_T0.1.xlsx'

original_data = pd.read_excel(input_path)
original_data = original_data.iloc[:, 1] #1st Turbine
event_data = pd.read_excel(events_path)

original_data.to_csv(f'compression/original.csv', index=False)
event_data[['t1', 't2', 'w_m(t1)', 'w_m(t2)']].to_csv(f'compression/extracted.csv', index=False)


