import time
import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import core.model
from core.helpers import save_xls, pre_markov, normalize


path1 = r'simulations/test_results/all_events/significant_events_T_0.1.xlsx'
path2 = r'simulations/test_results/all_events/stationary_events_T_0.1.xlsx'
excel1 = pd.ExcelFile(path1)
excel2 = pd.ExcelFile(path2)
major, stationary = {}, {}
for i in range(1, 9):
    df1 = pd.read_excel(excel1, f'Turbine_{i}')
    df2 = pd.read_excel(excel2, f'Turbine_{i}')
    major[f'Turbine_{i}'] = df1
    stationary[f'Turbine_{i}'] = df2

'''for i in range(1, 9):
    print(f'Max deltawm in Turbine_{i} is {major[f"Turbine_{i}"].loc[major[f"Turbine_{i}"]["∆w_m"].idxmax()]}')
    print(f'Max deltatm in Turbine_{i} is {major[f"Turbine_{i}"].loc[major[f"Turbine_{i}"]["∆t_m"].idxmax()]}')
    print(f'Max deltats in Turbine_{i} is {stationary[f"Turbine_{i}"].loc[stationary[f"Turbine_{i}"]["∆t_s"].idxmax()]}')

'''
#distributions of attributes

for i in range(1, 9):

    plt.plot(major[major[f"Turbine_{i}"]]['∆t_m'])
    plt.title('∆t_m')
    plt.show()
    kwargs = dict(alpha=0.5, bins=100, density=True, stacked=True)
    plt.hist(major[major[f"Turbine_{i}"]]['∆t_m'], **kwargs, color='g', label='Ideal')
plt.show()

plt.plot(major['Turbine_2']['∆w_m'])
plt.title('∆w_m')
plt.show()
plt.hist(major['Turbine_2']['∆w_m'], **kwargs, color='g', label='Ideal')
plt.show()

plt.plot(major['Turbine_2']['θ_m'])
plt.title('θ_m')
plt.show()
plt.hist(major['Turbine_2']['θ_m'], **kwargs, color='g', label='Ideal')
plt.show()

plt.plot(major['Turbine_2']['σ_m'])
plt.title('σ_m')
plt.show()
plt.hist(major['Turbine_2']['σ_m'], **kwargs, color='g', label='Ideal')
plt.show()

plt.plot(stationary['Turbine_2']['∆t_s'])
plt.title('∆t_s')
plt.show()
plt.hist(stationary['Turbine_2']['∆t_s'], **kwargs, color='g', label='Ideal')
plt.show()

plt.plot(stationary['Turbine_2']['σ_s'])
plt.title('σ_s')
plt.show()
plt.hist(stationary['Turbine_2']['σ_s'], **kwargs, color='g', label='Ideal')
plt.show()