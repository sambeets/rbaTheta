
"""
Testing all the data set
"""
import os
import pandas as pd
import multiprocessing
import core.model as model
from core.helpers import save_xls, pre_markov, normalize

os.chdir('..')
BASE_DIR = os.getcwd()
path = os.path.join(BASE_DIR, r'input_data/new_8_wind_turbine_data.xlsx')

wind_data = pd.read_excel(path)
wind_data = wind_data.iloc[:100, 1:]

#test wıth multıple thresholds
thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for t in thresholds:
    [significant_events, stationary_events, tao] = model.RBA_theta(data=wind_data,
                                                            nominal=2.5,
                                                            s=0.01, k=3, fc=0.3, threshold=t)

    save_xls(significant_events, f'simulations/test_results/multiple_T/major_T{t}.xlsx')
    save_xls(stationary_events, f'simulations/test_results/multiple_T/stationary_T{t}.xlsx')



'''
[significant_events, stationary_events, rainflow_events, tao] = model.RBA_theta(data=wind_data,
                                                                             nominal=2.5,
                                                                             s=0.01,
                                                                             k=3,
                                                                             fc=0.1,
                                                                             threshold=0.1)


save_xls(significant_events, f'test_results/shortset/significant_events_T_0.1_shortset.xlsx')
save_xls(stationary_events, f'test_results/shortset/stationary_events_T_0.1_shortset.xlsx')
save_xls(rainflow_events, f'test_results/shortset/rainflow_events_T_0.1_shortset.xlsx')
'''

'''
#to get variance of multiple t and fc values
thresholds = [0.01, 0.1, 0.3, 0.5]
cutoffs = [0.1, 0.3, 0.9]

for t in thresholds:
    [significant_events, stationary_events, rainflow_events, tao] = model.RBA_theta(data=wind_data,
                                                                                 nominal=2500,
                                                                                 s=0.01,
                                                                                 k=3,
                                                                                 fc=0.3,
                                                                                 threshold=t)
    writer = pd.ExcelWriter(f'test_results/table1_major_T{t}_variance.xlsx')
    significant_events["Turbine_1"].describe().to_excel(writer)
    writer.save()
    writer = pd.ExcelWriter(f'test_results/table1_stationary_T{t}_variance.xlsx')
    stationary_events["Turbine_1"].describe().to_excel(writer)
    writer.save()

for fc in cutoffs:
    [significant_events, stationary_events, rainflow_events, tao] = model.RBA_theta(data=wind_data,
                                                                                 nominal=2500,
                                                                                 s=0.01,
                                                                                 k=3,
                                                                                 fc=fc,
                                                                                 threshold=0.15)
    writer = pd.ExcelWriter(f'test_results/table2_major_fc{fc}_variance.xlsx')
    significant_events["Turbine_1"].describe().to_excel(writer)
    writer.save()
    writer = pd.ExcelWriter(f'test_results/table12_stationary_fc{fc}_variance.xlsx')
    stationary_events["Turbine_1"].describe().to_excel(writer)
    writer.save()
'''






