import pandas as pd
import statistics as st
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_palette("RdBu")
sns.set(context='talk', style='ticks')
font = {'family': 'Trebuchet MS',
        'color':  'black', #'darked'
        'weight': 'normal',
        'size': 15
        }
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


#number of events plot
#initializing
major_number, stationary_number = {}, {}
for i in range(1, 9):
    major_number[f'Turbine_{i}'] = []
    stationary_number[f'Turbine_{i}'] = []


for threshold in thresholds:
    path1 = f'simulations/test_results/multiple_T/major_T{threshold}.xlsx'
    path2 = f'simulations/test_results/multiple_T/stationary_T{threshold}.xlsx'
    excel1 = pd.ExcelFile(path1)
    excel2 = pd.ExcelFile(path2)
    for i in range(1, 9):
        df1 = pd.read_excel(excel1, f'Turbine_{i}')
        df2 = pd.read_excel(excel2, f'Turbine_{i}')
        major_number[f'Turbine_{i}'].append(len(df1))
        stationary_number[f'Turbine_{i}'].append(len(df2))


#deltat plot
#initializing
major_max, stationary_max = {}, {}
for i in range(1, 9):
    major_max[f'Turbine_{i}'] = []
    stationary_max[f'Turbine_{i}'] = []


for threshold in thresholds:
    path1 = f'simulations/test_results/multiple_T/major_T{threshold}.xlsx'
    path2 = f'simulations/test_results/multiple_T/stationary_T{threshold}.xlsx'
    excel1 = pd.ExcelFile(path1)
    excel2 = pd.ExcelFile(path2)
    for i in range(1, 9):
        df1 = pd.read_excel(excel1, f'Turbine_{i}')
        df2 = pd.read_excel(excel2, f'Turbine_{i}')
        major_max[f'Turbine_{i}'].append(max(df1['σ_m']))
        stationary_max[f'Turbine_{i}'].append(max(df2['σ_s']))

for i in range(1, 9):
    plt.figure(1, figsize=(15, 8))
    plt.subplot(2, 2, 1)
    plt.plot(thresholds, major_number[f'Turbine_{i}'])
    plt.gca().set_axisbelow(True)
    plt.gca().yaxis.grid(True, linestyle='--')
    plt.tick_params(direction='in', length=1, width=1, colors='b')
    plt.title('Significant events', fontsize=15)
    plt.xlabel('Threshold', fontsize=15)
    plt.ylabel('Number of events', fontsize=15)
    
    plt.subplot(2, 2, 2)
    plt.plot(thresholds, stationary_number[f'Turbine_{i}'])
    plt.gca().set_axisbelow(True)
    plt.gca().yaxis.grid(True, linestyle='--')
    plt.tick_params(direction='in', length=1, width=1, colors='b')
    plt.title('Stationary events', fontsize=15)
    plt.xlabel('Threshold', fontsize=15)
    plt.ylabel('Number of events', fontsize=15)
    
    plt.subplot(2, 2, 3)
    plt.plot(thresholds, major_max[f'Turbine_{i}'])
    plt.gca().set_axisbelow(True)
    plt.gca().yaxis.grid(True, linestyle='--')
    plt.tick_params(direction='in', length=1, width=1, colors='b')
    plt.title('Significant events', fontsize=15)
    plt.xlabel('Threshold', fontsize=15)
    plt.ylabel('Maximum among events', fontsize=15)

    plt.subplot(2, 2, 4)
    plt.plot(thresholds, stationary_max[f'Turbine_{i}'])
    plt.gca().set_axisbelow(True)
    plt.gca().yaxis.grid(True, linestyle='--')
    plt.tick_params(direction='in', length=1, width=1, colors='b')
    plt.title('Stationary events', fontsize=15)
    plt.xlabel('Threshold', fontsize=15)
    plt.ylabel('Maximum among events', fontsize=15)
    
    plt.subplots_adjust(wspace=0.12, hspace=0.3)
    plt.savefig('threshold_test', bbox_inches = 'tight')
plt.show()

