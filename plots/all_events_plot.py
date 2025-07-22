import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pylab import figure
import seaborn as sns
from matplotlib.lines import Line2D

os.chdir('..')
BASE_DIR = os.getcwd()
os.chdir('plots')

sns.set_palette("RdBu")
sns.set(context='talk', style='ticks')

font = {'family': 'Trebuchet MS',
        'color':  'black', #'darked'
        'weight': 'normal',
        'size': 15
        }

plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)


path1 = os.path.join(BASE_DIR, 'simulations/test_results/all_events/significant_events_T_0.1.xlsx')
path2 = os.path.join(BASE_DIR, 'simulations/test_results/all_events/stationary_events_T_0.1.xlsx')
excel1 = pd.ExcelFile(path1)
excel2 = pd.ExcelFile(path2)
major, stationary = {}, {}

for i in range(1, 9):
    df1 = pd.read_excel(excel1, f'Turbine_{i}')
    df2 = pd.read_excel(excel2, f'Turbine_{i}')
    major[f'Turbine_{i}'] = df1
    stationary[f'Turbine_{i}'] = df2

#major ---------------
fig = figure(figsize=(15, 8))
colors = ['red', 'lawngreen', 'black', 'orange', 'blue', 'saddlebrown', 'indigo', 'turquoise']
for i in range(len(major)):
    y = np.array(major['Turbine_' + str(i + 1)]['∆w_m'])
    x = np.array(major['Turbine_' + str(i + 1)]['∆t_m'])
    plt.scatter(y, x, s=10, c=colors[i])

plt.legend(['Turbine 1', 'Turbine 2', 'Turbine 3', 'Turbine 4', 'Turbine 5', 'Turbine 6', 'Turbine 7', 'Turbine 8'],
           fontsize=12
           , loc="upper right", frameon=False)

fig.text(0.5, 0.06, '∆t [hours]', fontdict=font, ha='center', va='center')
fig.text(0.08, 0.5, '∆w [per unit]', fontdict=font, ha='center', va='center', rotation='vertical')
fig.savefig(r"plotted_figures/allMajorEvents.png", dpi=300, bbox_inches='tight')

#stationary ----------
fig = figure(figsize=(15, 8))

for i in range(len(stationary)):
    y = np.array(stationary['Turbine_' + str(i + 1)]['σ_s'])
    x = np.array(stationary['Turbine_' + str(i + 1)]['∆t_s'])
    plt.scatter(y, x, s=10, c=colors[i])

plt.legend(['Turbine 1', 'Turbine 2', 'Turbine 3', 'Turbine 4', 'Turbine 5', 'Turbine 6', 'Turbine 7', 'Turbine 8'],
           fontsize=12, loc="upper right", frameon=False)


fig.text(0.5, 0.06, '∆t [hours]', fontdict=font, ha='center', va='center')
fig.text(0.08, 0.5, 'σ [per unit]', fontdict=font, ha='center', va='center', rotation='vertical')
fig.savefig(r"plotted_figures/allStationaryEvents.png", dpi=300, bbox_inches='tight')
plt.show()

