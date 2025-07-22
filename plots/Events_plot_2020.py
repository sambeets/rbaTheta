import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pylab import figure
import seaborn as sns
sns.set_palette("RdBu")
sns.set(context='talk', style='ticks')
from matplotlib.lines import Line2D
import matplotlib.patches as patches

font = {'family': 'Trebuchet MS',
        'color':  'black', #'darked'
        'weight': 'normal',
        'size': 15
        }
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

#%% Data import
nominal = 2.5
original = pd.read_excel(r'input_data\new_8_wind_turbine_data.xlsx', sheet_name=0)
original = original.iloc[:, 1:].values

path1 = r'simulations\test_results\all_events\significant_events_T_0.1.xlsx' #r'all_events\significant_events_T_0.15_fc_0.3.xlsx'
path2 = r'simulations\test_results\all_events\stationary_events_T_0.1_new.xlsx'#r'all_events\stationary_events_T_0.15_fc_0.3.xlsx'
excel1 = pd.ExcelFile(path1)
excel2 = pd.ExcelFile(path2)
major, stationary = {}, {}
for i in range(1, 9):
    df1 = pd.read_excel(excel1, f'Turbine_{i}')
    df2 = pd.read_excel(excel2, f'Turbine_{i}')
    major[f'Turbine_{i}'] = df1
    stationary[f'Turbine_{i}'] = df2

#%% plotting

limit = 120
y = np.arange(limit)

fig = figure(figsize=(15, 8))


for k in range(len(major)):

    m = [x / nominal for x in original[:limit, k]] #normalized
    plt.subplot(2, 4, k + 1)
    plt.plot(y, m, color='b')
    plt.grid('on')

# Significant/major
    for i in range(len(major[f'Turbine_{k + 1}'])):
        major_per_turbine = major[f'Turbine_{k + 1}']
        a = int(major_per_turbine.iloc[int(i), 1])
        b = int(major_per_turbine.iloc[int(i), 2]) + 1
        if a >= limit or b >= limit:
            break
        if m[a] > m[b]:
            plt.plot(y[a:b], m[a:b], color='turquoise', linewidth=10 , alpha =0.4, antialiased=True)
        else:
            plt.plot(y[a:b], m[a:b], color='tomato', linewidth=10 , alpha =0.4, antialiased=True)
        plt.gca().set_axisbelow(True)
        plt.gca().yaxis.grid(True, linestyle='--')
        plt.gca().xaxis.grid(False)
        #plt.title('Ramp Events with 4% Threshold')

# =============================================================================
# stationary
    for i in range(len(stationary[f'Turbine_{k + 1}'])):
        stationary_per_turbine = stationary[f'Turbine_{k + 1}']
        a = int(stationary_per_turbine.iloc[int(i), 1])
        b = int(stationary_per_turbine.iloc[int(i), 2]) + 1
        if a >= limit or b >= limit:
            break
        sigma = stationary_per_turbine.iloc[int(i), 4] - 0.075
        if stationary_per_turbine.loc[int(i), '∆t_s'] > 10: # modify 10 to 15...
            plt.plot(y[a:b], m[a:b], color='k', linewidth=10 , alpha =0.4, antialiased=True)

        #width = len(m[a:b])

        '''fig.patches.extend([plt.Rectangle((m[a], sigma), width=1, height=0.15,
                                          fill=True, color='b', alpha=0.5, zorder=1000,
                                          transform=fig.transFigure, figure=fig)])'''
# =============================================================================


legend_elements = [Line2D([0], [0], color='b', label='Original data'),
                   Line2D([0], [0], color='tomato', label='Up-ramp events', linewidth=10 , alpha =0.4, antialiased=True),
                   Line2D([0], [0], color='turquoise', label='Down-ramp events',  linewidth=10 , alpha =0.4, antialiased=True),
                   Line2D([0], [0], color='k', label='Stationary events',  linewidth=10 , alpha =0.4, antialiased=True)
                   ]

plt.legend(handles=legend_elements, prop={'size': 12}, bbox_to_anchor=(-1.5, 2.25),
           ncol=4, loc="center", frameon=False)
# Set common labels
fig.text(0.5, 0.06, 'Time [hours]', fontdict=font, ha='center', va='center')
fig.text(0.08, 0.5, 'Wind power generation [per unit]', fontdict=font, ha='center', va='center', rotation='vertical')
#fig.savefig(r"plotted_figures/RBAevents.png", dpi = 300, bbox_inches='tight')