import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pylab import figure
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.patches as patches

sns.set_palette("RdBu")
sns.set(context='talk', style='ticks')


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


#%% plotting

limit = 120
y = np.arange(limit)

fig = figure(figsize=(15, 8))


for k in range(2):

    m = [x / nominal for x in original[:limit, 0]] #normalized
    plt.subplot(1, 2, k + 1)
    plt.plot(y, m, color='b')
    plt.grid('on')
    plt.plot(m)
    plt.gca().set_axisbelow(True)
    plt.gca().xaxis.grid(True, linestyle='--')
    plt.gca().yaxis.grid(False)
    '''
    #major--------------------
    for i in range(len([f'Turbine_{k + 1}'])):
        major_per_turbine = major[f'Turbine_{k + 1}']
        a = int(major_per_turbine.iloc[int(i), 1])
        b = int(major_per_turbine.iloc[int(i), 2]) + 1
        if a >= limit or b >= limit:
            break
        if major_per_turbine.loc[int(i), '∆w_m'] > 0:
            plt.plot(y[a:b], m[a:b], color='g')
        else:
            plt.plot(y[a:b], m[a:b], color='r')
    
        plt.gca().set_axisbelow(True)
        plt.gca().yaxis.grid(True, linestyle='--')
        plt.gca().xaxis.grid(False)
        #plt.title('Ramp Events with 4% Threshold')

    #stationary---------------------
    for i in range(len(stationary[f'Turbine_{k + 1}'])):
        stationary_per_turbine = stationary[f'Turbine_{k + 1}']
        a = int(stationary_per_turbine.iloc[int(i), 1])
        b = int(stationary_per_turbine.iloc[int(i), 2]) + 1
        if a >= limit or b >= limit:
            break
        sigma = stationary_per_turbine.iloc[int(i), 4] - 0.075
        if stationary_per_turbine.loc[int(i), '∆t_s'] > 10:
            plt.plot(y[a:b], m[a:b], color='y')

        #width = len(m[a:b])

        fig.patches.extend([plt.Rectangle((m[a], sigma), width=1, height=0.15,
                                          fill=True, color='b', alpha=0.5, zorder=1000,
                                          transform=fig.transFigure, figure=fig)])'''

    # plt.title('Ramp Events with 4% Threshold')


legend_elements = [Line2D([0], [0], color='k', label='Half-cycle down'),
                   Line2D([0], [0], color='r', label='Half-cycle up'),
                   Line2D([0], [0], color='g', label='Full cycle')]

plt.legend(handles=legend_elements, prop={'size': 12}, bbox_to_anchor=(.1, 1.05),
           ncol=3, loc="center", frameon=False)
# Set common labels
fig.text(0.5, 0.06, 'Time [hours]', fontdict=font, ha='center', va='center')
fig.text(0.08, 0.5, 'Wind power generation [per unit]', fontdict=font, ha='center', va='center', rotation='vertical')
plt.show()


fig.savefig("plotted_figures/New_RainflowEvents.pdf", bbox_inches='tight')