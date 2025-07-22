import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pylab import figure, scatter, show

nominal = 2500
original = pd.read_excel(r'C:\Users\EsinÖren\Dropbox\RBA theta\RBA_theta\code\input_data\8_wind_turbine_data.xlsx', sheet_name=0).values


path1 = r'C:\Users\EsinÖren\Dropbox\RBA theta\RBA_theta\code\test_results\all_events\significant_events_T_0.15_fc_0.3.xlsx'
path2 = r'C:\Users\EsinÖren\Dropbox\RBA theta\RBA_theta\code\test_results\all_events\stationary_events_T_0.15_fc_0.3.xlsx'
excel1 = pd.ExcelFile(path1)
excel2 = pd.ExcelFile(path2)
major, stationary = {}, {}
for i in range(1, 9):
    df1 = pd.read_excel(excel1, f'Turbine_{i}')
    df2 = pd.read_excel(excel2, f'Turbine_{i}')
    major[f'Turbine_{i}'] = df1
    stationary[f'Turbine_{i}'] = df2


limit = 100
y = np.arange(limit)
fig = plt.figure(figsize=(20, 20))

for k in range(len(major)):

    m = [x / nominal for x in original[:limit, k]] #normalized
    plt.subplot(2, 4, k + 1)
    plt.plot(y, m, color='b')
    plt.grid('on')

    #major--------------------
    for i in range(len(major[f'Turbine_{k + 1}'])):
        major_per_turbine = major[f'Turbine_{k + 1}']
        a = int(major_per_turbine.iloc[int(i), 1])
        b = int(major_per_turbine.iloc[int(i), 2]) + 1
        if a >= limit or b >= limit:
            break
        if m[a] > m[b]:
            plt.plot(y[a:b], m[a:b], color='g')  # linewidth=1,
        else:
            plt.plot(y[a:b], m[a:b], color='r')

        plt.legend(['Power output', 'down events', 'up events'], loc=0)

    # plt.title('Ramp Events with 4% Threshold')

    #stationary---------------------
    for i in range(len(stationary[f'Turbine_{k + 1}'])):
        stationary_per_turbine = stationary[f'Turbine_{k + 1}']
        a = int(stationary_per_turbine.iloc[int(i), 1])
        b = int(stationary_per_turbine.iloc[int(i), 2]) + 1
        if a >= limit or b >= limit:
            break
        sigma = stationary_per_turbine.iloc[int(i), 4] - 0.075
        #plt.plot(y[a:b], m[a:b], color='y')

        fig.patches.extend([plt.Rectangle((m[a], sigma), width=len(m[a:b]), height=0.15,
                                          fill=True, color='b', alpha=0.5, zorder=1000,
                                          transform=fig.transFigure, figure=fig)])

        plt.legend(['Power output', 'stationary event'], loc=0)
        plt.xlabel('Time [/h]')
        plt.ylabel('Power [W/W]')
    # plt.title('Ramp Events with 4% Threshold')



fig.tight_layout()
#fig.savefig("C:\\Users\\esino\\Desktop\\events.pdf", bbox_inches='tight')
plt.show()