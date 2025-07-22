
"""
Reconstructing the original time-series array from the events
"""

import pandas as pd
import numpy as np
import scipy.interpolate as interpolate
from sklearn.metrics import mean_squared_error
import multiprocessing
import matplotlib.pyplot as plt
import core.model as model
from core.helpers import save_xls, pre_markov, normalize

'''
input_path = r'input_data/new_8_wind_turbine_data.xlsx'
events_path = r'test_results/multiple_T/major_T0.1.xlsx'
nominal = 2.5
length = 100
wind_data = pd.read_excel(input_path)
original_data = wind_data.iloc[:length, 1] #1st Turbine

major_data = pd.read_excel(events_path, sheet_name='Turbine_1')
def limit_func(df, end):
    for k in range(len(df)):
        if df.loc[k, 't1'] > end:
            stop = k
            break
    return df.iloc[:stop, :]

significant_events = limit_func(end=length, df= major_data)

x, y = [], []

for i in range(len(significant_events)):
    start = int(significant_events.loc[i, 't1'])
    stop = int(significant_events.loc[i, 't2'])
    start_amp = significant_events.loc[i, 'w_m(t1)'] * nominal
    stop_amp = significant_events.loc[i, 'w_m(t2)'] * nominal
    num = stop - start
    time_span = np.linspace(start=start, num=num + 1, stop=stop)
    section = np.linspace(start=start_amp, num=num + 1, stop=stop_amp)
    x.extend(time_span)
    y.extend(section)

x = np.array(x)
y = np.array(y)


t, c, k = interpolate.splrep(x, y, s=0.01, k=3)
print(''''''\
            t: {}
            c: {}
            k: {}
            ''''''.format(t, c, k))
N = len(original_data)
xmin, xmax = x.min(), x.max()
xx = np.linspace(xmin, xmax, N)
spline = interpolate.BSpline(t, c, k, extrapolate=False)

original = (original_data / 2.5)[:length]
splined = (spline(xx) / 2.5)[:length]
difference = original_data - spline(xx)
print(original_data[40:50])
print(spline(xx)[40:50])
print(difference[40:50])

mse = mean_squared_error(original, splined)
print(mse)

plt.figure(1, figsize=(15, 8))
plt.plot(original_data, label='Original')
plt.bar(xx, difference, label='Differences')
plt.plot(xx, spline(xx), 'r', label='Reconstructed')
plt.gca().set_axisbelow(True)
plt.gca().yaxis.grid(True, linestyle='--')
plt.tick_params(direction='in', length=1, width=1, colors='b')
plt.xlabel('Time')
plt.ylabel('Power production')
plt.legend(loc='best')
#plt.savefig('plotted_figures/recon.png', dpi =300, bbox_inches='tight')
plt.show()
'''

'''
input_path = r'input_data/new_8_wind_turbine_data.xlsx'
events_path = r'test_results/multiple_T/major_T0.1.xlsx'
nominal = 2.5
length = 100
wind_data = pd.read_excel(input_path)
original_data = wind_data.iloc[:length, 1] #1st Turbine

major_data = pd.read_excel(events_path, sheet_name='Turbine_1')

def limit_func(df, end):
    for k in range(len(df)):
        if df.loc[k, 't1'] > end:
            stop = k
            break
    return df.iloc[:stop, :], stop

loop_over = (len(original_data) // length) + 1
splined = []
for m in range(loop_over):
    significant_events, stop = limit_func(end=length*(m+1), df= major_data)
    print(significant_events)
    major_data = major_data.iloc[stop:, :].reset_index()

    x, y = [], []

    for i in range(len(significant_events)):
        start = int(significant_events.loc[i, 't1'])
        stop = int(significant_events.loc[i, 't2'])
        start_amp = significant_events.loc[i, 'w_m(t1)'] * nominal
        stop_amp = significant_events.loc[i, 'w_m(t2)'] * nominal
        num = stop - start
        time_span = np.linspace(start=start, num=num + 1, stop=stop)
        section = np.linspace(start=start_amp, num=num + 1, stop=stop_amp)
        x.extend(time_span)
        y.extend(section)

    x = np.array(x)
    y = np.array(y)


    t, c, k = interpolate.splrep(x, y, s=0.01, k=3)
    print(''''''\
                t: {}
                c: {}
                k: {}
                '''''''.format(t, c, k))
    N = length
    xmin, xmax = x.min(), x.max()
    xx = np.linspace(xmin, xmax, N)
    spline = interpolate.BSpline(t, c, k, extrapolate=False)
    splined.append(spline(xx))
    #difference = original_data - spline(xx)

    #print(difference[40:50])

mse = mean_squared_error(original_data, splined)
print(mse)

plt.figure(1, figsize=(15, 8))
plt.plot(original_data, label='Original')
plt.bar(xx, difference, label='Differences')
plt.plot(xx, spline(xx), 'r', label='Reconstructed')
plt.gca().set_axisbelow(True)
plt.gca().yaxis.grid(True, linestyle='--')
plt.tick_params(direction='in', length=1, width=1, colors='b')
plt.xlabel('Time')
plt.ylabel('Power production')
plt.legend(loc='best')
#plt.savefig('plotted_figures/recon.png', dpi =300, bbox_inches='tight')
plt.show()
'''

from scipy.interpolate import interp1d

input_path = r'.nput_data/new_8_wind_turbine_data.xlsx'
events_path = r'test_results/multiple_T/major_T0.1.xlsx'
nominal = 2.5
#length = 100
wind_data = pd.read_excel(input_path)
original_data = wind_data.iloc[:, 1] #1st Turbine

significant_events = pd.read_excel(events_path, sheet_name='Turbine_1')
print(sum(significant_events['∆t_m']) / len(significant_events))
y = []

for i in range(len(significant_events) - 1):
    if i == 0:
        start = int(significant_events.loc[i, 't1'])
        if start != 0:
            for k in range(len(start)):
                y.append(None)
    start = int(significant_events.loc[i, 't1'])
    stop = int(significant_events.loc[i, 't2'])
    next_start = int(significant_events.loc[i + 1, 't1'])
    start_amp = significant_events.loc[i, 'w_m(t1)'] * nominal
    stop_amp = significant_events.loc[i, 'w_m(t2)'] * nominal
    num = stop - start
    section = np.linspace(start=start_amp, num=num, stop=stop_amp)
    y.extend(section)
    if stop != next_start:
        for k in range(next_start - stop):
            y.append(None)

start = int(significant_events['t1'].iloc[-1])
stop = int(significant_events['t2'].iloc[-1])
start_amp = significant_events['w_m(t1)'].iloc[-1] * nominal
stop_amp = significant_events['w_m(t2)'].iloc[-1] * nominal
num = stop - start
time_span = np.linspace(start=start, num=num + 1, stop=stop)
section = np.linspace(start=start_amp, num=num + 1, stop=stop_amp)
y.extend(section)

y = pd.DataFrame(y)
mse = mean_squared_error((original_data[:len(y)] / 2.5), (y.interpolate() / 2.5))
print(mse)

x = list(range(100))
original_data = original_data[:100].values
interpolated = [item for sublist in y.interpolate()[:100].values for item in sublist]
difference = original_data - interpolated

plt.figure(1, figsize=(15, 8))
plt.plot(x, original_data, label='Original')
plt.bar(range(len(difference)), difference, color='g', label='Differences')
plt.plot(x, interpolated, 'r', label='Reconstructed')
plt.gca().set_axisbelow(True)
plt.gca().yaxis.grid(True, linestyle='--')
plt.tick_params(direction='in', length=1, width=1, colors='b')
plt.xlabel('Time')
plt.ylabel('Power production')
plt.legend(loc='best')
plt.savefig('plotted_figures/recon.png', dpi =300, bbox_inches='tight')
plt.show()

print(len(significant_events))



'''
plt.figure(1, figsize=(15, 8))
plt.plot(original_data, label='Original')
plt.bar(xx, difference, label='Differences')
plt.plot(xx, spline(xx), 'r', label='Reconstructed')
plt.gca().set_axisbelow(True)
plt.gca().yaxis.grid(True, linestyle='--')
plt.tick_params(direction='in', length=1, width=1, colors='b')
plt.xlabel('Time')
plt.ylabel('Power production')
plt.legend(loc='best')
#plt.savefig('plotted_figures/recon.png', dpi =300, bbox_inches='tight')
plt.show()
'''