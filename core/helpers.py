"""Helper function to aid the core functionalities"""
from typing import Union

import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
import math
import mapclassify as mc

def normalize(data, nominal):
    """
    Normalizes data between [0,1]
    """
    normalized = [x / nominal for x in data]
    return normalized


def bspline(series, s, k):
    """Data smoothing using b-spline"""
    x = np.arange(len(series))
    t, c, k = interpolate.splrep(x, series, s=s, k=k)
    # t=vector of knots, c=bspline coeff, k=degree of spline

    N = 10 * len(x)  # makes the data /minute
    xmin, xmax = x.min(), x.max()
    xx = np.linspace(xmin, xmax, N)
    spline = interpolate.BSpline(t, c, k, extrapolate=False)
    bsplined = interpolate.splev(xx, [t, c, k], der=0)
    return bsplined

#sampling_time
def filter_blackman(data, fc):
    '''
    Blackman filter
    takes in a timeseries and cut off frequency. gives it out low pass filtered
    '''
    b = 0.08
    N = int(np.ceil((4 / b)))
    if not N % 2 : N += 1
    n = np.arange(N)
 
    sinc_func = np.sinc(2 * fc * (n - (N - 1) / 2.))
    window = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
    sinc_func = sinc_func * window
    sinc_func = sinc_func / np.sum(sinc_func)

    filtered = np.convolve(data, sinc_func)
    filtered = filtered[25:-25]

    return filtered

def rainflow_count(data):
    '''
    Finds the total count of rainflow cycles in given data.
    '''
    prev = data[0]
    delta = []
    for item in data[1:]:
        delta.append(item - prev)
        prev = item

    start = 0
    maximum = [data[start], start]
    sign = lambda x: (1, -1)[x < 0]

    for i in range(len(delta) - 1):
        if sign(delta[i]) == sign(delta[i + 1]):
            continue
        else:
            new_event = [data[i + 1], i + 1]
            maximum.append(new_event)
            start = i + 1

    maximum.append([data[len(data) - 1], len(data) - 1])

    # rainflow with Power range and time range of cycles

    l = len(maximum)  # total size of input array
    array_out = np.zeros((l - 1, 8))  # initialize output array
    pr = 0  # index of input array
    po = 0  # index of output array
    j = -1  # index of temporary array "a"
    a = np.empty((len(maximum), 2))  # temporary array for algorithm

    for i in range(l):

        j += 1  # increment "a" counter
        a[j] = maximum[pr]  # put turning point into temporary array
        pr += 1  # increment input array pointer

        while ((j >= 2) & (np.fabs(a[j - 1, 0] - a[j - 2, 0]) <= np.fabs(a[j, 0] - a[j - 1, 0]))):
            lrange = np.fabs(a[j - 1, 0] - a[j - 2, 0])

            # partial range
            if j == 2:
                mean = (a[0, 0] + a[1, 0]) / 2.
                Pstart = a[0, 0]
                Pend = a[1, 0]
                tstart = a[0, 1]
                tend = a[1, 1]
                a[0] = a[1]
                a[1] = a[2]
                j = 1
                if (lrange > 0):
                    array_out[po, 0] = Pstart
                    array_out[po, 1] = Pend
                    array_out[po, 2] = tstart
                    array_out[po, 3] = tend
                    array_out[po, 4] = Pend - Pstart
                    array_out[po, 5] = tend - tstart
                    array_out[po, 6] = mean
                    array_out[po, 7] = uc_mult
                    po += 1
            # full range
            else:
                mean = (a[j - 1, 0] + a[j - 2, 0]) / 2.
                Pstart = a[j - 2, 0]
                Pend = a[j - 1, 0]
                tstart = a[j - 2, 1]
                tend = a[j - 1, 1]
                a[j - 2] = a[j]
                j = j - 2
                if (lrange > 0):
                    array_out[po, 0] = Pstart
                    array_out[po, 1] = Pend
                    array_out[po, 2] = tstart
                    array_out[po, 3] = tend
                    array_out[po, 4] = Pend - Pstart
                    array_out[po, 5] = tend - tstart
                    array_out[po, 6] = mean
                    array_out[po, 7] = 1.00
                    po += 1

        # partial range
    for i in range(j):
        lrange = np.fabs(a[i, 0] - a[i + 1, 0])
        mean = (a[i, 0] + a[i + 1, 0]) / 2.
        Pstart = a[i, 0]
        Pend = a[i + 1, 0]
        tstart = a[i, 1]
        tend = a[i + 1, 1]

        if (lrange > 0):
            array_out[po, 0] = Pstart
            array_out[po, 1] = Pend
            array_out[po, 2] = tstart
            array_out[po, 3] = tend
            array_out[po, 4] = Pend - Pstart
            array_out[po, 5] = tend - tstart
            array_out[po, 6] = mean
            array_out[po, 7] = uc_mult
            po += 1

            # get rid of unused entries
    mask = np.ones(len(array_out), dtype=bool)

    for i in range(len(array_out)):
        if array_out[i, 7] == 0.0:
            mask[i] = False
    array_out = array_out[mask]

    angles = [0]

    for i in range(len(array_out)):
        n = array_out[i, 4] / array_out[i, 5]
        new = np.arctan(n)
        angles = np.vstack((angles, new))
    angles = angles[1:]
    angles = np.degrees(angles)
    array_out = np.c_[array_out, angles]

    """   angles=[]

    for i in range(len(array_out)):
        new=array_out[i,4]*10/array_out[i,5]
        angles.append(new)
        #else:
          #  new=math.atan2(lg[i,0],lg[i,1])
          #  angles.append(new)
    angles=np.arctan(angles)
    angles=np.degrees(angles)
    array_out=np.c_[array_out,angles]"""

    def pers(binn, array):
        freq = np.zeros((len(binn), 1))
        persi = np.zeros((len(array), 1))
        for k in range(len(binn) - 1):
            for i in range(len(array)):
                if array[i] == binn[k] or binn[k] < array[i] < binn[k + 1]:
                    freq[k] += 1  # how many times this(between the range of particular bin) occured

        for k in range(len(binn) - 1):
            for i in range(len(array)):
                if array[i] == binn[k] or binn[k] < array[i] < binn[k + 1]:
                    persi[i] = freq[k]
        return persi

    bins = int(1 // threshold)
    amp_bins = np.linspace(-1, 1, bins)
    a = array_out[:, 4]
    amp_pers = pers(amp_bins, a)

    angle_bins = np.linspace(-90, 90, bins)
    angle_pers = pers(angle_bins, angles)

    max_time = max(array_out[:, 5])
    time_bins = np.linspace(1, max_time, bins)
    b = array_out[:, 5]
    time_pers = pers(time_bins, b)

    unique_amp = np.unique(array_out[:, 4:], axis=0)

    # count of cycles with amplitude
    cycles = np.zeros((len(unique_amp), 1))
    for k in range(len(unique_amp)):
        for i in range(len(array_out)):
            if array_out[i, 4] == unique_amp[k, 0]:
                cycles[k] += array_out[i, 7]

    cyclecount = np.zeros([len(array_out), 1])
    for k in range(len(unique_amp)):
        for i in range(len(array_out)):
            if array_out[i, 4] == unique_amp[k, 0]:
                cyclecount[i] = cycles[k]

    # persistance=np.hstack((amp_pers,time_pers,angle_pers,cyclecount))
    array_out = np.append(array_out, amp_pers, axis=1)
    array_out = np.append(array_out, time_pers, axis=1)
    array_out = np.append(array_out, angle_pers, axis=1)
    array_out = np.append(array_out, cyclecount, axis=1)
    Array_out = pd.DataFrame(array_out,
                             columns=['w1', 'w2', 't1', 't2', '∆w_r', '∆t_r', 'σ_r', 'cycle', 'θ_r', 'amp_freq',
                                      'time_freq', 'angle_freq', 'cyclecount'])
    return Array_out


def lam(events, threshold):
    """
    finds the frequency of occurence of the attributes in events, in the bins they are in.
    """
    number_of_bins = int(1 / threshold)

    # stationary events
    if len(events.columns) < 5:
        lambdas = pd.DataFrame(columns=['λ(∆t_s)', 'λ(σ_s)', 'λ(∆t_s, σ_s)'])

        deltat_bins = np.linspace(1, events[['∆t_s']].max(), number_of_bins)
        sigma_bins = np.linspace(-1, 1, number_of_bins)

        lambdas['λ(∆t_s)'] = frequencies(events[['∆t_s']], deltat_bins)
        lambdas['λ(σ_s)'] = frequencies(events[['σ_s']], sigma_bins)

        return lambdas

    # major events
    else:
        lambdas = pd.DataFrame(columns=['λ(∆t_m)', 'λ(∆w_m)', 'λ(θ_m)', 'λ(σ_m)', 'λ(σ_m, θ_m)'])

        deltat_bins = np.linspace(1, events[['∆t_m']].max(), number_of_bins)
        deltaw_bins = np.linspace(-1, 1, number_of_bins)
        theta_bins = np.linspace(-90, 90, number_of_bins)
        sigma_bins = np.linspace(-1, 1, number_of_bins)

        lambdas['λ(∆t_m)'] = frequencies(events[['∆t_m']], deltat_bins)
        lambdas['λ(∆w_m)'] = frequencies(events[['∆w_m']], deltaw_bins)
        lambdas['λ(θ_m)'] = frequencies(events[['θ_m']], theta_bins)
        lambdas['λ(σ_m)'] = frequencies(events[['σ_m']], sigma_bins)

        return lambdas

def frequencies(data, bins):
    f = len(bins) * [0]
    frequency = len(data) * [0]

    for k in range(len(bins) - 1):
        for i in range(len(data)):
            if (bins[k] <= data.iloc[i]).bool() & (data.iloc[i] < bins[k + 1]).bool():
                f[k] += 1  # how many times this attribute occurred in this bin

    for k in range(len(bins) - 1):
        for i in range(len(data)):
            if (bins[k] <= data.iloc[i]).bool() & (data.iloc[i] < bins[k + 1]).bool():
                frequency[i] = f[k]
    return frequency

def save_xls(dict_df, path):
    """
    Save a dictionary of dataframes to an excel file, with each dataframe as a seperate page
    """
    writer = pd.ExcelWriter(path)
    for key in dict_df:
        dict_df[key].to_excel(writer, key)
    writer.save()


def pre_markov(major_events, stationary_events):
    """
    takes in the output of RBA_theta
    gives out the matrices that will go into markov()
    to coincide the events timewise for the turbines,since we have multpile turbines in different locations with varying events happening simultaneously or not,
    it repeats the value of the parameter over the period of time
    that particular event occurs
    e.g say for the parameter '∆w_s', there are events between t=2
    and t=5 with ∆w_s=0.3 and between t=7 and t=8 with ∆w_s=0.15,
    the series of data for ∆w_s goes:
    0,0,0.3,0.3,0.3,0.3,0,0.15,0.15,...
    This way all series of parameters for turbines coincide.
    """

    '''n = len(major_events)  # number of turbines
    columns = [f'Turbine_{i}' for i in range(1, n + 1)]

    # initializing the attribute dfs with 0 when there is no event
    deltat_m = pd.DataFrame(np.zeros((tao, n)), columns=columns)
    deltaw_m = pd.DataFrame(np.zeros((tao, n)), columns=columns)
    theta_m = pd.DataFrame(np.zeros((tao, n)), columns=columns)
    sigma_m = pd.DataFrame(np.zeros((tao, n)), columns=columns)

    for i in range(1, n + 1):
        k = len(major_events[f'Turbine_{i}'])
        for m in range(k):
            if math.isnan(major_events[f'Turbine_{i}'].loc[m, 't1']):
                break
            else:
                a = int(major_events[f'Turbine_{i}'].loc[m, 't1'])
                b = int(major_events[f'Turbine_{i}'].loc[m, 't2'])
                deltat_m.iloc[a:b, int(i - 1)] = major_events[f'Turbine_{i}'].loc[m, '∆t_m']
                deltaw_m.iloc[a:b, int(i - 1)] = major_events[f'Turbine_{i}'].loc[m, '∆w_m']
                theta_m.iloc[a:b, int(i - 1)] = major_events[f'Turbine_{i}'].loc[m, 'θ_m']
                sigma_m.iloc[a:b, int(i - 1)] = major_events[f'Turbine_{i}'].loc[m, 'σ_m']

    deltat_s = pd.DataFrame(np.zeros((tao, n)), columns=columns)
    sigma_s = pd.DataFrame(np.zeros((tao, n)), columns=columns)

    for i in range(1, n + 1):
        k = len(stationary_events[f'Turbine_{i}'])
        for m in range(k):
            if math.isnan(stationary_events[f'Turbine_{i}'].loc[m, 't1']):
                break
            else:
                a = int(stationary_events[f'Turbine_{i}'].loc[m, 't1'])
                b = int(stationary_events[f'Turbine_{i}'].loc[m, 't2'])
                deltat_s.iloc[a:b, int(i - 1)] = stationary_events[f'Turbine_{i}'].loc[m, '∆t_s']
                sigma_s.iloc[a:b, int(i - 1)] = stationary_events[f'Turbine_{i}'].loc[m, 'σ_s']

    major_attr, stationary_attr = {}, {}

    major_attr['∆t_m'] = np.array([mc.Quantiles(y, k=5).yb for y in deltat_m.values]).transpose()
    major_attr['∆w_m'] = np.array([mc.Quantiles(y, k=5).yb for y in deltaw_m.values]).transpose()
    major_attr['θ_m'] = np.array([mc.Quantiles(y, k=5).yb for y in theta_m.values]).transpose()
    major_attr['σ_m'] = np.array([mc.Quantiles(y, k=5).yb for y in sigma_m.values]).transpose()

    stationary_attr['∆t_s'] = np.array([mc.Quantiles(y, k=5).yb for y in deltat_s.values]).transpose()
    stationary_attr['σ_s'] = np.array([mc.Quantiles(y, k=5).yb for y in sigma_s.values]).transpose()'''

    n = len(major_events)  # number of turbines
    columns = [f'Turbine_{i}' for i in range(1, n + 1)]

    # initializing the attribute dfs with 0 when there is no event
    deltat_m = pd.DataFrame(columns=columns)
    deltaw_m = pd.DataFrame(columns=columns)
    theta_m = pd.DataFrame(columns=columns)
    sigma_m = pd.DataFrame(columns=columns)

    for i in range(1, n + 1):
        deltat_m[f'Turbine_{i}'] = major_events[f'Turbine_{i}']['∆t_m']
        deltaw_m[f'Turbine_{i}'] = major_events[f'Turbine_{i}']['∆w_m'] * 100
        theta_m[f'Turbine_{i}'] = major_events[f'Turbine_{i}']['θ_m']
        sigma_m[f'Turbine_{i}'] = major_events[f'Turbine_{i}']['σ_m'] * 100

    deltat_m = deltat_m.dropna()
    deltaw_m = deltaw_m.dropna()
    theta_m = theta_m.dropna()
    sigma_m = sigma_m.dropna()

    deltat_s = pd.DataFrame(columns=columns)
    sigma_s = pd.DataFrame(columns=columns)

    for i in range(1, n + 1):
        deltat_s[f'Turbine_{i}'] = stationary_events[f'Turbine_{i}']['∆t_s']
        sigma_s[f'Turbine_{i}'] = stationary_events[f'Turbine_{i}']['σ_s'] * 100

    deltat_s = deltat_s.dropna()
    sigma_s = sigma_s.dropna()
    major_attr, stationary_attr = {}, {}


    #for lisa markov
    major_attr['∆t_m'] = np.array([mc.Quantiles(y, k=8).yb for y in deltat_m.values]).transpose()
    major_attr['∆w_m'] = np.array([mc.Quantiles(y, k=8).yb for y in deltaw_m.values]).transpose()
    major_attr['θ_m'] = np.array([mc.Quantiles(y, k=8).yb for y in theta_m.values]).transpose()
    major_attr['σ_m'] = np.array([mc.Quantiles(y, k=8).yb for y in sigma_m.values]).transpose()

    stationary_attr['∆t_s'] = np.array([mc.Quantiles(y, k=8).yb for y in deltat_s.values]).transpose()
    stationary_attr['σ_s'] = np.array([mc.Quantiles(y, k=8).yb for y in sigma_s.values]).transpose()

    '''#for spatial markov
    major_attr['∆t_m'] = np.array(deltat_m.values).transpose()
    major_attr['∆w_m'] = np.array(deltaw_m.values).transpose()
    major_attr['θ_m'] = np.array(theta_m.values).transpose()
    major_attr['σ_m'] = np.array(sigma_m.values).transpose()

    stationary_attr['∆t_s'] = np.array(deltat_s.values).transpose()
    stationary_attr['σ_s'] = np.array(sigma_s.values).transpose()'''

    return major_attr, stationary_attr
