
import numpy as np
from pandas import Series,DataFrame
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

"""
-------------------------------------------------------------------------------
USAGE:
To call the function in a script on array of turning points <array_ext>:  
    import rainflow as rf  
    array_out = rf.rainflow(array_ext)
To run the demonstration from a Python console:  
    >>> execfile('demo_rainflow.py')
From the terminal (Windows or UNIX-based):  
    $ python demo_rainflow.py
-------------------------------------------------------------------------------
DEPENDENCIES:
- Numpy >= v1.3
-------------------------------------------------------------------------------
NOTES:
Python code modified from rainflow.c code with mex function for Matlab from 
WISDEM project: https://github.com/WISDEM/AeroelasticSE/tree/master/src/AeroelasticSE/rainflow

"""
from numpy import fabs as fabs
from pandas import Series,DataFrame
import pandas as pd
import numpy as np

def rainflow(array,bins=100,
             flm=0, l_ult=1e16, uc_mult=0.5):
    """ Rainflow counting of a signal's turning points with Goodman correction

        Args:
            array_ext (numpy.ndarray): array of turning points

        Keyword Args:
            flm (float): fixed-load mean [opt, default=0]
            l_ult (float): ultimate load [opt, default=1e16]
            uc_mult (float): partial-load scaling [opt, default=0.5]

        Returns:
            array_out (numpy.ndarray): (5 x n_cycle) array of rainflow values:
                                        1) load range
                                        2) range mean
                                        3) Goodman-adjusted range
                                        4) cycle count
                                        5) Goodman-adjusted range with flm = 0

    """
    
    #extracting peak and through values first 
    
    diff=np.diff(array)           # differences between consequtive values                 
    start=0
    maximum=[array[start],start]
    
    for i in range(len(diff)-1):
        if np.sign(diff[i])== np.sign(diff[i+1]):
            continue
        else:
            new_event=np.array([array[i+1],i+1])
            maximum=np.row_stack((maximum,new_event)) #new row
            start=i+1
    maximum=np.row_stack((maximum,[array[len(array)-1],len(array)-1]))
    
    
    #rainflow with Power range and time range of cycles
    
    l = len(maximum)                      # total size of input array
    array_out = np.zeros((l-1,8))     # initialize output array
    pr = 0                                  # index of input array
    po = 0                                  # index of output array
    j = -1                                  # index of temporary array "a"
    a  = np.empty((len(maximum),2))          # temporary array for algorithm
    # loop through each turning point stored in input array
    for i in range(l):
        
        j += 1                  # increment "a" counter
        a[j] = maximum[pr]    # put turning point into temporary array
        pr += 1                 # increment input array pointer

        
        while ((j >= 2) & (fabs( a[j-1,0] - a[j-2,0] ) <= fabs( a[j,0] - a[j-1,0]) ) ):
            lrange = fabs( a[j-1,0] - a[j-2,0] )
            
            # partial range
            if j == 2:
                mean      = ( a[0,0] + a[1,0] ) / 2.
                Pstart=a[0,0]
                Pend=a[1,0]
                tstart=a[0,1]
                tend=a[1,1]
                a[0]=a[1]
                a[1]=a[2]
                j=1
                if (lrange > 0):
                    array_out[po,0] = Pstart
                    array_out[po,1] = Pend
                    array_out[po,2] = tstart
                    array_out[po,3] = tend
                    array_out[po,4] = Pend-Pstart
                    array_out[po,5] = tend-tstart
                    array_out[po,6] = mean
                    array_out[po,7] = uc_mult
                    po += 1
            # full range
            else:
                mean      = ( a[j-1,0] + a[j-2,0] ) / 2.
                Pstart=a[j-2,0]
                Pend=a[j-1,0]
                tstart=a[j-2,1]
                tend=a[j-1,1]
                a[j-2]=a[j]
                j=j-2
                if (lrange > 0):
                    array_out[po,0] = Pstart
                    array_out[po,1] = Pend
                    array_out[po,2] = tstart
                    array_out[po,3] = tend
                    array_out[po,4] = Pend-Pstart
                    array_out[po,5] = tend-tstart
                    array_out[po,6] = mean
                    array_out[po,7] = 1.00
                    po += 1

    # partial range
    for i in range(j):
        lrange    = fabs( a[i,0] - a[i+1,0] );
        mean      = ( a[i,0] + a[i+1,0] ) / 2.
        Pstart=a[i,0]
        Pend=a[i+1,0]
        tstart=a[i,1]
        tend=a[i+1,1]
        if (lrange > 0):
            array_out[po,0] = Pstart
            array_out[po,1] = Pend
            array_out[po,2] = tstart
            array_out[po,3] = tend
            array_out[po,4] = Pend-Pstart
            array_out[po,5] = tend-tstart
            array_out[po,6] = mean
            array_out[po,7] = uc_mult
            po += 1  
    
    # get rid of unused entries
    mask=np.ones(len(array_out), dtype=bool)
    
    for i in range(len(array_out)):
        if array_out[i,7] == 0.0:
            mask[i]=False
    array_out=array_out[mask]

    
    
    #persistance 
    angles=[0]
    
    for i in range(len(array_out)):
        n=array_out[i,4]/array_out[i,5]
        new=np.arctan(n)
        angles=np.vstack((angles,new))
    angles=angles[1:]
    angles=np.degrees(angles)
    array_out=np.c_[array_out,angles]

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
    def pers(binn,array):
        freq=np.zeros((len(binn),1))
        persi=np.zeros((len(array),1))
        for k in range(len(binn)-1): 
            for i in range(len(array)):
                if array[i]==binn[k]or binn[k] < array[i]< binn[k+1]:
                    freq[k]+=1 #how many times this(between the range of particular bin) occured
    
        for k in range(len(binn)-1): 
            for i in range(len(array)):
                if array[i]==binn[k]or binn[k] < array[i]< binn[k+1]:
                    persi[i]=freq[k]
        return persi
    
    amp_bins=np.linspace(-1,1,bins)
    a=array_out[:,4]
    amp_pers=pers(amp_bins,a)
    
    angle_bins=np.linspace(-90,90,bins)
    angle_pers=pers(angle_bins,angles)
    
    max_time=max(array_out[:,5])
    time_bins=np.linspace(1,max_time,bins)
    b=array_out[:,5]
    time_pers=pers(time_bins,b)
    
    
    #count of cycles with amplitude
    cycles=np.zeros((len(unique_amp),1))
    for k in range(len(unique_amp)):
        for i in range(len(array_out)):
            if array_out[i,4]==unique_amp[k,0]:
                cycles[k]+=array_out[i,7]
    
    cyclecount=np.zeros([len(array_out),1])
    for k in range(len(unique_amp)):
        for i in range(len(array_out)):
            if array_out[i,4]==unique_amp[k,0]:
                cyclecount[i]=cycles[k]
    
    #persistance=np.hstack((amp_pers,time_pers,angle_pers,cyclecount))
    array_out=np.append(array_out,amp_pers,axis=1) 
    array_out=np.append(array_out,time_pers,axis=1) 
    array_out=np.append(array_out,angle_pers,axis=1) 
    array_out=np.append(array_out,cyclecount,axis=1) 
    Array_out=DataFrame(array_out,columns=['P_start','P_end','t_start','t_end','P_diff','t_diff','mean','cycle','angle','amp_pers','time_pers','angle_pers','cyclecount'])
    writer = pd.ExcelWriter('C:\\Users\\\esino\\Dropbox\\Master Thesis Work - Esin Ören\\Data\\rainflow.xlsx')
    norm.to_excel(writer,sheet_name='Sheet1',columns=col)
    writer.save()
    return Array_out

# rainflow()