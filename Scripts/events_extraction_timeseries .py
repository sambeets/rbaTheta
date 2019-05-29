# In[]
#   Module imports

import numpy as np
from numpy import sign
import pandas as pd
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_context("talk", font_scale=0.7)
import scipy
import math
from scipy import signal



# In[]

#   to normalize the input between [0,1]
def normalize(wind_data,nominal):
    
    norm=wind_data/nominal
    return norm


# In[]:

#   Blackman filter
#   takes in a timeseries and cut off frequency. gives it out low pass filtered 
def filterb(splined,fc):
    
    b = 0.08
    N = int(np.ceil((4 / b)))
    if not N % 2: N += 1
    n = np.arange(N)
 
    sinc_func = np.sinc(2 * fc * (n - (N - 1) / 2.))
    window = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
    sinc_func = sinc_func * window
    sinc_func = sinc_func / np.sum(sinc_func)

    filtered = np.convolve(splined, sinc_func)
    filtered=filtered[25:-25]

    return filtered


# In[]:

#   finds the frquency of occurence of the values in array, cathegorized as in bins
    
def freq(binn,array):
    
    freq=np.zeros((len(binn),1))
    f=np.zeros((len(array),1))
    for k in range(len(binn)-1): 
        for i in range(len(array)):
            if array[i]==binn[k]or binn[k] < array[i]< binn[k+1]:
                freq[k]+=1 #how many times this(between the range of particular bin) occured
    
    for k in range(len(binn)-1): 
        for i in range(len(array)):
            if array[i]==binn[k]or binn[k] < array[i]< binn[k+1]:
                f[i]=freq[k]
    return f


# In[]:


#   finds the significant(higher than amplitude) events calling out to the above functions

def signif_events(filtered,threshold,bins):
    #threshold*= if the change in amplitude is bigger than or equal to the threshold, there is a significant event
    #bins*= number of bins that the range of parameters (∆a(-1,1), ∆t(1,max(∆t)), α(-90,90), mean(min(mean),max(mean))) will be divided into, to count the frequency of the events that fall into those bins
    
    delta=np.diff(filtered)  #the change between consequential points 
    
    #consequential events
    events=np.empty([1,4])
    lengths=np.empty([1,2])
    start=0
    for i in range(len(delta)-1):
        
        if sign(delta[i])== sign(delta[i+1]):
            continue
        else:
            new_event=np.array([filtered[start],filtered[i+1],start,i+1])
            events=np.row_stack((events,new_event)) #new row
            start=i+1
    events=events[1:]
    
    #amplitude and length(time) of those events
    for i in range(len(events)):
        lengths=np.row_stack((lengths,[events[i,1]-events[i,0],events[i,3]-events[i,2]]))
    lengths=lengths[1:]
            
    #sifted events with a threshold
    neglect = np.ones(len(lengths), dtype=bool)
    
    for i in range(len(lengths)):
        if abs(lengths[i,0])< threshold:
            neglect[i]=False
    
    masked=events[neglect]
    masked_lg=np.empty([1,2])
    for i in range(len(masked)):
        masked_lg=np.row_stack((masked_lg,[masked[i,1]-masked[i,0],masked[i,3]-masked[i,2]]))
    masked_lg=masked_lg[1:]
    new_masked=masked
    for i in range(len(masked)-1):
        if sign(masked_lg[i,0])==sign(masked_lg[i+1,0]):
            new_masked[i,1]=masked[i+1,1]
            new_masked[i,3]=masked[i+1,3]
            new_masked=np.delete(masked,(i+1),axis=0)
                       
    
    lg=np.empty([1,2])
    for i in range(len(new_masked)):
        lg=np.row_stack((lg,[new_masked[i,1]-new_masked[i,0],new_masked[i,3]-new_masked[i,2]]))
    lg=lg[1:]

    
    angles=[]
    for i in range(len(lg)):

        new=lg[i,0]*10/lg[i,1]
        angles.append(new)

    angles=np.arctan(angles)
    angles=np.degrees(angles)
    
    events=np.append(new_masked,lg,axis=1)
    events=np.c_[events,angles]

    mean=[]
    for i in range(len(events)):
        mean.append((events[i,0]+events[i,1])/2)
    events=np.c_[events,mean]
   
    #frequency
    amp_bins=np.linspace(-1,1,bins)
    a=events[:,4]
    amp_fre=freq(amp_bins,a)
    
    max_time=max(events[:,5]) # what is the purpose of max_time?
    time_bins=np.linspace(min(events[:,5]),max(events[:,5]),bins)
    b=events[:,5]
    time_fre=freq(time_bins,b)
    
    angle_bins=np.linspace(-90,90,bins)
    c=events[:,6]
    angle_fre=freq(angle_bins,c)
    
    mean_bins=np.linspace(min(mean),max(mean),bins)
    d=events[:,7]
    mean_fre=freq(mean_bins,d)

    events=np.append(events,amp_fre,axis=1) 
    events=np.append(events,time_fre,axis=1) 
    events=np.append(events,angle_fre,axis=1) 
    events=np.append(events,mean_fre,axis=1) 
    Events=pd.DataFrame(events,columns=['w_s(t)','w_s(t+∆t)','t','t+∆t','∆w_s','∆t','θ(∆w_s)','mean(∆w_s)','f(∆w_s)','f(∆t)','f(θ(∆w_s))','f(mean(∆w_s))'])
    '''
    w_s(t) = Amplitude of the event at start
    w_s(t+∆t) = Amplitude of the event at end
    t = where w_s(t) happended / starting time point of the event
    t+∆t = ending time point of the event
    ∆w_s = amplitude of the evnet
    ∆t = time range of the event
    θ(∆w_s) = angle of the event
    mean(∆w_s) = mean of the event
    f(∆w_s) = frequency of the amplitude
    f(∆t) = frequency of the time range
    f(θ(∆w_s)) = frequency of the angle
    f(mean(∆w_s)) = frequency of the mean
    '''
    return Events


# In[17]:


def stag_events(filtered,threshold):
    #threshold*= if the change in amplitude is smaller the threshold, there might be a significant event
    col=['w_s(t)','w_s(t+∆t)','t','t+∆t','∆w_s','mean','∆t-persistence']
    events=pd.DataFrame(columns=col)  
    persistent_events=pd.DataFrame(columns=col) # is it never used?

    i=0
    t=len(filtered)
    limit=threshold/2
    
    while i<t: #loop over the timeline
        k=t-i
        forward=0
        backward=0
        
        for j in range(1,k):   #loop to compare 
            
            if filtered.iloc[i] - limit < filtered.iloc[i+j] < filtered.iloc[i] + limit: 
                forward+=1
                #print("pers", pers)
            else:
                if i>0:
                    for k in range(1,i+1):
                        if filtered.iloc[i] - limit < filtered.iloc[i-k] < filtered.iloc[i] + limit: 
                            backward+=1
                
                new=pd.DataFrame([[filtered.iloc[i-backward],filtered.iloc[i+forward],i-backward,i+forward,
                                   (filtered.iloc[i+forward]-filtered.iloc[i-backward]),filtered.iloc[i],
                                   forward+backward]],columns=col)
                
                events=events.append(new,ignore_index=True)
                break
        #print(events)
        i+=1 
        
    events.drop_duplicates(subset=['t','t+∆t'])
    #new=events.sort_values(by=['∆t','∆w_s'],ascending=False)
    #print(new)
    #persistent_events = persistent_events.append(new.iloc[0,:])        
    return events #persistent_events

# plotting
#pevent = pers_events(wind_data, threshold=0.3*2500,period=1)
#print(pevent)

#plt.figure(1)
#pevent.iloc[20].plot.barh(stacked=True, colormap='cubehelix');


# In[]:


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

        
        while ((j >= 2) & (np.fabs( a[j-1,0] - a[j-2,0] ) <= np.fabs( a[j,0] - a[j-1,0]) ) ):
            lrange = np.fabs( a[j-1,0] - a[j-2,0] )
            
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
        lrange    = np.fabs( a[i,0] - a[i+1,0] );
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
    
    
    unique_amp=np.unique(array_out[:,4:],axis=0)
    
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
    Array_out=pd.DataFrame(array_out,columns=['P_start','P_end','t_start','t_end','P_diff','t_diff','mean','cycle','angle','amp_freq','time_freq','angle_freq','cyclecount'])
    return Array_out


# In[]:


def RBA_theta(wind_data_path, nominal, s=0.01,k=3, fc=0.3, threshold=0.15, bins=100):
    #nominal= nominal value in Watts
    #s=smoothness factor of Bspline
    #k=degree of Bspline
    #fc=cutoff frequency for Blackman filter-
    # add into the function later shapefile
    

    #wind_data=[amplitude through time(/10minutes), # of turbines] in Watts
    wind_data = pd.read_excel(wind_data_path, sheetname=0)
    wind_data=wind_data.iloc[:40,:]
    n=len(wind_data.columns) #n=number of turbines
    t=len(wind_data)
    
    col=[]
    for i in range(1,n+1):
        col.append('WT'+'%s' %str(i)) #column names as WT1,WT2,WT3,...
    
    #predefining 
    wind_cf=pd.DataFrame(columns=col) 
    splined=pd.DataFrame(columns=col)
    filtered=pd.DataFrame(columns=col)
    significant_events=pd.Panel() 
    persistent_events=pd.Panel()

    for i in range(n):
        
        wind_cf.iloc[:,i]=normalize(wind_data.iloc[:,i].values,nominal)  #wind_cf=capacity factors in W/W
        #splined.iloc[:,i]=bspline(wind_cf.iloc[:,i],s,k)    #splined=smoothed data with bspline in 
        filtered.iloc[:,i]=filterb(wind_cf.iloc[:,i],fc)    #filtered=blackman filtered data with fc
        
    tao=len(filtered)    
    #initializing the panels
    #significant_events={'WT1': signif_events(filtered.iloc[:,0],threshold,bins)}
    #persistent_events={'WT1': pers_events(filtered.iloc[:,0],threshold,period)}
    significant_events={'WT1': signif_events(filtered.iloc[:,0],threshold,bins)}
    stagnant_events={'WT1': stag_events(filtered.iloc[:,0],threshold)}
    rainflow_events={'WT1': rainflow(filtered.iloc[:,0],threshold)}
    
    # the event information for every turbine is in the panel as a dataframe named 'WT1,WT2,..'
    for i in range(n):
        
        #significant_events[col[i]]= signif_events(filtered.iloc[:,i],threshold,bins)
        #persistent_events[col[i]]= pers_events(filtered.iloc[:,i],threshold,period) 
        significant_events[col[i]]= signif_events(filtered.iloc[:,i],threshold,bins)
        stagnant_events[col[i]]= stag_events(filtered.iloc[:,i],threshold)
        rainflow_events[col[i]]= rainflow(filtered.iloc[:,i],threshold)
    #delta_w=significant_events.minor_xs(4)#'∆w_s'
    #delta_t=significant_events.minor_xs(5)#'∆t'
    #theta=significant_events.minor_xs(6)#'θ(∆w_s)'
    #mean=significant_events.minor_xs(7)#'mean(∆w_s)'
    
    return  [stagnant_events,significant_events,rainflow_events, tao]


# In[]:

'''
Testing

1) Threshold, beyond which we start to count the events 
threshold	Stagnant  	Significant  	Number of  	Persistent  	Rain flow  
            events 	    events 	      Peaks 	   events 	      cycles 	
0.1 					
0.2 					
0.3 					
0.4 					
0.5 					

2) cut-off frequency
fc	   Stagnant  	Significant  	Number of  	Persistent  	Rain flow  
    	events 	    events 	      Peaks 	   events 	      cycles 
0.3					
0.4					
0.5					
0.6					
0.7					

The idea here is to increase the window, does bin refer to window or the period?

bins	Stagnant  	Significant  	Number of  	Persistent  	Rain flow  
    	events 	    events 	      Peaks 	   events 	      cycles 		
100					
90					
80					
70					
60					

'''

file = r'/media/sambeet/SSD/RBA 20190511/input data/8_wind_turbine_data.xlsx'
wind_data = pd.read_excel(file, sheetname=0, header=None)
wind_data=wind_data.iloc[:50,:]
[stagnant_events,significant_events,rainflow_events, tao]=RBA_theta(file,nominal=2500, fc=0.3, threshold=0.15, bins=100)
stagnant_events['WT1']

#   Lopp through the data and store the results in both excel and sql


