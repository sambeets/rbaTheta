
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
plt.style.use("seaborn-paper")
import seaborn as sns
sns.set_palette("RdBu")
sns.set(context='talk', style='ticks')
from mpl_toolkits.axes_grid.inset_locator import inset_axes

font = {'family': 'Trebuchet MS',
        'color':  'black', #'darked'
        'weight': 'normal',
        'size': 15
        }
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

# from pylab import figure, scatter, show
data = pd.read_excel(r'H:\Dropbox\RBA theta\RBA_theta\simulations\data\8_wind_turbine_data.xlsx', sheet_name=0).values

#%%
m=data[:672,0]
y=np.arange(len(m))

#     PLOT
fig = plt.figure(figsize=(12,7))
plt.scatter(y[:672], m[:672], color='black', s=50, marker = "x", alpha = 1.0, label='Wind power [per unit]')
'''
The issue is Events!
'''
#x=Events(m,4)
x = data(m,4)
for i in range(len(x)):
    a=int(x.iloc[int(i),2])
    b=int(x.iloc[int(i),3])+1
    if m[a]>m[b]:
        plt.plot(y[a:b], m[a:b], color='lightskyblue', linewidth=15, alpha =0.4, antialiased=True)
    else:
        plt.plot(y[a:b], m[a:b],  color='plum', linewidth=15 , alpha =0.4, antialiased=True)


plt.gca().set_axisbelow(True)
plt.gca().yaxis.grid(True, linestyle='--')
plt.legend( ['Down-ramp events','Up-ramp events'], prop={'size': 15},
           bbox_to_anchor=(0.5, 1.1), ncol=2, loc="upper center", frameon=False)
plt.xlabel('Time [hours]', fontdict=font)
plt.ylabel('Wind power generation [per unit]', fontdict=font)
plt.text(-1.0, 0.06, "$x$   Wind power [per unit] ", { 'fontsize': 15}) #, 'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})
fig.savefig("./plotted_figures/RBAevents.pdf", bbox_inches='tight')

plt.gca().set_axisbelow(True)
plt.gca().yaxis.grid(True, linestyle='--')

plt.text(-1.0, 0.06, "$\mathsf{x}$   Wind power [per unit] ", { 'fontsize': 15}) #, 'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})
fig.savefig("./plotted_figures/RBAevents.pdf", bbox_inches='tight')