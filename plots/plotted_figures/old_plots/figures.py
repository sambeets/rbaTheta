import numpy as np
import pandas as pd
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use("seaborn-paper")
sns.set_palette("RdBu")
sns.set(context='talk', style='ticks')


font = {'family': 'Trebuchet MS',
        'color':  'black', #'darked'
        'weight': 'normal',
        'size': 15
        }
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

#%%
# data import
input = pd.read_excel(r'all_events/8_wind_turbine_data.xlsx')
major_event = pd.read_excel(r'all_events/significant_events_T_0.15_fc_0.3.xlsx')
tt1 = pd.read_excel(r'all_events/t.xlsx') #t1 of all events

#%%
## post processing
wt1 = pd.DataFrame(input.iloc[:, 0] / input.iloc[:, 0].max())
w_m = pd.DataFrame(major_event['∆w_m'])

#%%

m = wt1
y = np.arange(len(m))

# plot original data
fig = plt.figure(figsize=(12,7))
plt.scatter(y[:670], m[:670], color='black', s=50, marker="x", alpha=1.0, label='Wind power [per unit]')


plt.gca().set_axisbelow(True)
plt.gca().yaxis.grid(True, linestyle='--')
plt.legend(['Down-ramp events', 'Up-ramp events'], prop={'size': 15},
           bbox_to_anchor=(0.5, 1.1), ncol=2, loc="upper center", frameon=False)
plt.xlabel('Time [hours]', fontdict=font)
plt.ylabel('Wind power generation [per unit]', fontdict=font)
#plt.text(-1.0, 0.06, "$x$   Wind power [per unit] ", { 'fontsize': 15}) #, 'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)})
fig.savefig("RBAevents.pdf", bbox_inches='tight')

#%%
# find the spot

t1 = major_event['t1']
_time = np.arange(8759)

df = pd.DataFrame()
df = pd.DataFrame(_time, columns=['time'])
df['wt'] = wt1
df['t1'] = t1
df = df.fillna(0)

#%%
# loop

if t1 == df.iloc[:, 0]:
    print(df.iloc[:, 1])
else:
    pass

#%%

condition = [df['time'].eq(tt1)]
choice = [df['wt']]

df['que'] = np.select(condition, choice, default=np.nan)
df['que'] = df['que'].fillna(0)

df.to_excel("output.xlsx")

#%%

df['comp'] = np.where(df["time"] == df["t1"], True, False)
