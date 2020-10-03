import matplotlib.pyplot as plt
import pandas as pd

path = r'H:\Dropbox\RBA theta\RBA_theta\simulations\plot\RBA_events.xlsx'
plot_data = pd.read_excel(path, skiprows=1, sheet_name='significant_events',header=None,
                          names=["index", "w_s(t)", "w_s(t+∆t)", "t",
                                 "t+∆t", "∆w_s", "∆t", "θ(∆w_s)",
                                 "mean(∆w_s)",	"f(∆w_s)", "f(∆t)", "f(θ(∆w_s))",
                                 "mean(∆w_s)"])
#plot_data.iloc[0, 0] = 0

#%% All input_data plot

plt.figure(1)
x = plot_data['index']

plt.plot(x, plot_data)
plt.savefig('test.pdf', bbox_inches = 'tight')

#%%

plt.figure(2)
from pandas.plotting import radviz
radviz(plot_data, 'index');
plt.savefig('test2.pdf', bbox_inches='tight')

#%%

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
import seaborn as sns

new_plot_data = plot_data.drop(['index'], axis =1)

corr = new_plot_data.corr()
f, ax = plt.subplots(figsize=(10, 6))
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Attributes Correlation Heatmap', fontsize=14)
plt.savefig('Correlation heatmap.png', bbox_inches = 'tight', dpi=300)

#%%

pp = sns.pairplot(new_plot_data, height=1.8, aspect=1.8,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))

fig = pp.fig
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('Attributes Pairwise Plots', fontsize=14)
plt.savefig('Attributes Pairwise Plots.png', bbox_inches = 'tight', dpi=300)
