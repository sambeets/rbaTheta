
"""
Markovian chain for the geospatial location probabilities
"""

import time
import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import core.model as model
from core.helpers import save_xls, pre_markov, normalize


shp_file = r'docs/QGISfiles/wind_park_thiessen.shp'
path1 = r'test_results/all_events/significant_events_T_0.1.xlsx'
path2 = r'test_results/all_events/stationary_events_T_0.1.xlsx'
excel1 = pd.ExcelFile(path1)
excel2 = pd.ExcelFile(path2)
major, stationary = {}, {}
for i in range(1, 9):
    df1 = pd.read_excel(excel1, f'Turbine_{i}')
    df2 = pd.read_excel(excel2, f'Turbine_{i}')
    major[f'Turbine_{i}'] = df1
    stationary[f'Turbine_{i}'] = df2

major_attr, stationary_attr = pre_markov(major, stationary)
matrices = model.markov(major_attr, stationary_attr, shp_file)
'''print(matrices['∆t_m_tran'].p)
print(matrices['∆w_m_tran'].p)
print(matrices['θ_m_tran'].p)
print(matrices['σ_m_tran'].p)
print(matrices['∆t_s_tran'].p)
print(matrices['σ_s_tran'].p)
'''

#spatial markov correlation plotted_figures
attributes = ['∆t_m_tran', '∆w_m_tran', 'θ_m_tran', 'σ_m_tran', '∆t_s_tran', 'σ_s_tran']
titles = [r'$\Delta t_m$', '$\Delta w_m$', '$\sigma_m$', '$\Theta_m$', '$\Delta t_s$', '$\sigma_s$',]
fig, axes = plt.subplots(2,3,figsize = (15,10))
for i in range(2):
    for j in range(3):
        ax = axes[i, j]
        # Loop over data dimensions and create text annotations.
        p_temp = matrices[attributes[i*3 + j]].p
        print(f'{titles[i*3 + j]} : {matrices[attributes[i*3 + j]].chi_2}')
        for x in range(len(p_temp)):
            for y in range(len(p_temp)):
                text = ax.text(y, x, round(p_temp[x, y], 2),
                               ha="center", va="center", color="w")
        im = ax.imshow(p_temp, cmap="coolwarm", vmin=0, vmax=1)
        fig.tight_layout(pad=3.0)
        ax.set_title(f'{titles[i*3 + j]}', fontsize=10)
plt.subplots_adjust(wspace=0.08, hspace=0.4)
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.95, 0.228, 0.01, 0.5])
fig.colorbar(im, cax=cbar_ax)
plt.show()
#fig.savefig(r'plots/plotted_figures/lisa_markov_turbines.pdf', dpi = 300)
