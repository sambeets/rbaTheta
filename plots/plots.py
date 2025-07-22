import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import numpy as np

#------------------------------------
#for the plot of original and the filtered data (Figure 3.b)

x = np.arange(len(splined))
y = splined
N = len(filtered)
xmin, xmax = x.min(), x.max()
xx = np.linspace(xmin, xmax, N)

from pylab import figure, scatter, show
fig=figure(figsize=(9,6))
plt.plot(x, y, 'k', label='Original points')
plt.plot(xx, filtered,'r', label='fc=0.3 Hz')

plt.xlabel('Time [/min]')
plt.ylabel('Power [W/W]')
plt.plot()
plt.grid()
plt.legend(loc='best')
fig.savefig(r'plotted_figures\filtered.pdf', bbox_inches='tight')
plt.show()

#--------------------------------------------
#for the periodogram plot (Figure 3a)

from pylab import plot, figure, scatter, show

fig=figure(figsize=(9,6))
f,Pxx_den = signal.periodogram(filtered, fs=1/60)
f,Pxx_den = signal.welch(filtered, fs=1/60,window='blackman',nperseg=len(filtered))
plt.semilogy(f, Pxx_den)
#plt.ylim([1e-7, 1e7])
plt.xlabel('Frequency [Hz]')
plt.grid()
plt.ylabel('PSD [V^2/Hz]')
fig.savefig('periodogram.pdf', bbox_inches='tight')
plt.show()
