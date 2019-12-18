# load first 2 lines and see if it contains 2 words


import numpy as np
import matplotlib.pyplot as plt

# Create some mock data
t = np.arange(0.01, 10.0, 0.01)
data1 = np.exp(t)
data2 = np.sin(2 * np.pi * t)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('exp', color=color)
ax1.plot(t, data1, color=color)
ax1.tick_params(axis='y', labelcolor=color)

start, end = ax1.get_ylim()
ax1.yaxis.set_ticks(np.arange(start, end, (end-start)/10))

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
ax2.plot(t, data2, color=color)
ax2.tick_params(axis='y', labelcolor=color)


start, end = ax2.get_ylim()
ax2.yaxis.set_ticks(np.arange(start, end, (end-start)/10))

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()