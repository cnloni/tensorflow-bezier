#! /usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

fp = FontProperties(fname=r'/usr/share/fonts/truetype/fonts-japanese-gothic.ttf', size=18)

file1 = '../data/main3.res'
file2 = '../data/main1.1e-5.res'

fig = plt.figure(figsize=(8., 6.), frameon=False)
ax = fig.add_subplot(111)
ax.axis([0, 80000, 0., 4.], 'scaled')
ax.set_title('Fig.3', fontsize=20, fontweight='bold')

d1raw = np.genfromtxt(file1, delimiter=' ',
    dtype=[('phase','S2'),('nstep',int),('diff',float)])

d1x = []
d1y = []
for i in range(len(d1raw['nstep'])):
    if i == 0 or d1raw['nstep'][i] != d1raw['nstep'][i-1]:
        d1x.append(d1raw['nstep'][i])
        d1y.append(d1raw['diff'][i])

d2 = np.genfromtxt(file2, delimiter=' ',
    dtype=[('nstep',int),('diff',float)])
d2x = d2['nstep']
d2y = d2['diff']

d1ylog = np.log10(d1y)
d2ylog = np.log10(d2y)

ax.plot(d2x, d2ylog, color='r', marker='None')
ax.plot(d1x, d1ylog, color='b', marker='None')

ax.set_xlabel('steps', fontsize=18)
ax.set_ylabel('diff   (log)', fontsize=18)
ax.legend(['実装１', '実装２'], prop=fp, loc='center right')

#plt.show()
plt.savefig('fig3.png', dpi=60)

