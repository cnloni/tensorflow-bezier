#! /usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

fp = FontProperties(fname=r'/usr/share/fonts/truetype/fonts-japanese-gothic.ttf', size=18)

file1 = '../cdata/main3.res'
file2 = '../cdata/main1.1e-5.res'
file3 = '../cdata/main1.1e-4.res'

fig = plt.figure(figsize=(8., 6.), frameon=False)
ax = fig.add_subplot(111)
ax.axis([0, 600000, 0., 4.], 'scaled')
ax.set_title('Fig.4', fontsize=20, fontweight='bold')
ax.set_xlabel('steps', fontsize=18)
ax.set_ylabel('diff   (log)', fontsize=18)

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

d3 = np.genfromtxt(file3, delimiter=' ',
    dtype=[('nstep',int),('diff',float)])
d3x = d3['nstep']
d3y = d3['diff']

d1ylog = np.log10(d1y)
d2ylog = np.log10(d2y)
d3ylog = np.log10(d3y)

ax.plot(d2x, d2ylog, color='r', marker='None')
ax.plot(d3x, d3ylog, color='g', marker='None')
ax.plot(d1x, d1ylog, color='b', marker='None')
ax.legend(['実装１ 1e-5', '実装１ 1e-4', '実装２'], prop=fp, loc='lower right')

#plt.show()
plt.savefig('fig4.png', dpi=50)

