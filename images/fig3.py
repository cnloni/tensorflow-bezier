#! /usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

fp = FontProperties(fname=r'/usr/share/fonts/truetype/fonts-japanese-gothic.ttf', size=18)

file1 = '../cdata/main3.res'
file2 = '../cdata/main1.1e-5.res'

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

plt.plot(d2x, d2ylog, color='r', marker='None')
plt.plot(d1x, d1ylog, color='b', marker='None')

plt.axis([0, 80000, 0, 4.0])
plt.xlabel('steps', fontsize=18)
plt.ylabel('diff   (log)', fontsize=18)
plt.title('Fig.3', fontsize=20, fontweight='bold')
plt.legend(['実装１', '実装２'], prop=fp, loc='center right')

#plt.show()
plt.savefig('fig3.png', dpi=60)

