#! /usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import sys
sys.path.append('..')
import fitter4 as ft


fp = FontProperties(fname=r'/usr/share/fonts/truetype/fonts-japanese-gothic.ttf', size=14)

fig = plt.figure(figsize=(7., 7.), frameon=False)

r0 = np.loadtxt('../cdata/m-cap.2.dat', delimiter=',')
plt.plot(r0[:,0], r0[:,1], color='r', marker='o', markersize=10, linestyle='None')

gr1 = ft.GraphPhase1()

data2 = np.load('../cdata/m-cap.2.main1.npz')
bs = data2['bs']
t2 = data2['t']

t = np.linspace(t2[0], t2[-1], 101)
r = gr1.get_points(bs, t)
plt.plot(r[:,0], r[:,1], color='red')

data1 = np.load('../cdata/m-cap.2.main4.npz')
bss = data1['bss']
t = np.linspace(0., 1., 101)

r = gr1.get_points(bss[-1], t)
plt.plot(r[:,0], r[:,1], color='blue')

plt.axis([-300, 200, -250, 250], 'equal', 'scaled')
plt.xticks([-200, -100, 0, 100, 200])
plt.yticks([-200, -100, 0, 100, 200])
plt.legend(['学習データ', '実装１ 400万step', '実装２ 8万step', ], prop=fp, loc='lower left')
plt.title('Fig.5', fontsize=20, fontweight='bold')
#plt.show()
plt.savefig('fig5.png', dpi=50)

