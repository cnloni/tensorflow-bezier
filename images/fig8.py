#! /usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
import fitter4


fig = plt.figure(figsize=(6., 4.5), frameon=False)
ax = fig.add_subplot(111)
ax.axis([-300, 200, -220, 230], 'equal', 'scaled')
ax.set_xticks([-300, -200, -100, 0, 100, 200])
ax.set_yticks([-200, -100, 0, 100, 200])
ax.set_title('Fig.8', fontsize=20, fontweight='bold')

gr1 = fitter4.GraphPhase1()
data = np.load('../data/m-cap.2.main4.npz')
bss = data['bss']
t = np.linspace(0., 1., 101)

ap = dict(arrowstyle="->", connectionstyle='arc3',
    shrinkB=1, color=(.6, .6, .6))

it0 = 37
for i in range(4):
    id = np.power(2,i)
    r = gr1.get_points(bss[id], t)
    ax.plot(r[:,0], r[:,1], color=(.6, .6, .6))
    it = it0 + i*3;
    ax.annotate(id, xy=(r[it][0], r[it][1]),
        xytext=(-60,-10 - i*10), textcoords='offset points',
        fontsize=14, ha='center', va='center',
        arrowprops=ap)

r0 = np.loadtxt('../data/m-cap.2.dat', delimiter=',')
ax.plot(r0[:,0], r0[:,1], color='r', marker='o', markersize=10, linestyle='None')

r = gr1.get_points(bss[-1], t)
ax.plot(r[:,0], r[:,1], color='blue')
it = 34
ax.annotate(len(bss) - 1, xy=(r[it][0], r[it][1]),
    xytext=(70,40), textcoords='offset points',
    fontsize=14, ha='center', va='center',
    arrowprops=ap)

#plt.show()
plt.savefig('fig8.png', dpi=80)

