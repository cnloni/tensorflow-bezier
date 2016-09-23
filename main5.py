#! /usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import fitter4 as ft


r0 = np.loadtxt('data/m-cap.2.dat', delimiter=',')
plt.plot(r0[:,0], r0[:,1], color='r', marker='o', markersize=10, linestyle='None')

gr1 = ft.GraphPhase1()
data = np.load('data/m-cap.2.main3.npz')
t = np.linspace(0., 1., 101)
r = gr1.get_points(data['bs'], t)
plt.plot(r[:,0], r[:,1], color='blue')

plt.axis([-300, 200, -200, 200], 'equal', 'scaled')
plt.xticks([-200, -100, 0, 100, 200])
plt.yticks([-200, -100, 0, 100, 200])
plt.show()
#plt.savefig('images/fig4.png', dpi=50)

