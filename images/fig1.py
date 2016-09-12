#! /usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(7.5, 6.), frameon=False)

r0 = np.loadtxt('../cdata/m-cap.2.dat', delimiter=',')
plt.plot(r0[:,0], r0[:,1], color='r', marker='o', markersize=10, linestyle='None')

plt.axis([-300,200,-200,200], 'equal', 'scaled')
plt.xticks([-200, -100, 0, 100, 200])
plt.yticks([-200, -100, 0, 100, 200])
plt.title('Fig.1', fontsize=20, fontweight='bold')

#plt.show()
plt.savefig('fig1.png', dpi=40)

