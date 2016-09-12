#! /usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

a = 2.5
b = 0.2

def fun(x):
    global a, b
    return a - (x - b) * (x - b)

x1 = x2 = 1.7
y1 = fun(x1)
y2 = 2.
c = np.power((x1 - b) / 2., 1/3.)
x3 = b + c
y3 = fun(x3)

def dfun(x):
    global b, c
    return -2. * c * (x - b - c) + fun(b + c)

x4 = x3 - .4
y4 = dfun(x4)
x5 = x3 + .4
y5 = dfun(x5)

fig = plt.figure(figsize=(5.5,6.5), frameon=False)
x = np.linspace(-.25, 2.25, 101)
y = fun(x)
plt.plot(x,y)

plt.plot([x1, x2, x3], [y1, y2, y3], color='k', marker='o', markersize=5)
plt.plot([x4, x5], [y4, y5], color='k', marker='None')
plt.text((x1 + x2) / 2. + .05, (y1 + y2 ) / 2., '$d_1$', fontsize=18)
plt.text((x2 + x3) / 2. - .08, (y2 + y3 ) / 2. + .08, '$d_2$', fontsize=18)

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
ax.xaxis.set_major_locator(plt.NullLocator())
ax.yaxis.set_major_locator(plt.NullLocator())
plt.axis([-.25, 2.5, -.25, 3.], 'scaled')
plt.title('Fig.2', fontsize=20, fontweight='bold')

#plt.show()
plt.savefig('fig2.png', dpi=50)

