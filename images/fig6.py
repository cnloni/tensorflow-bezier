#! /usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
import fitter4 as ft


fig = plt.figure(figsize=(4.2, 3.5), frameon=False)

# ax: Axes
ax = fig.add_subplot(111)
ax.axis([-5., 9., -5., 5.], 'equal', 'scaled')

ap1 = dict(arrowstyle="->", connectionstyle='arc3',
    shrinkA=4, shrinkB=36)
ap2 = dict(arrowstyle="->", connectionstyle='arc3',
    shrinkA=4, shrinkB=0)

bbox = dict(boxstyle="circle", fc='1.')
bbox2 = dict(boxstyle="round", fc='1.')

axy = (-3., 3.)
axy_desc = (-0.25, 2.25)
bxy = (-3., -3.)
bxy_desc = (-0.25, -.9)
cxy = (3., 0)
cxy_desc = (6.25, .6)
dxy = (8.5, 0.)

ax.annotate('2', xy=cxy, xycoords='data',
    xytext=axy,
    fontsize=24, ha='center', va='center',
    bbox=bbox, arrowprops=ap1)

ax.annotate('a', xy=axy_desc, xycoords='data',
    fontsize=20, ha='center', va='center')

ax.annotate('3', xy=cxy, xycoords='data',
    xytext=bxy,
    fontsize=24, ha='center', va='center',
    bbox=bbox, arrowprops=ap1)

ax.annotate('b', xy=bxy_desc, xycoords='data',
    fontsize=20, ha='center', va='center')

ax.annotate('add', xy=dxy, xycoords='data',
    xytext=cxy,
    fontsize=24, ha='center', va='center',
    bbox=bbox2, arrowprops=ap2)

ax.annotate('c', xy=cxy_desc, xycoords='data',
    fontsize=20, ha='center', va='center')

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.xaxis.set_major_locator(plt.NullLocator())
ax.yaxis.set_major_locator(plt.NullLocator())

#plt.show()
plt.savefig('fig6.png', dpi=60)

