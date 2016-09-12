#! /usr/bin/python3
import tensorflow as tf

rate = 0.4
loop = 10
x = tf.Variable(0.)
y = (x - 2.) * (x - 2.) +  1

"""
# rate = 0.06 0.05 0.04 0.03 0.02
rate = 0.02
loop = 100
x = tf.Variable(2.9)
y = 3 * x * x * x * x - 32 * x * x * x  + 114 * x * x - 144 * x + 59
"""

init = tf.initialize_all_variables()
optimize = tf.train.GradientDescentOptimizer(rate).minimize(y)

with tf.Session() as sess:
    sess.run(init)
    x0, y0 = sess.run([x,y])
    print(0, ') x =', x0, ', y =', y0)
    for step in range(loop):
        sess.run(optimize)
        x0, y0 = sess.run([x,y])
        print(step+1, ') x =', x0, ', y =', y0)

