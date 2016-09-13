#! /usr/bin/python3
import numpy as np
import tensorflow as tf


def get_initial_control_points(r):
    ini = 0
    last = len(r) - 1
    ax = (r[ini][0] + r[last][0]) / 2
    ay = (r[ini][1] + r[last][1]) / 2
    dnc = (r[last][0] - r[ini][0]) / 2
    dns = (r[last][1] - r[ini][1]) / 2
    x1 = ax - dnc
    y1 = ay - dns
    x2 = ax - dnc / 3
    y2 = ay - dns / 3
    x3 = ax + dnc / 3
    y3 = ay + dns / 3
    x4 = ax + dnc
    y4 = ay + dns
    return np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.float32)


def get_initial_ts(r, bs):
    npt = len(r)
    nb = len(bs)
    ts = [0] * npt
    bx12 = bs[nb - 1][0] - bs[0][0]
    by12 = bs[nb - 1][1] - bs[0][1]
    dd = bx12 * bx12 + by12 * by12
    for i in range(npt):
        ts[i] = ((r[i][0] - bs[0][0]) * bx12 + (r[i][1] - bs[0][1]) * by12) \
            / dd
    return np.array(ts, np.float32)


def print_result(label, nc, sess, diff, feed_dict=None):
    if feed_dict is None:
        value = sess.run(diff)
    else:
        value = sess.run(diff, feed_dict=feed_dict)
    print(label, nc, value)


#
# ベジェ制御点{bs[i]:i=0,3}についての最適化
#
# bs0: 制御点の初期値。ndarray。shape=(4,2)
# t0: データ点に対応する媒介変数の初期値。ndarray。shape=(n,)
# r0: データ点の座標。ndarray。shape=(n,2)
# rate: 学習係数
# loop: 逐次回数
# nc: 総ステップ数
#
# 注) コード中の説明で、obj.shapeという表現は簡単のためであり、
# objがnp.ndarrayの場合は正しいが、objがtf.Tensorの場合には正しくない。
# この場合には、tf.Tensorの場合にshapeを求めるには、tf.Tensor.get_shape()
# 関数を使用する
#
def phase1(bs0, t0, r0, rate, loop, nc):
    global summary_writer
    g = tf.Graph()
    with g.as_default():
        # データ点の数
        N = len(r0)

        # 入力となるN個の教師データ。shape = (N,)
        t = tf.placeholder(tf.float32, shape=(N,))

        # ラベルとなるN個の教師データ。shape = (N,2)
        r_ = tf.placeholder(tf.float32, shape=(N, 2))

        # tpw.shape = (4,N)
        s = 1 - t
        tpw = tf.pack([s * s * s, 3 * s * s * t, 3 * s * t * t, t * t * t])

        # 4個の制御点の座標。最適化パラメータ。bs.shape = (4,2)
        bs = tf.Variable(bs0, tf.float32)

        # 各データ点に対応する補完曲線上の点。r.shape = (N,2)
        r = tf.matmul(tpw, bs, transpose_a=True)

        # tf.Variableの初期化を行うOP
        init = tf.initialize_all_variables()

        # 目的関数
        diff = tf.reduce_mean(tf.square(r - r_))

        # トレーニング操作（最適化）
        train = tf.train.GradientDescentOptimizer(rate).minimize(diff)

        # placeholderに値を代入するための辞書
        feed_dict = {t: t0, r_: r0}
        tf.scalar_summary('diff', diff)
        summary_op = tf.merge_all_summaries()

    with tf.Session(graph=g) as sess:
        sess.run(init)
        print_result('P1', nc, sess, diff, feed_dict)
        for step in range(loop):
            sess.run(train, feed_dict=feed_dict)
            nc = nc + 1
            if (step + 1) % 100 == 0:
                print_result('P1', nc, sess, diff, feed_dict)
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, nc)
        bs1 = sess.run(bs)
    return bs1, t0, nc


#
# 各データ点に対応する媒介変数値{t[i]:i=0,N-1}についての最適化
# 各引数は、phase1()に同じ
#
def phase2(bs1, t1, r0, rate, loop, nc):
    g = tf.Graph()
    with g.as_default():
        # データ点の数
        N = len(r0)

        # t1の両端を切り取る st1.shape = (N-2,)
        st1 = t1[1:N - 1]

        # t[0]を0に固定するための定数
        fti = tf.constant([0.])

        # t[N-1]を1に固定するための定数
        ftf = tf.constant([1.])

        # 両端を切り取ったt1を初期値として、1階の変数Tensorを作成する
        st = tf.Variable(st1)

        # stの両端に定数を加えて、1階のTensorを作成する
        t = tf.concat(0, [fti, st, ftf])

        # 入力に相当する1個で、4x2サイズの教師データ。bs.shape = (4,2)
        bs = tf.placeholder(tf.float32, shape=(4, 2))

        # ラベルに相当する1個、Nx2サイズの教師データ。r_.shape = (N,2)
        r_ = tf.placeholder(tf.float32, shape=(N, 2))

        # tpw.shape = (N, 4)
        s = 1 - t
        tpw = tf.transpose(tf.pack(
            [s * s * s, 3 * s * s * t, 3 * s * t * t, t * t * t]))

        # 各{t_i;i=0,n-1}に対して、補完曲線を計算する。r.shape = (N, 2)
        r = tf.matmul(tpw, bs)

        # phase1に同じ
        init = tf.initialize_all_variables()
        diff = tf.reduce_mean(tf.square(r - r0))
        train = tf.train.GradientDescentOptimizer(rate).minimize(diff)
        feed_dict = {bs: bs1, r_: r0}
        tf.scalar_summary('diff', diff)
        summary_op = tf.merge_all_summaries()

    with tf.Session(graph=g) as sess:
        sess.run(init)
        print_result('P2', nc, sess, diff, feed_dict)
        for step in range(loop):
            sess.run(train, feed_dict=feed_dict)
            nc = nc + 1
            if (step + 1) % 10 == 0:
                print_result('P2', nc, sess, diff, feed_dict)
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, nc)
        t2 = sess.run(t)
    return bs1, t2, nc


def do_cycle(nd, r0):
    bs2 = get_initial_control_points(r0)
    t2 = get_initial_ts(r0, bs2)
    nc = 0
    for id in range(nd):
        bs1, t1, nc = phase1(bs2, t2, r0, 0.5, 1000, nc)
        bs2, t2, nc = phase2(bs1, t1, r0, 1e-5, 30, nc)
    return bs2, t2


summary_writer = tf.train.SummaryWriter('cdata')
r0 = np.loadtxt('cdata/m-cap.2.dat', delimiter=',')
ncycles = 80
bsn, tn = do_cycle(ncycles, r0)
np.savez('cdata/m-cap.2.main3.npz', bs=bsn, t=tn, nc=ncycles)
