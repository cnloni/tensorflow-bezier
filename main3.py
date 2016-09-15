#! /usr/bin/python3
import numpy as np
import tensorflow as tf


def get_initial_control_points(r):
    ini = 0
    last = len(r) - 1
    ax = (r[ini][0] + r[last][0]) / 2
    ay = (r[ini][1] + r[last][1]) / 2
    dnstep = (r[last][0] - r[ini][0]) / 2
    dns = (r[last][1] - r[ini][1]) / 2
    x1 = ax - dnstep
    y1 = ay - dns
    x2 = ax - dnstep / 3
    y2 = ay - dns / 3
    x3 = ax + dnstep / 3
    y3 = ay + dns / 3
    x4 = ax + dnstep
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


def print_result(label, nstep, sess, diff, feed_dict=None):
    if feed_dict is None:
        value = sess.run(diff)
    else:
        value = sess.run(diff, feed_dict=feed_dict)
    print(label, nstep, value)


#
# ベジェ制御点{bs[k]:k=0,3}についての最適化
#
# bs0: 制御点の初期値。ndarray。shape=(4,2)
# t0: データ点に対応する媒介変数の初期値。ndarray。shape=(n,)
# r0: データ点の座標。ndarray。shape=(n,2)
# rate: 学習係数
# nstep: 総ステップ数
# loop: 逐次回数
#
# 注) コード中の説明で、obj.shapeという表現は簡単のためであり、
# objがnp.ndarrayの場合は正しいが、objがtf.Tensorの場合には正しくない。
# この場合には、tf.Tensorの場合にshapeを求めるには、tf.Tensor.get_shape()
# 関数を使用する
#
def phase1(bs0, t0, r0, rate, nstep, loop):
    g = tf.Graph()
    with g.as_default():
        # [グラフの作成開始]
        # 入力となるN個の教師データ。shape = (N,)
        t = tf.constant(t0, tf.float32)

        # T.shape = (N, 4)
        # ちょっと手抜きをした記述
        s = 1 - t
        T = tf.pack([s * s * s, 3 * s * s * t, 3 * s * t * t, t * t * t])

        # 4個の制御点の座標。最適化パラメータ。bs.shape = (4,2)
        bs = tf.Variable(bs0, tf.float32)

        # 各データ点に対応する補完曲線上の点。r.shape = (N,2)
        r = tf.matmul(T, bs, transpose_a=True)

        # tf.Variableの初期化を行うOP
        init = tf.initialize_all_variables()

        # 目的関数
        # 手抜きをしてr0を直接、演算に投入している
        diff = tf.reduce_mean(tf.square(r - r0))

        # トレーニング操作（最適化）
        train = tf.train.GradientDescentOptimizer(rate).minimize(diff)

        # [グラフの作成終了]
        g.finalize()

    with tf.Session(graph=g) as sess:
        sess.run(init)
        print_result('P1', nstep, sess, diff)
        for step in range(loop):
            sess.run(train)
            nstep = nstep + 1
            if (step + 1) % 100 == 0:
                print_result('P1', nstep, sess, diff)
        bs1 = sess.run(bs)
    return bs1, t0, nstep


#
# 各データ点に対応する媒介変数値{t[i]:i=0,N-1}についての最適化
# 各引数は、phase1()に同じ
#
def phase2(bs1, t1, r0, rate, nstep, loop):
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
        bs = tf.constant(bs1, shape=(4, 2))

        # T.shape = (4, N)
        s = 1 - t
        T = tf.pack([s * s * s, 3 * s * s * t, 3 * s * t * t, t * t * t])

        # 各{t_i;i=0,n-1}に対して、補完曲線を計算する。r.shape = (N, 2)
        r = tf.matmul(T, bs, transpose_a=True)

        # phase1に同じ
        init = tf.initialize_all_variables()
        diff = tf.reduce_mean(tf.square(r - r0))
        train = tf.train.GradientDescentOptimizer(rate).minimize(diff)
        g.finalize()

    with tf.Session(graph=g) as sess:
        sess.run(init)
        print_result('P2', nstep, sess, diff)
        for step in range(loop):
            sess.run(train)
            nstep = nstep + 1
            if (step + 1) % 10 == 0:
                print_result('P2', nstep, sess, diff)
        t2 = sess.run(t)
    return bs1, t2, nstep


def do_cycle(nd, r0):
    bs2 = get_initial_control_points(r0)
    t2 = get_initial_ts(r0, bs2)
    nstep = 0
    for id in range(nd):
        bs1, t1, nstep = phase1(bs2, t2, r0, 0.5, nstep, 1000)
        bs2, t2, nstep = phase2(bs1, t1, r0, 1e-5, nstep, 30)
    return bs2, t2, nstep


r0 = np.loadtxt('cdata/m-cap.2.dat', delimiter=',')
nstepycles = 80
bsn, tn, nstep = do_cycle(nstepycles, r0)
np.savez('cdata/m-cap.2.main3.npz', bs=bsn, t=tn, nstep=nstep)
