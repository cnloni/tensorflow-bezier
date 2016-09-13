#! /usr/bin/python3
import numpy as np
import tensorflow as tf
import os


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


def print_result(label, sess, diff):
    value = sess.run(diff)
    print(label, value)
    if np.isnan(value):
        exit()


#
# 制御点座標{bs[i][],i=0,3}と、媒介変数{t_i,i=0,n-1}について最適化
# N=データ点の数
#
# bs0: 制御点の初期値。ndarray。shape=(4,2)
# t0: データ点に対応する媒介変数の初期値。ndarray。shape=(n,)
# r0: データ点の座標。ndarray。shape=(n,2)
# rate: 学習係数
# loop: 逐次回数
#
# 注) コード中の説明で、obj.shapeという表現は簡単のためであり、
# objがnp.ndarrayの場合は正しいが、objがtf.Tensorの場合には正しくない。
# この場合には、tf.Tensorの場合にshapeを求めるには、tf.Tensor.get_shape()
# 関数を使用する
#
def steps(bs0, t0, r0, nstep, rate, loop):
    # 新しいGraphを作成
    g = tf.Graph()

    # 作成したGraphをデフォルトにして、Operationを登録する
    with g.as_default():
        # [グラフの作成開始]
        # 各データ点に対する媒介変数の値を最適化パラメータとする。t.shape=(N,)
        t = tf.Variable(t0, tf.float32)

        # 制御点を最適化パラメータとする。bs.shape = (4,2)
        bs = tf.Variable(bs0, tf.float32)

        # 各制御点との積をとるための、tおよび(1-t)の冪。tpw.shape = (4, N)
        s = 1 - t
        tpw = tf.pack([s * s * s, 3 * s * s * t, 3 * s * t * t, t * t * t])

        # 各{t_i;i=0,n-1}に対して、補完曲線を計算する。r.shape = (N, 2)
        r = tf.matmul(tpw, bs, transpose_a=True)

        # 変数の初期化操作
        init = tf.initialize_all_variables()

        # 目的関数（Loss関数）の作成操作
        diff = tf.reduce_mean(tf.square(r - r0))

        # 最適化操作
        train = tf.train.GradientDescentOptimizer(rate).minimize(diff)
        # [グラフの作成終了]

    # Sessionを作成（明示的にGraphを渡す）
    with tf.Session(graph=g) as sess:
        # 初期化の実行
        sess.run(init)
        print_result(nstep, sess, diff)

        for step in range(loop):
            # 最適化の実行
            sess.run(train)
            nstep = nstep + 1
            if nstep % 1000 == 0:
                print_result(nstep, sess, diff)

        # Tensorの値を出力
        bs1, t1 = sess.run([bs, t])
    return bs1, t1, nstep


def get_parameters(result_file):
    r0 = np.loadtxt('cdata/m-cap.2.dat', delimiter=',')
    if os.path.exists(result_file):
        data = np.load(result_file)
        bs0 = data['bs']
        t0 = data['t']
        nstep = data['nstep']
    else:
        bs0 = get_initial_control_points(r0)
        t0 = get_initial_ts(r0, bs0)
        nstep = 0
    return r0, bs0, t0, nstep


result_file = 'cdata/m-cap.2.main1.test.npz'
r0, bs0, t0, nstep = get_parameters(result_file)
bs1, t1, nstep = steps(bs0, t0, r0, nstep, 1e-4, 2000000)
np.savez(result_file, bs=bs1, t=t1, nstep=nstep)
