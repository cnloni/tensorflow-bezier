#! /usr/bin/python3
#
# real	1m8.270s for original on myboy
# real	0m51.617s for GraphPhase1/GraphPhase2 on myboy
# real	0m41.655s for common sessions in GraphPhase1/GraphPhase2 on myboy
#
import numpy as np
import fitter4 as ft

def do_cycle(nd, r0):
    bs2 = ft.get_initial_control_points(r0)
    t2 = ft.get_initial_ts(r0, bs2)
    gr1 = ft.GraphPhase1(0.5)
    gr2 = ft.GraphPhase2(1e-5, len(t2))
    writer = ft.create_summary_writer('cdata')
    gr1.set_summary_writer(writer)
    gr2.set_summary_writer(writer)
    dk = ft.DataKeeper()
    nc = 0
    dk.append(bs2, t2, nc)
    for id in range(nd):
        bs1, t1, nc = gr1.execute(bs2, t2, r0, 1000, nc)
        bs2, t2, nc = gr2.execute(bs1, t1, r0, 30, nc)
        dk.append(bs2, t2, nc)
    return dk


ncycles = 50
r0 = np.loadtxt('cdata/m-cap.2.dat', delimiter=',')
dk = do_cycle(ncycles, r0)
dk.save('cdata/m-cap.2.main4.npz')

