# This code computes the minimum time trajectories with, at most, 5 bang with an initial point inside the limit cycle, (0.5,0), for a choosen bound (k). It differs from the rest of the codes since now, we want trajectories to start from a specific point. Since we know the structure of the optimal trajectories, we build a function depending on change times and we minimize it using differential evolution from scipy.optimize to obtain it.

import sys
import numpy as np
import scipy.integrate as integr
import matplotlib.pyplot as plt
import scipy.optimize as opt

x0 = 0.5
v0 = 0
mu = 0.1
tfmax = 15
k = float(sys.argv[1])

#  Limit cycle data:
data = np.loadtxt('limitcycle.dat')
xfs = data[:, 0]
vfs = data[:, 1]

# Interpolation of the limit cycle data:
mask = vfs > 0
xfs_pos = xfs[mask]
vfs_pos = vfs[mask]
sort_idx_pos = np.argsort(xfs_pos)
xfs_pos_sorted = xfs_pos[sort_idx_pos]
vfs_pos_sorted = vfs_pos[sort_idx_pos]
def vf_pos(x):
     return np.interp(x,xfs_pos_sorted, vfs_pos_sorted)
mask2 = vfs < 0
xfs_neg = xfs[mask2]
vfs_neg = vfs[mask2]
sort_idx_neg = np.argsort(xfs_neg)
xfs_neg_sorted = xfs_neg[sort_idx_neg]
vfs_neg_sorted = vfs_neg[sort_idx_neg]
def vf_neg(x):
     return np.interp(x,xfs_neg_sorted, vfs_neg_sorted)

x_div = xfs[np.argmin(vfs)]
mask3 = xfs > x_div
xfs_pos2 = xfs[mask3]
vfs_pos2 = vfs[mask3]
sort_idv_pos = np.argsort(vfs_pos2)
xfs_pos_sorted2 = xfs_pos2[sort_idv_pos]
vfs_pos_sorted2 = vfs_pos2[sort_idv_pos]
def xf_pos(v):
    return np.interp(v,vfs_pos_sorted2, xfs_pos_sorted2)
mask4 = xfs < x_div
xfs_neg2 = xfs[mask4]
vfs_neg2 = vfs[mask4]
sort_idv_neg = np.argsort(vfs_neg2)
xfs_neg_sorted2 = xfs_neg2[sort_idv_neg]
vfs_neg_sorted2 = vfs_neg2[sort_idv_neg]
def xf_neg(v):
    return np.interp(v,vfs_neg_sorted2, xfs_neg_sorted2)

def event(t,y):
    return vf_neg(y[0]) - y[1]
event.terminal = False
direction = 1

def event2(t,y):
    return vf_pos(y[0]) - y[1]
event2.terminal = False 
direction2 = -1

def fmin(t,y):
    return [y[1], mu * (1 - y[0]**(2)) * y[1] - y[0] - k]
def fmax(t,y):
    return [y[1], mu * (1 - y[0]**(2)) * y[1] - y[0] + k]

# We let the trajectories to have, at most, 5 bangs. Knowing the structure of the optimal trajectories, we build a function depending on change times and we minimize it using differential evolution from scipy.optimize.
def tfmin_5bangs(t1,t2,t3,t4):
    solmin1 = integr.solve_ivp(fmin, [0, t1], [x0, v0], 'Radau', rtol=1e-6, atol=1e-6, dense_output=True, events=event)
    x1 = solmin1.y[0][-1]
    v1 = solmin1.y[1][-1]

    solmax1 = integr.solve_ivp(fmax, [t1, t2], [x1, v1], 'Radau', rtol=1e-6, atol=1e-6, dense_output=True, events=event2)
    x2 = solmax1.y[0][-1]
    v2 = solmax1.y[1][-1]

    solmin2 = integr.solve_ivp(fmin, [t2, t3], [x2, v2], 'Radau', rtol=1e-6, atol=1e-6, dense_output=True, events=event)
    x3 = solmin2.y[0][-1]
    v3 = solmin2.y[1][-1]

    solmax2 = integr.solve_ivp(fmax, [t3, t4], [x3, v3], 'Radau', rtol=1e-6, atol=1e-6, dense_output=True, events=event2)
    x4 = solmax2.y[0][-1]
    v4 = solmax2.y[1][-1]

    solmin3 = integr.solve_ivp(fmin, [t4, tfmax], [x4, v4], 'Radau', rtol=1e-6, atol=1e-6, dense_output=True, events=event)
    tfmin = tfmax

    if len(solmin1.t_events[0]) > 0:
        tfmin = solmin1.t_events[0][0]
        return tfmin
    if len(solmax1.t_events[0]) > 0:
        tfmin = solmax1.t_events[0][0]
        return tfmin
    if len(solmin2.t_events[0]) > 0:
        tfmin = solmin2.t_events[0][0]
        return tfmin
    if len(solmax2.t_events[0]) > 0:
        tfmin = solmax2.t_events[0][0]
        return tfmin
    if len(solmin3.t_events[0]) > 0:
        tfmin = solmin3.t_events[0][0]
        return tfmin
    return tfmax
# We assume constraints to ensure t1<=t2<=t3<=t4:
lc = opt.LinearConstraint([[-1, 1, 0, 0]], 0, np.inf)
lc2 = opt.LinearConstraint([[0, -1, 1, 0]], 0, np.inf)
lc3 = opt.LinearConstraint([[0, 0, -1, 1]], 0, np.inf)
result = opt.differential_evolution(lambda x : tfmin_5bangs(x[0],x[1],x[2],x[3]),bounds=[(0,tfmax),(0,tfmax),(0,tfmax),(0,tfmax)],constraints=[lc,lc2,lc3])
t1 = result.x[0]
t2 = result.x[1]
t3 = result.x[2]
t4 = result.x[3]
tfmin = result.fun
if t1>tfmin:
    t1 = tfmin
if t2>tfmin:
    t2 = tfmin
if t3>tfmin:
    t3 = tfmin
if t4>tfmin:
    t4 = tfmin

# Save the results in a file:
if tfmin<tfmax:
    tminfile = open('t_min_5bangs_inside_mu_' + str(mu) + '_x0' + str(x0) + '.dat','a')
    tminfile.write(f'{k} {t1} {t2} {t3} {t4} {tfmin}\n')
    tminfile.close()