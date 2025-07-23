import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import scipy.integrate as integr

# This script contains the algorithm to compute the minimum time trajectories connecting to the limit cycle of the van der Pol oscillator. 
mu = 0.1
k = float(sys.argv[1])


data = np.loadtxt('limitcycle.dat')
xfs = data[:, 0]
vfs = data[:, 1]


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

t_exp = 20
t_s10 = -t_exp
t_s20 = -t_exp
t_s30 = -t_exp
t_s40 = -t_exp
t_s50 = -t_exp

def f(t,u):
    x= u[0]
    v = u[1]
    p1 = u[2]
    p2 = u[3]
    return [v, mu * (1 - x**2) * v - x + F, mu*p2*2*x*v + p2, mu*p2*(x**2-1) - p1]

fig1,ax1 = plt.subplots()
fig2,ax2 = plt.subplots()

for i in range(0,len(xfs)):
    xf = xfs[i]
    vf = vfs[i]
    k0 = k
    t_s1 = t_s10
    t_s2 = t_s20
    t_s3 = t_s30
    t_s4 = t_s40
    adjust = 0
    if vf > 0:
        F = -k0
    else:
        F = k0
    def event(t,y):
        return y[3]
    event.terminal = True
    event.direction = -np.sign(F)
    F1=F
    sol = integr.solve_ivp(f, [0,t_s10], [xf, vf,-(mu * (1 - xf**2) * vf - xf) / (F * vf),1/F], method='Radau', rtol=1e-6, atol=1e-6,events =event, dense_output=True)
    colour = red if F1 > 0 else blue
    if sol.t_events[0].size ==0:
        t_s2 = 0
        t_s3 = 0
        t_s4 = 0
        adjust = -np.min(sol.t)
    else:
        t_s1 = sol.t_events[0][0]
        y_event = sol.y_events[0][0]
        x = y_event[0]
        v = y_event[1]
        if F1 > 0:
            x_1switch_pos.append(x)
            v_1switch_pos.append(v)
        else:
            x_1switch_neg.append(x)
            v_1switch_neg.append(v) 
        p1 = y_event[2]
        p2 = y_event[3]
        F = -F
        def event(t,y):
            return y[3]
        event.terminal = True
        event.direction = -np.sign(F)
        F2 = F
        def event(t,y):
            return y[3]
        event.terminal = True
        event.direction = -np.sign(F)

        sol2 = integr.solve_ivp(f, [t_s1,t_s10], [x, v, p1, np.sign(F)*1e-10], method='Radau', rtol=1e-6, atol=1e-6,events =event, dense_output=True)
        colour2 = red if F2 > 0 else blue
        if sol2.t_events[0].size ==0:
            t_s3 = 0
            t_s4 = 0
            adjust = -np.min(sol2.t)
        else:
            t_s2 = sol2.t_events[0][0]
            y_event = sol2.y_events[0][0]
            x = y_event[0]
            v = y_event[1]
            if F2 > 0:
                x_2switch_pos.append(x)
                v_2switch_pos.append(v)
            else:
                x_2switch_neg.append(x)
                v_2switch_neg.append(v) 
            p1 = y_event[2]
            p2 = y_event[3]
            F = -F
            def event(t,y):
                return y[3]
            event.terminal = True
            event.direction = -np.sign(F)
            F3 = F
            sol3 = integr.solve_ivp(f, [t_s2,t_s30], [x, v, p1, np.sign(F)*1e-10], method='Radau', rtol=1e-6, atol=1e-6,events = event, dense_output=True)
            colour3 = red if F3 > 0 else blue
            if sol3.t_events[0].size ==0:
                t_s4 = 0
                adjust = -np.min(sol3.t)
            else:   
                t_s3 = sol3.t_events[0][0]
                y_event = sol3.y_events[0][0]
                x = y_event[0]
                v = y_event[1]
                if F3 > 0:
                    x_3switch_pos.append(x)
                    v_3switch_pos.append(v)
                else:
                    x_3switch_neg.append(x)
                    v_3switch_neg.append(v) 
                p1 = y_event[2]
                p2 = y_event[3]
                F = -F
                def event(t,y):
                    return y[3]
                event.terminal = True
                event.direction = -np.sign(F)
                F4 = F
                sol4 = integr.solve_ivp(f, [t_s3,t_s40], [x, v, p1, np.sign(F)*1e-10], method='Radau', rtol=1e-6, atol=1e-6,events = event, dense_output=True)
                colour4 = red if F4 > 0 else blue
                if sol4.t_events[0].size == 0:
                    adjust = -np.min(sol4.t)
                else:
                    t_s4 = sol4.t_events[0][0]
                    y_event = sol4.y_events[0][0]
                    x = y_event[0]
                    v = y_event[1]
                    p1 = y_event[2]
                    p2 = y_event[3]
                    F = -F
                    def event(t,y):
                        return y[3]
                    event.terminal = True
                    event.direction = -np.sign(F)
                    F5 = F
                    sol5 = integr.solve_ivp(f, [t_s4,t_s50], [x, v, p1, np.sign(F)*1e-10], method='Radau', rtol=1e-6, atol=1e-6,events = event, dense_output=True)
                    colour5 = red if F5 > 0 else blue5
                    if sol5.t_events[0].size == 0:
                        adjust = -np.min(sol5.t)
                    else:
                        t_s5 = sol5.t_events[0][0]
                        y_event = sol5.y_events[0][0]
                        x = y_event[0]
                        v = y_event[1]
                        p1 = y_event[2]
                        p2 = y_event[3]
                        F = -F
                        def event(t,y):
                            return y[3]
                        event.terminal = True
                        event.direction = -np.sign(F)
        ax1.plot(sol.y[0], sol.y[1], color=colour1)
        if sol.t_events[0].size > 0:
            ax1.plot(sol2.y[0], sol2.y[1], color=colour2)
            if sol2.t_events[0].size > 0:
                ax1.plot(sol3.y[0], sol3.y[1], color=colour3)
                if sol3.t_events[0].size > 0:
                    ax1.plot(sol4.y[0], sol4.y[1], color=colour4)
                    if sol4.t_events[0].size > 0:
                        ax1.plot(sol5.y[0], sol5.y[1], color=colour5)
        ax2.plot(adjust + sol.t, sol.y[3], color=colour1)
        if sol.t_events[0].size > 0:
            ax2.plot(adjust + sol2.t, sol2.y[3], color=colour2)
            if sol2.t_events[0].size > 0:
                ax2.plot(adjust + sol3.t, sol3.y[3], color=colour3)
                if sol3.t_events[0].size > 0:
                    ax2.plot(adjust+ sol4.t, sol4.y[3], color=colour4)
                    if sol4.t_events[0].size > 0:
                        ax2.plot(adjust + sol5.t, sol5.y[3], color=colour5)


fig1.set_size_inches(6, 6, forward=True)
ax1.tick_params(axis='both', which='major', direction='in', top=True, right=True, labeltop=False, labelright=False, labelsize=22)
ax1.set_xlabel(r'$x_1$', fontsize=22)
ax1.set_ylabel(r'$x_2$', fontsize=22)
ax1.set_aspect('equal', adjustable='box')
fig1.subplots_adjust(left=0, right=1, top=1, bottom=0) 
fig1.tight_layout(pad=0) 
fig2.set_size_inches(6, 6, forward=True)
ax2.set_xlabel(r'$t$', fontsize=22)
ax2.set_ylabel(r'$p_2$', fontsize=22)
ax2.tick_params(axis='both', which='major', direction='in', top=True, right=True, labeltop=False, labelright=False, labelsize=22)
ax2.set_aspect('equal', adjustable='box')
fig2.subplots_adjust(left=0, right=1, top=1, bottom=0) 
fig2.tight_layout(pad=0) 



ax2.plot([0,15], [0,0], 'k--', label='Zero Line')
plt.show()