# This code generates optimal time synchronization trajectories for the van der Pol oscillator for initial phase points inside the limit cycle. It also generates a color map indicating the number of bangs necessary to reach the limit cycle. It is adapted for two values of the bound of the control: k=2.0 and k=0.5 (Fig3).

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integr
import sys
from scipy.interpolate import interp1d


mu = 0.1
k = float(sys.argv[1])


data = np.loadtxt('limitcycle.dat')
xfs = data[:, 0]
vfs = data[:, 1]
def f(t,u):
    x= u[0]
    v = u[1]
    p1 = u[2]
    p2 = u[3]
    return [v, mu * (1 - x**2) * v - x + F, mu*p2*2*x*v + p2, mu*p2*(x**2-1) - p1]

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

# This case is more complex than the oustside case. Since curves starting from different points on the limit cycle intersect, we need to choose, for each point of the phase space, which of the trajectories reaches the limit cycle first. This is done in Fig3_data.py, which generates the data file 'reds_blues' + str(k)+ '.dat'. This contains the curve defining the change of sign of the control for each point in the phase space inside the limit cycle.

data = np.loadtxt('reds_blues' + str(k)+ '.dat')
x = data[:, 0]
v = data[:, 1]


# Smooth interpolation of the curve defining the change of sign of the control:
window_size = 15  
if len(x) > window_size:
    x_smooth = np.convolve(x, np.ones(window_size)/window_size, mode='valid')
    v_smooth = np.convolve(v, np.ones(window_size)/window_size, mode='valid')
else:
    x_smooth = x
    v_smooth = v

x = x_smooth
v = v_smooth
def v_int(x_val):
    return np.interp(x_val, x, v)
v_int = np.vectorize(v_int)





def event(t,y):
    return y[1] - v_int(y[0])
event.terminal = True

# k = 2
if k == 2.0:
    fig1,ax1 = plt.subplots()

    for i in range(len(xfs)):
        if np.mod(i,200)==0:
            xf = xfs[i]
            vf = vfs[i]
            if vf > 0:
                F = k
            else:
                F = -k
            sol = integr.solve_ivp(f, [0,-3], [xf, vf,-(mu * (1 - xf**2) * vf - xf) / (F * vf),1/F], method='Radau', rtol=1e-6, atol=1e-6,events =event, dense_output=True)
            if F > 0:
                ax1.plot(sol.y[0], sol.y[1],'-', color='tab:red')
            else:
                ax1.plot(sol.y[0], sol.y[1],'-', color='tab:blue')

    ax1.plot(xfs, vfs, 'k', lw=2, label='Limit Cycle')
    ax1.plot(x,v,color = 'black',linestyle='--')

    r = ax1.fill_between(x, v, vf_pos(x), color='purple', alpha=0.3)
    b = ax1.fill_between(x, vf_neg(x), v, color='purple', alpha=0.3)
    b2 = ax1.fill_between([5,6],[1,1],[2,3], color="#16d1de", alpha=0.3)
    ax1.legend([r,b2], 
            [r'1-bang', 
                r'2-bang'], 
            loc='upper right', fontsize=20)
    ax1.set_xlim(-2.05, 2.05)
    ax1.set_ylim(-2.05, 2.05)
    ax1.set_xlabel(r'$x_1$', fontsize=26)
    ax1.set_ylabel(r'$x_2$', fontsize=26)
    # Labeling of the curves:
    if True:
        dragging = {"active": False, "text": None}

        def on_press(event):
            if event.inaxes != ax1:
                return
            for t in texts:
                contains, _ = t.contains(event)
                if contains:
                    dragging["active"] = True
                    dragging["text"] = t
                    break

        def on_release(event):
            dragging["active"] = False
            dragging["text"] = None
            fig1.canvas.draw()

        def on_motion(event):
            if dragging["active"] and event.inaxes == ax1:
                dragging["text"].set_position((event.xdata, event.ydata))
                fig1.canvas.draw()

        fig1.canvas.mpl_connect("button_press_event", on_press)
        fig1.canvas.mpl_connect("button_release_event", on_release)
        fig1.canvas.mpl_connect("motion_notify_event", on_motion)
        texts = []
        texts.append(ax1.text(-0.1, 0, r'$B_c$', color='black', fontsize=22, ha='right'))



    # General properties of the figure and saving:
    fig1.set_size_inches(6, 6, forward=True)
    ax1.set_xticks(np.arange(-2.0, 2.05, 0.5))
    ax1.set_yticks(np.arange(-2.0, 2.05, 0.5))
    ax1.set_xticklabels([rf"${x:.0f}$" if x % 1 == 0 else "" for x in np.arange(-2.0, 2.05, 0.5)])
    ax1.set_yticklabels([rf"${y:.0f}$" if y % 1 == 0 else "" for y in np.arange(-2.0, 2.05, 0.5)])
    ax1.tick_params(axis='both', which='major', direction='in', top=True, right=True, labeltop=False, labelright=False, labelsize=22)
    ax1.set_aspect('equal', adjustable='box')
    fig1.tight_layout(pad=0)  # Minimiza márgenes sin ocluir etiqueta
    plt.savefig('colormap_inside_k' + str(k) + '.pdf')

# k = 0.5
if k == 0.5:
    fig1,ax1 = plt.subplots()
    F = -k
    def event2(t,y):
        return y[3]
    event2.terminal = True
    event2.direction = np.sign(F)
    solb = integr.solve_ivp(f, [0,-5], [-np.max(xfs),0,1,0], method='Radau', rtol=1e-6, atol=1e-6,events =event2, dense_output=True)
    xcut = solb.y[0][-1]
    vcut = solb.y[1][-1]
    mask = x> xcut
    def event(t,y):
        return y[1] - v_int(y[0])
    event.terminal = False
    def eventp2(t,y):
        return y[3]
    eventp2.terminal = True
    for i in range(len(xfs)):
        xf = xfs[i]
        vf = vfs[i]
        if np.mod(i,200)==0 and np.abs(xf) < 1.75:
            if vf > 0:
                F = k
            else:
                F = -k
            sol = integr.solve_ivp(f, [0,-10], [xf, vf,-(mu * (1 - xf**2) * vf - xf) / (F * vf),1/F], method='Radau', rtol=1e-6, atol=1e-6,events =eventp2, dense_output=True)
            t_switch = sol.t[-1]
            x_switch = sol.y[0][-1]
            v_switch = sol.y[1][-1]
            p1_switch = sol.y[2][-1]
            p2_switch = sol.y[3][-1]
            if F > 0:
                ax1.plot(sol.y[0], sol.y[1],'-', color='tab:red')
            else:
                ax1.plot(sol.y[0], sol.y[1],'-', color='tab:blue')
            F = -F
            sol2 = integr.solve_ivp(f, [t_switch,-6], [x_switch, v_switch, p1_switch, p2_switch], method='Radau', rtol=1e-6, atol=1e-6,events =event, dense_output=True)
            t_events = sol2.t_events[0]
            if len(t_events) == 1:
                tts = sol2.t
                mask_tt = tts > t_events[0]
                xxs = sol2.sol(tts[mask_tt])[0]
                vvs = sol2.sol(tts[mask_tt])[1]
                if F > 0:
                    ax1.plot(xxs, vvs, '-', color='tab:red')
                else:
                    ax1.plot(xxs, vvs, '-', color='tab:blue')
            else: 
                tts = sol2.t
                mask_tt = tts > t_events[1]
                if xf == 1.3437588208610494:
                    print(t_events)
                    # print(tts[mask_tt])
                    print(sol2.sol(t_events[0])[0])
                xxs = sol2.sol(tts[mask_tt])[0]
                vvs = sol2.sol(tts[mask_tt])[1]
                if F > 0:
                    ax1.plot(xxs, vvs, '-', color='tab:red')
                else:
                    ax1.plot(xxs, vvs, '-', color='tab:blue')


    xr = np.concatenate((solb.y[0],x[mask]))
    vr = np.concatenate((solb.y[1],v[mask]))
    r = ax1.fill_between(xr,vf_neg(xr), vr, color='purple', alpha=0.3)
    b = ax1.fill_between(-xr, -vr, vf_pos(-xr), color='purple', alpha=0.3)

    r2 = ax1.fill_between(solb.y[0], solb.y[1],v_int(solb.y[0]), color="#16d1de", alpha=0.3)
    b2 = ax1.fill_between(-solb.y[0],v_int(-solb.y[0]), -solb.y[1], color="#16d1de", alpha=0.3)
    ax1.plot(solb.y[0], solb.y[1], 'b--', lw=2, label='Trajectory')
    ax1.plot(-solb.y[0], -solb.y[1], 'r--', lw=2, label='Trajectory')
    mask1 = np.abs(x) < xcut
    xx1 = np.append(x[mask1], xcut)
    vv1 = np.append(v[mask1], 0)
    xx1 = np.insert(xx1, 0, -xcut)
    vv1 = np.insert(vv1, 0, 0)
    mask2 = x < -xcut
    xx2 = x[mask2]
    vv2 = v[mask2]
    mask3 = x > xcut
    xx3 = x[mask3]
    vv3 = v[mask3]
    ax1.plot(xx1, vv1, color='black', linestyle='--')
    ax1.plot(xx2,vv2,color = 'black',linestyle=':')
    ax1.plot(xx3,vv3,color = 'black',linestyle=':')
    ax1.plot(xfs, vfs, 'k', lw=2, label='Limit Cycle')
    ax1.set_xlim(-2.05, 2.05)
    ax1.set_ylim(-2.05, 2.05)
    ax1.set_xlabel(r'$x_1$', fontsize=26, labelpad=4)
    ax1.set_ylabel(r'$x_2$', fontsize=26, labelpad=5)

    # Labeling of the curves:
    if True:
        dragging = {"active": False, "text": None}

        def on_press(event):
            if event.inaxes != ax1:
                return
            for t in texts:
                contains, _ = t.contains(event)
                if contains:
                    dragging["active"] = True
                    dragging["text"] = t
                    break

        def on_release(event):
            dragging["active"] = False
            dragging["text"] = None
            fig1.canvas.draw()

        def on_motion(event):
            if dragging["active"] and event.inaxes == ax1:
                dragging["text"].set_position((event.xdata, event.ydata))
                fig1.canvas.draw()

        fig1.canvas.mpl_connect("button_press_event", on_press)
        fig1.canvas.mpl_connect("button_release_event", on_release)
        fig1.canvas.mpl_connect("motion_notify_event", on_motion)
        texts = []
        texts.append(ax1.text(-0.1, 0, r'$B_c$', color='black', fontsize=22, ha='right'))
        texts.append(ax1.text(0, -1, r'BL$-$', color='tab:blue', fontsize=22, ha='left'))
        texts.append(ax1.text(0, -1.5, r'BR$+$', color='tab:red', fontsize=22, ha='right'))
        texts.append(ax1.text(0, 1, r'S$-$', color='black', fontsize=22, ha='left'))
        texts.append(ax1.text(0, 1.5, r'S$+$', color='black', fontsize=22, ha='right'))

    # General properties of the figure and saving:
    fig1.set_size_inches(6, 6, forward=True)
    fig1.set_size_inches(6, 6, forward=True)
    ax1.set_xticks(np.arange(-2.0, 2.05, 0.5))
    ax1.set_yticks(np.arange(-2.0, 2.05, 0.5))
    ax1.set_xticklabels([rf"${x:.0f}$" if x % 1 == 0 else "" for x in np.arange(-2.0, 2.05, 0.5)])
    ax1.set_yticklabels([rf"${y:.0f}$" if y % 1 == 0 else "" for y in np.arange(-2.0, 2.05, 0.5)])
    ax1.tick_params(axis='both', which='major', direction='in', top=True, right=True, labeltop=False, labelright=False, labelsize=22)
    ax1.set_aspect('equal', adjustable='box')
    fig1.tight_layout(pad=0) 
    fig1.savefig('colormap_inside_k' + str(k) + '.pdf')




