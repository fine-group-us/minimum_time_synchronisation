# This code generates optimal time synchronization trajectories for the van der Pol oscillator for initial phase points outside the limit cycle. It also generates a color map indicating the number of bangs necessary to reach the limit cycle. It is adapted for two values of the bound of the control: k=2.0 and k=0.2 (Fig1)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import scipy.integrate as integr


mu = 0.1
k = float(sys.argv[1])

#  Limit cycle data:
data = np.loadtxt('limitcycle1bisbis2000.dat')
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



# Time of integration depending on the value of k:
if k == 2.0:
    t_exp = 3
if k == 0.2:
    t_exp = 10

# Pontryagin's equation:
def f(t,u):
    x= u[0]
    v = u[1]
    p1 = u[2]
    p2 = u[3]
    return [v, mu * (1 - x**2) * v - x + F, mu*p2*2*x*v + p2, mu*p2*(x**2-1) - p1]

fig1,ax1 = plt.subplots()

def lighten_color(color_name, amount):
    rgb = np.array(mcolors.to_rgb(color_name))
    white = np.array([1, 1, 1])
    return rgb + (white - rgb) * amount
density = False

red1 = lighten_color('tab:red', 0.0)  
red2 = lighten_color('tab:red', 0.0)
red3 = lighten_color('tab:red', 0.0)
red4 = lighten_color('tab:red', 0.0)
red5 = lighten_color('tab:red', 0.0)

blue1 = lighten_color('tab:blue', 0.0)
blue2 = lighten_color('tab:blue', 0.0)
blue3 = lighten_color('tab:blue', 0.0)
blue4 = lighten_color('tab:blue', 0.0)
blue5 = lighten_color('tab:blue', 0.0)

x_1switch_pos = []
v_1switch_pos = []
x_1switch_neg = []
v_1switch_neg = []
x_2switch_pos = []
v_2switch_pos = []
x_2switch_neg = []
v_2switch_neg = []
x_3switch_pos = []
v_3switch_pos = []
x_3switch_neg = []
v_3switch_neg = []

for i in range(0,len(xfs)):
    xf = xfs[i]
    vf = vfs[i]
    k0 = k
    t_s1 = -t_exp
    t_s2 = -t_exp
    t_s3 = -t_exp
    t_s4 = -t_exp
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
    sol = integr.solve_ivp(f, [0,-t_exp], [xf, vf,-(mu * (1 - xf**2) * vf - xf) / (F * vf),1/F], method='Radau', rtol=1e-6, atol=1e-6,events =event, dense_output=True)
    if density == True:
        output_data = np.column_stack((sol.y[0], sol.y[1], -sol.t))
        with open("trajectories.dat", "a") as file_out:
            np.savetxt(file_out, output_data)
    colour1 = red1 if F1 > 0 else blue1
    if sol.t_events[0].size > 0:
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
        sol2 = integr.solve_ivp(f, [t_s1,-t_exp], [x, v, p1, np.sign(F)*1e-10], method='Radau', rtol=1e-6, atol=1e-6,events =event, dense_output=True)
        if density == True:
            output_data = np.column_stack((sol2.y[0], sol2.y[1], -sol2.t))
            with open("trajectories.dat", "ab") as textito:
                np.savetxt(textito, output_data)
        colour2 = red2 if F2 > 0 else blue2
        if sol2.t_events[0].size >0:
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
            sol3 = integr.solve_ivp(f, [t_s2,-t_exp], [x, v, p1, np.sign(F)*1e-10], method='Radau', rtol=1e-6, atol=1e-6,events = event, dense_output=True)
            colour3 = red3 if F3 > 0 else blue3
            if sol3.t_events[0].size > 0:
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
                sol4 = integr.solve_ivp(f, [t_s3,-t_exp], [x, v, p1, np.sign(F)*1e-10], method='Radau', rtol=1e-6, atol=1e-6,events = event, dense_output=True)
                colour4 = red4 if F4 > 0 else blue4
                if sol4.t_events[0].size > 0:
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
                    sol5 = integr.solve_ivp(f, [t_s4,-t_exp], [x, v, p1, np.sign(F)*1e-10], method='Radau', rtol=1e-6, atol=1e-6,events = event, dense_output=True)
                    colour5 = red5 if F5 > 0 else blue5
                    if sol5.t_events[0].size > 0:
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
                        direction = -np.sign(F)
    if k == 2.0:
        if (np.mod(i,60) == 0 and (np.abs(xf)<1.7)):
            ax1.plot(sol.y[0], sol.y[1], color=colour1,label='Trajectory')
            if sol.t_events[0].size > 0:
                ax1.plot(sol2.y[0], sol2.y[1], color=colour2, label='Trajectory')
                if sol2.t_events[0].size > 0:
                    ax1.plot(sol3.y[0], sol3.y[1], color=colour3, label='Trajectory')
                    if sol3.t_events[0].size > 0:
                        ax1.plot(sol4.y[0], sol4.y[1], color=colour4, label='Trajectory')
                        if sol4.t_events[0].size > 0:
                            ax1.plot(sol5.y[0], sol5.y[1], color=colour5, label='Trajectory')
    if k == 0.2:
        if (np.mod(i,200) == 0 and (np.abs(xf)<1.65)):
            ax1.plot(sol.y[0], sol.y[1], color=colour1,label='Trajectory')
            if sol.t_events[0].size > 0:
                ax1.plot(sol2.y[0], sol2.y[1], color=colour2, label='Trajectory')
                if sol2.t_events[0].size > 0:
                    ax1.plot(sol3.y[0], sol3.y[1], color=colour3, label='Trajectory')
                    if sol3.t_events[0].size > 0:
                        ax1.plot(sol4.y[0], sol4.y[1], color=colour4, label='Trajectory')
                        if sol4.t_events[0].size > 0:
                            ax1.plot(sol5.y[0], sol5.y[1], color=colour5, label='Trajectory')
F = k

# Color mapping of the different zones in the phase space

# Critical trajectories:
def event_p2(t,y):
    return y[3]
event_p2.terminal = True
event_p2.direction = -1
if k == 2.0:
    time_backward = 2.5
    time_backward2 = 2.5
    time_backward3 = 2.5
if k == 0.2:
    time_backward = 3.5
    time_backward2 = 3.5
    time_backward3 = 3
backwards1 = integr.solve_ivp(f, [0, -time_backward], [-np.max(xfs),0 , 1, 0], 'Radau', rtol=1e-6, atol=1e-6, dense_output=True,events = event_p2)
backwards_x1 = backwards1.y[0]
backwards_v1 = backwards1.y[1]
event_p2.direction = 1
F = -k
backwards2 = integr.solve_ivp(f, [0, -time_backward2], [backwards_x1[-1],backwards_v1[-1] , backwards1.y[2][-1],0], 'Radau', rtol=1e-6, atol=1e-6, dense_output=True, events=event_p2)
backwards_x2 = backwards2.y[0]
backwards_v2 = backwards2.y[1]
F = k
event_p2.direction = -1
backwards3 = integr.solve_ivp(f, [0, -time_backward3], [backwards_x2[-1],backwards_v2[-1], backwards2.y[2][-1],0 ], 'Radau', rtol=1e-6, atol=1e-6, dense_output=True, events=event_p2)
backwards_x3 = backwards3.y[0]
backwards_v3 = backwards3.y[1]

# Coloring depending on the value of k: 2.0 or 0.2
if k == 2.0:
    xxx1 = np.linspace(-11, 0, 1000)
    xxx2 = np.linspace(0, 11, 1000)
    sort_xt1 = np.argsort(x_1switch_pos)
    x1s_sorted = np.array(x_1switch_pos)[sort_xt1]
    v1s_sorted = np.array(v_1switch_pos)[sort_xt1]
    x1s_sorted = np.insert(x1s_sorted,0, np.max(xfs))
    v1s_sorted = np.insert(v1s_sorted, 0, 0)
    def v1_pos(x):
        return np.interp(x, x1s_sorted, v1s_sorted)
    def fun_bajo(x):
        if x > np.max(xfs):
            return v1_pos(x)
        elif -np.max(xfs) < x < np.max(xfs):
            return vf_neg(x)
        else:
            return -v1_pos(-x)
    def fun_alto(x):
        if x > np.max(xfs):
            return v1_pos(x)
        elif -np.max(xfs) < x < np.max(xfs):
            return vf_pos(x)
        else:
            return -v1_pos(-x)
    fun_bajo = np.vectorize(fun_bajo)
    fun_alto = np.vectorize(fun_alto)

    ax1.plot(backwards_x1, backwards_v1, '--', color='red', label='B-K')
    ax1.plot(-backwards_x1, -backwards_v1, '--', color='blue', label='B+K')
    ax1.plot(x1s_sorted, v1s_sorted , ':', markersize= 5, color='black', label='S-')
    ax1.plot(-x1s_sorted, -v1s_sorted, ':', markersize =5, color='black', label='S+')
    r = ax1.fill_between(backwards_x1, backwards_v1, fun_bajo(backwards_x1), color='purple', alpha=0.3)
    b = ax1.fill_between(-backwards_x1, fun_alto(-backwards_x1), -backwards_v1, color='purple', alpha=0.3)
    xr2 = np.concatenate((np.linspace(-11, -np.max(xfs)), backwards_x1))
    r2 = ax1.fill_between(xr2, np.min(fun_bajo(backwards_x1)) * np.ones((len(xr2))),
    np.concatenate((fun_alto(np.linspace(-11, -np.max(xfs))), backwards_v1)),
    color="#16d1de", alpha=0.3)
    xb2 = np.concatenate((-backwards_x1[::-1], np.linspace(np.max(xfs), 11)))
    b2 = ax1.fill_between(xb2, 
        np.concatenate((-backwards_v1[::-1], fun_bajo(np.linspace(np.max(xfs), 11)))),
        -np.min(fun_bajo(backwards_x1)) * np.ones((len(xb2))),
        color="#16d1de", alpha=0.3
    )
    b3 = ax1.fill_between([20,21], 
        [0,0], [1,1],
        color='yellow', alpha=0.3
    )
    b4 = ax1.fill_between([20,21], 
        [0,0], [1,1],
        color='green', alpha=0.3
    )
    ax1.legend([r,b2,b3,b4], 
            [r'1-bang', 
                r'2-bang',
                r'3-bang',
                r'4-bang'], 
            loc='upper right', fontsize=20)

if k == 0.2:
    ax1.plot(backwards_x1, backwards_v1, '--', color='red', label='B-K')
    ax1.plot(-backwards_x1, -backwards_v1, '--', color='blue', label='B+K')
    ax1.plot(backwards_x2, backwards_v2, '--', color='blue')
    ax1.plot(-backwards_x2, -backwards_v2, '--', color='red')
    ax1.plot(backwards_x3, backwards_v3, '--', color='red')
    ax1.plot(-backwards_x3, -backwards_v3, '--', color='blue')
    sort_xt1 = np.argsort(x_1switch_pos)
    x1s_sorted = np.array(x_1switch_pos)[sort_xt1]
    v1s_sorted = np.array(v_1switch_pos)[sort_xt1]
    sort_xt2 = np.argsort(x_2switch_pos)
    x2s_sorted = np.array(x_2switch_pos)[sort_xt2]
    v2s_sorted = np.array(v_2switch_pos)[sort_xt2]
    sort_xt3 = np.argsort(x_3switch_pos)
    x3s_sorted = np.array(x_3switch_pos)[sort_xt3]
    v3s_sorted = np.array(v_3switch_pos)[sort_xt3]
    def v1_pos(x):
        return np.interp(x, x1s_sorted, v1s_sorted)
    def fun_alto1(x):
        if x > np.max(xfs):
            return v1_pos(x)
        if -np.max(xfs) < x < np.max(xfs):
            return vf_neg(x)
    def fun_bajo1(x):
        if -np.max(xfs) < x < np.max(xfs):
            return vf_pos(x)
        if x < -np.max(xfs):
            return -v1_pos(-x)
        # else:
        #     return np.nan  # fallback to nan if none of the above
    fun_alto1 = np.vectorize(fun_alto1, otypes=[float])
    fun_bajo1 = np.vectorize(fun_bajo1, otypes=[float])
    sort_back1x = np.argsort(backwards_x1)
    sort_backwards_x1 = backwards_x1[sort_back1x]
    sort_backwards_v1 = backwards_v1[sort_back1x]
    xx1 = np.linspace(-np.max(xfs), backwards_x1[-1], 5000)
    def v_back1(x):
        return np.interp(x, sort_backwards_x1, sort_backwards_v1)
    v_back1 = np.vectorize(v_back1, otypes=[float])
    b1 = ax1.fill_between(
        xx1, 
        v_back1(xx1), 
        fun_alto1(xx1), 
        color='purple', 
        alpha=0.3)
    ax1.fill_between(
        -xx1, 
        fun_bajo1(-xx1), 
        -v_back1(xx1), 
        color='purple', 
        alpha=0.3        
    )
    def v2_pos(x):
        return np.interp(x, x2s_sorted, v2s_sorted)
    def fun_alto2(x):
        if -backwards_x2[-1] > x > backwards_x1[-1]:
            return v2_pos(x)
        if -np.max(xfs) < x < backwards_x1[-1]:
            return v_back1(x)
        if x < -np.max(xfs):
            return -v1_pos(-x)
    fun_alto2 = np.vectorize(fun_alto2, otypes=[float])
    def fun_bajo2(x):
        if np.max(xfs) < x < backwards_x1[-1]:
            return v1_pos(x)
        if -backwards_x1[-1] < x < np.max(xfs):
            return -v_back1(-x)
        if x < -backwards_x1[-1]:
            return -v2_pos(-x)
    fun_bajo2 = np.vectorize(fun_bajo2, otypes=[float])
    sort_back2x = np.argsort(backwards_x2)
    sort_backwards_x2 = backwards_x2[sort_back2x]
    sort_backwards_v2 = backwards_v2[sort_back2x]
    def v_back2(x):
        return np.interp(x, sort_backwards_x2, sort_backwards_v2)
    v_back2 = np.vectorize(v_back2, otypes=[float])
    xx2 = np.linspace(-backwards_x1[-1], -backwards_x2[-1], 5000)
    b2 = ax1.fill_between(xx2, -v_back2(-xx2), fun_alto2(xx2),
        color="#16d1de", 
        alpha=0.3)
    ax1.fill_between(-xx2, fun_bajo2(-xx2), v_back2(-xx2),
        color="#16d1de", 
        alpha=0.3)
    def v3_pos(x):
        return np.interp(x, x3s_sorted, v3s_sorted)
    def fun_alto3(x):
        if backwards_x2[-1] < x < -backwards_x1[-1]:
            return -v2_pos(-x)
        if -backwards_x1[-1] < x < -backwards_x2[-1]:
            return -v_back2(-x)
        if x > -backwards_x2[-1]:
            return v3_pos(x)
    fun_alto3 = np.vectorize(fun_alto3, otypes=[float])
    def fun_bajo3(x):
        if backwards_x1[-1] < x < -backwards_x2[-1]:
            return v2_pos(x)
        if backwards_x2[-1] < x < backwards_x1[-1]:
            return v_back2(x)
        if x < backwards_x2[-1]:
            return -v3_pos(-x)
    fun_bajo3 = np.vectorize(fun_bajo3, otypes=[float])
    b3 = ax1.fill_between(backwards_x3, backwards_v3, fun_alto3(backwards_x3),
        color='yellow', 
        alpha=0.3)
    ax1.fill_between(-backwards_x3, fun_bajo3(-backwards_x3), -backwards_v3,
        color='yellow', 
        alpha=0.3)
    ax1.plot(x1s_sorted, v1s_sorted, ':', markersize=5, color='black', label='S+')
    ax1.plot(-x1s_sorted, -v1s_sorted, ':', markersize=5, color='black')
    ax1.plot(x2s_sorted, v2s_sorted, ':', markersize=5, color='black')
    ax1.plot(-x2s_sorted, -v2s_sorted, ':', markersize=5, color='black')
    ax1.plot(x3s_sorted, v3s_sorted, ':', markersize=5, color='black')
    ax1.plot(-x3s_sorted, -v3s_sorted, ':', markersize=5, color='black')

    
    
    xx = np.linspace(-10,backwards_x2[-1],1000)
    xxx = np.concatenate((xx,backwards_x3), axis = None)
    vvv = np.concatenate((-v3_pos(-xx),backwards_v3), axis = None)
    b4 = ax1.fill_between(xxx,-10*np.ones(len(xxx)), vvv,
        color='green', 
        alpha=0.3)
    pun = ax1.fill_between(xxx,-10*np.ones(len(xxx)), vvv, facecolor='none',
        linewidth=0.0,
        alpha=0.3)
    ax1.fill_between(
        -xxx, -vvv, 10*np.ones(len(xxx)),
        facecolor='green',
        alpha=0.3
    )
ax1.plot(xfs, vfs, '-k', label='Data Points')
ax1.set_xlim(-6,6)
ax1.set_ylim(-6,6)
ax1.set_xticks(np.arange(-6, 7, 2))
ax1.set_yticks(np.arange(-6, 7, 2))
ax1.set_xlabel(r'$x_1$', fontsize=26)
ax1.set_ylabel(r'$x_2$', fontsize=26)


# Ignore this section. It allows to drag the labels in the plot.
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

    texts.append(ax1.text(5, -4, r'BL$+$', color='tab:red', fontsize=22, ha='left'))
    texts.append(ax1.text(0.3, 3, r'BR$-$', color='tab:blue', fontsize=22, ha='right'))
    texts.append(ax1.text(-5, 2, r'S$-$', color='black', fontsize=22, ha='left'))
    texts.append(ax1.text(2.7, 0, r'S$+$', color='black', fontsize=22, ha='right'))


# General figure properties and saving
fig1.set_size_inches(6, 6, forward=True)
ax1.tick_params(axis='both', which='major', direction='in', top=True, right=True, labeltop=False, labelright=False, labelsize=22)
ax1.set_aspect('equal', adjustable='box')
fig1.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Elimina márgenes
fig1.tight_layout(pad=0)  # Sin padding extra
fig1.savefig('prueba'+str(k)+'.pdf')