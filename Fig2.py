# This code generate Figure 2 of the article, a plot of the number of bangs of the optimal time trajectories versus the bound of the control. It calculates the critical values of K by integrating backwards the critical trajectories and finding where they intersect the x_1-axis.

import sys
import numpy as np
import scipy.integrate as integr
import matplotlib.pyplot as plt
import scipy.optimize as opt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)

# Limit cycle data:
data = np.loadtxt('limitcycle1bisbis2000.dat')
xfs = data[:, 0]
vfs = data[:, 1]

mu = 0.1
k = 10
x0s = np.arange(2.1,5.1,0.2)
v0 = 0
tfmax = 100

def fmin(t,y):
    return [y[1], mu * (1 - y[0]**(2)) * y[1] - y[0] - k]
def fmax(t,y):
    return [y[1], mu * (1 - y[0]**(2)) * y[1] - y[0] + k]

# Outside points:
ks = np.concatenate([
    np.linspace(0, 0.05, 50),
    np.linspace(0.05, 0.1, 100),
    np.linspace(0.1, 0.2, 200),
    np.linspace(0.2, 0.3, 300),
    np.linspace(0.3, 1.25, 100)
])
xs = []
kcrit1 = []
kcrit2 = []
kcrit3 = []
kcrit4 = []
xf = -np.max(xfs)
vf = 0
def event3(t,y):
    direction = 1
    return y[1]

def fmin(t,y):
    return [y[1], mu * (1 - y[0]**(2)) * y[1] - y[0] - k]

def fmax(t,y):
    return [y[1], mu * (1 - y[0]**(2)) * y[1] - y[0] + k]
for k in ks:
    solmax = integr.solve_ivp(fmax,[0,-10],[xf,vf],'Radau',rtol=1e-6,atol=1e-6,events=event3)
    if len(solmax.y_events[0])>1:
        xs.append(solmax.y_events[0][1][0])
        kcrit1.append(k)
kc1 = np.max(kcrit1)
print(r'$K_{c1}^\infty$ is ' + str(kc1))
xss = []
for i in range(len(kcrit1)):
    k = kcrit1[i]
    xf = -xs[i]
    vf = 0
    solmax = integr.solve_ivp(fmax,[0,-10],[xf,vf],'Radau',rtol=1e-6,atol=1e-6,events=event3)
    if len(solmax.y_events[0])>1:
        xss.append(solmax.y_events[0][1][0])
        kcrit2.append(k)
kc2 = np.max(kcrit2)
print(r'$K_{c2}^\infty$ is ' + str(kc2))
xsss = []
for i in range(len(kcrit2)):
    k = kcrit2[i]
    xf = -xss[i]
    vf = 0
    solmax = integr.solve_ivp(fmax,[0,-10],[xf,vf],'Radau',rtol=1e-6,atol=1e-6,events=event3)
    if len(solmax.y_events[0])>1:
        xsss.append(solmax.y_events[0][1][0])
        kcrit3.append(k)
kc3 = np.max(kcrit3)
print(r'$K_{c3}^\infty$ is ' + str(kc3))
xssss = []
for i in range(len(kcrit3)):
    k = kcrit3[i]
    xf = -xsss[i]
    vf = 0
    solmax = integr.solve_ivp(fmax,[0,-10],[xf,vf],'Radau',rtol=1e-6,atol=1e-6,events=event3)
    if len(solmax.y_events[0])>1:
        xssss.append(solmax.y_events[0][1][0])
        kcrit4.append(k)
kc4 = np.max(kcrit4)
print(r'$K_{c4}^\infty$ is ' + str(kc4))


plt.figure()
a = plt.plot(kcrit1,xs,'-',color="#07555b")[0]
b = plt.plot(kcrit2,xss,'-',color="#866C03")[0]
c = plt.plot(kcrit3,xsss,'-',color='tab:green')[0]
# d = plt.plot(kcrit4,xssss,'-',color='tab:orange')[0]
k1color = np.append(kcrit1,[kc1,1.5])
x01 = np.append(xs, [6.5,6.5])
e = plt.fill_between(k1color ,np.ones(len(x01))*np.max(xfs),x01 , color="#16d1de", alpha=0.3)
x02 = np.append(xss, 6.5*np.ones(len(xs[len(xss)-1:-1])))
f = plt.fill_between(kcrit1 ,xs ,x02 , color='yellow', alpha=0.3)
x03 = np.append(xsss, 6.5*np.ones(len(xss[len(xsss)-1:-1])))
g = plt.fill_between(kcrit2 ,xss ,x03 , color='green', alpha=0.3)
x04 = np.append(xssss, 6.5*np.ones(len(xsss[len(xssss)-1:-1])))
# h = plt.fill_between(kcrit3 ,xsss ,x04 , color='orange', alpha=0.3)


plt.plot([kc1,kc1],[np.max(xfs),6],'k--')
plt.text(kc1 - 0.01, 5.1, r'$K_{c1}$', fontsize=14)
plt.plot([kc2,kc2],[np.max(xfs),6],'k--')
plt.text(kc2 - 0.01, 5.1, r'$K_{c2}$', fontsize=14)
plt.plot([kc3,kc3],[np.max(xfs),6],'k--')
plt.text(kc3 - 0.01, 5.1, r'$K_{c3}$', fontsize=14)
plt.plot([0,1],[np.max(xfs),np.max(xfs)],'k--')
plt.ylim(0,5)
plt.xlim(0,1.25)


# Inside points:
def event(t,y):
    return y[1]
event.terminal = False
direction2 = 1
ks = np.arange(0,1.25,0.01)
x0scrit1 = []
x0scrit2 = []
x0scrit3 = []
x0scrit4 = []
kscrit1 = []
kscrit2 = []
kscrit3 = []
kscrit4 = []
xf = -np.max(xfs)
vf = 0
for k in ks:
    sol = integr.solve_ivp(fmin, [0,-tfmax], [xf, vf], events = event, dense_output = True)
    x0 = sol.y_events[0][1][0]
    if x0 > 0 and x0 < np.max(xfs):
        x0scrit1.append(x0)
        kscrit1.append(k)
        tfmin = -sol.t_events[0][1]
        tt = np.linspace(0,-tfmin,100)
for i in range(len(kscrit1)):
    xfss = x0scrit1[i]
    k = kscrit1[i]
    sol = integr.solve_ivp(fmin, [0,-tfmax], [-xfss, vf], events = event, dense_output = True)
    x0 = sol.y_events[0][1][0]
    if x0 > 0 and x0 < np.max(xfs):
        x0scrit2.append(x0)
        kscrit2.append(k)
        tfmin = -sol.t_events[0][1]
        tt = np.linspace(0,-tfmin,100)

for i in range(len(kscrit2)):
    xfsss = x0scrit2[i]
    k = kscrit2[i]
    sol = integr.solve_ivp(fmin, [0,-tfmax], [-xfsss, vf], events = event, dense_output = True)
    x0 = sol.y_events[0][1][0]
    if x0 > 0 and x0 < np.max(xfs):
        x0scrit3.append(x0)
        kscrit3.append(k)
        tfmin = -sol.t_events[0][1]
        tt = np.linspace(0,-tfmin,100)
for i in range(len(kscrit3)):
    xfsss = x0scrit3[i]
    k = kscrit3[i]
    sol = integr.solve_ivp(fmin, [0,-tfmax], [-xfsss, vf], events = event, dense_output = True)
    x0 = sol.y_events[0][1][0]
    if x0 > 0 and x0 < np.max(xfs):
        x0scrit4.append(x0)
        kscrit4.append(k)
        tfmin = -sol.t_events[0][1]
        tt = np.linspace(0,-tfmin,100)
a = plt.plot(kscrit1,x0scrit1,':',color='tab:purple')[0]
b = plt.plot(kscrit2,x0scrit2,':',color="#07555b")[0]
c = plt.plot(kscrit3,x0scrit3,':',color="#866C03")[0]
d = plt.plot(kscrit4,x0scrit4,':',color='tab:green')[0]

k1color = np.append(kscrit1, 1.5)
x01 = np.append(x0scrit1, 0)
k2color = np.append(kscrit2, kscrit1[len(kscrit2):-1])
x02 = np.append(x0scrit2,np.zeros(len(kscrit1[len(kscrit2):-1])))
k3color = np.append(kscrit3, kscrit2[len(kscrit3):-1])
x03 = np.append(x0scrit3,np.zeros(len(kscrit2[len(kscrit3):-1])))
k4color = np.append(kscrit4, kscrit3[len(kscrit4):-1])
x04 = np.append(x0scrit4,np.zeros(len(kscrit3[len(kscrit4):-1])))
ii = plt.fill_between(k1color , x01,np.max(x0scrit1)*np.ones(len(x01)), color='purple', alpha=0.3)
j = plt.fill_between(k2color, x02,x0scrit1[0:len(k2color)], color="#16d1de", alpha=0.3)
kkk = plt.fill_between(k3color, x03,x0scrit2[0:len(k3color)], color='yellow', alpha=0.3)
l = plt.fill_between(k4color, x04,x0scrit3[0:len(k4color)], color='green', alpha=0.3)

plt.plot(kscrit1[-1], 0, marker='^',color ='tab:purple')
plt.plot(kscrit2[-1], 0, marker='^',color = "#07555b")
plt.plot(kscrit3[-1], 0, marker='^',color = "#866C03")
plt.plot(kscrit4[-1], 0, marker='^',color = 'tab:green')
plt.plot([np.min(ks),np.max(ks)],[np.max(xfs),np.max(xfs)],'--',color='k')
plt.legend([ii,e,f,g],[r'1-bang', r'2-bang',r'3-bang',r'4-bang'],fontsize=16)

plt.text(kscrit1[-1] - 0.02, 0.15, r'$\hat{K}_{c1}$', fontsize=14)
plt.text(kscrit2[-1] - 0.02, 0.15, r'$\hat{K}_{c2}$', fontsize=14)
plt.text(kscrit3[-1] - 0.02, 0.15, r'$\hat{K}_{c3}$', fontsize=14)
plt.text(kscrit4[-1] - 0.02, 0.15, r'$\hat{K}_{c4}$', fontsize=14)
plt.text(1.27, 2, r'$x_{lc}^{\max}$', fontsize=14)




# General properties of the figure and saving:
plt.xlabel(r'$K$',fontsize=20)
plt.ylabel(r'$x_{10}$',fontsize=20)
plt.gca().xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: '1.0' if np.isclose(x, 1) else ('0' if np.isclose(x, 0) else '{:.1f}'.format(x)))
)
plt.tick_params(axis='both', which='major', direction='in',top=True, right=True, labeltop=False, labelright=False, labelsize=16)
plt.tight_layout()
plt.savefig('try_xcVSk_new.pdf')
