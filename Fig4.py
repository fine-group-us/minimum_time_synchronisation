# This code generates a log-log plot of the minimum time versus K for both inside and outside initial points. It uses previously generated files with the data of trajectories starting in (5,0) and (0.5,0).

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)

# OUTSIDE (left)
data5_out = np.loadtxt('t_min_5bangs_mu_0.1_x05.0.dat')
ks5_out = data5_out[:,0]
tfmin5_out = data5_out[:, 5]

# INSIDE (right)
data5_in = np.loadtxt('t_min_5bangs_inside_mu_0.1_x00.5.dat')
ks5_in = data5_in[:,0]
tfmin5_in = data5_in[:, 5]

fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

# OUTSIDE subplot
ax = axs[0]
ax.loglog(ks5_out, tfmin5_out, '.', color='tab:blue')
ax.fill_betweenx([30, 31], 0.67, 10, color='purple', alpha=0.3, label='1-bang')
ax.fill_betweenx([0.1, 17], 0.67, 10, color="#16d1de", alpha=0.3, label='2-bang')
ax.fill_betweenx([0.1, 17], 0.26, 0.67, color='yellow', alpha=0.3, label='3-bang')
ax.fill_betweenx([0.1, 17], 0.13, 0.26, color='green', alpha=0.3, label='4-bang')
ax.fill_betweenx([0.1, 17], 0.07, 0.13, color='orange', alpha=0.3, label='5-bang')
ax.plot([0.67, 0.67], [0.1, 17], '--k')
ax.plot([0.26, 0.26], [0.1, 17], '--k')
ax.plot([0.13, 0.13], [0.1, 17], '--k')
ax.plot([0.07, 0.07], [0.1, 17], '--k')
ax.set_xlim(0.08, 10)
ax.set_ylim(0.1,17)
ax.set_xlabel(r'$K$', fontsize=26)
ax.set_ylabel(r'$t_f^{\min}$', fontsize=26)
ax.legend(loc='upper right', fontsize=20)
ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=22)

# INSIDE subplot
ax = axs[1]
ax.loglog(ks5_in, tfmin5_in, '.', color='tab:blue')
ax.fill_betweenx([0.1, 17], 0.755, 10, color='purple', alpha=0.3, label='One-bang')
ax.fill_betweenx([0.1, 17], 0.3421, 0.755, color="#16d1de", alpha=0.3, label='Two-bangs')
ax.fill_betweenx([0.1, 17], 0.2111, 0.3421, color='yellow', alpha=0.3, label='Three-bangs')
ax.fill_betweenx([0.1, 17], 0.146, 0.2111, color='green', alpha=0.3, label='Four-bangs')
ax.fill_betweenx([20, 21], 0.146, 0.2111, color='orange', alpha=0.3, label='Five-bangs')
ax.plot([0.755, 0.755], [0.1, 17], '--k')
ax.plot([0.3421, 0.3421], [0.1, 17], '--k')
ax.plot([0.2111, 0.2111], [0.1, 17], '--k')

ax.set_xlim(0.146, 10)
ax.set_xticks([0.1, 1.0, 10.0])  
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.get_xaxis().set_minor_formatter(plt.NullFormatter())
ax.set_xscale('log')

ax.set_xlim(0.146, 10)
ax.set_ylim(0.1,17)
ax.set_xlabel(r'$K$', fontsize=26)
ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=22)


fig.tight_layout(pad=0)
plt.savefig('Fig4.pdf')
