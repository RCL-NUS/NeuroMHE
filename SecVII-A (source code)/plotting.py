"""
This file plots figures showing wrench estimation of NeuroMHE compared to NeuroBEM
==============================================================================================
Wang Bingheng, at Control and Simulation Lab, NUS, Singapore
first version: 24 Dec. 2021
second version: 27 May. 2022
thrid version: 9 Dec. 2022 after receving reviewers' comments
wangbingheng@u.nus.edu
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from sklearn.metrics import mean_squared_error
import matplotlib.ticker
import os
from matplotlib import rc




"""--------------Load data----------------"""
# NeuroBEM data ('2021-02-18-18-08-45_seg_1' 3D circle trajectory)
# NeuroBEM data ('2021-02-23-17-27-24_seg_2' Figure-8 trajectory)
Bemnn_fxy = np.load('trained_data/Bemnn_fxy_f.npy')
Bemnn_fz  = np.load('trained_data/Bemnn_fz_f.npy')
Bemnn_txy = np.load('trained_data/Bemnn_txy_f.npy')
Bemnn_tz  = np.load('trained_data/Bemnn_tz_f.npy')
Bemnn_f   = np.load('trained_data/Bemnn_f_f.npy')
Bemnn_t   = np.load('trained_data/Bemnn_t_f.npy')
# Ground truth data 
Gt_fxy    = np.load('trained_data/Gt_fxy_f.npy')
Gt_fz     = np.load('trained_data/Gt_fz_f.npy')
Gt_txy    = np.load('trained_data/Gt_txy_f.npy')
Gt_tz     = np.load('trained_data/Gt_tz_f.npy')
Gt_f      = np.load('trained_data/Gt_f_f.npy')
Gt_t      = np.load('trained_data/Gt_t_f.npy')
# NeuroMHE data
fxy_mhe   = np.load('trained_data/fxy_mhe_f.npy')
fz_mhe    = np.load('trained_data/fz_mhe_f.npy')
txy_mhe   = np.load('trained_data/txy_mhe_f.npy')
tz_mhe    = np.load('trained_data/tz_mhe_f.npy')
f_mhe     = np.load('trained_data/f_mhe_f.npy')
t_mhe     = np.load('trained_data/t_mhe_f.npy')
# Time
time      = np.load('trained_data/Time_f.npy')
"""------------Plot figures---------------"""
font1 = {'family':'Times New Roman',
         'weight':'normal',
         'style':'normal', 'size':7}

rc('font', **font1)        
cm_2_inch = 2.54
# class ScalarFormatterClass(ScalarFormatter):
#     def _set_format(self):
#         self.format = "%1.2f"
class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self,order=0,fformat="%1.1f",offset=True,mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self, useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self,vmin=None,vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$'% self.format 



if not os.path.exists("plots_in_paper"):
    os.makedirs("plots_in_paper")

n_start = 0
n_duration = len(time)
# force and torque estimation by NeuroMHE
plt.figure(1, figsize=(5/cm_2_inch,5*0.65/cm_2_inch), dpi=600)
ax = plt.gca()
plt.plot(time[n_start:n_start+n_duration]-time[n_start], Gt_fxy[n_start:n_start+n_duration], linewidth=1.5, label='Ground truth',color='black') # The author of NeuroBEM shifted the beginning of time, we do the same
plt.plot(time[n_start:n_start+n_duration]-time[n_start], fxy_mhe[n_start:n_start+n_duration], linewidth=0.6, label='NeuroMHE',color='orange')
plt.plot(time[n_start:n_start+n_duration]-time[n_start], Bemnn_fxy[n_start:n_start+n_duration], linewidth=0.5, label='NeuroBEM',color='blue')
plt.xlabel('Time [s]', labelpad=-6.5,**font1)
plt.xticks(np.arange(0,43,14), **font1)
plt.ylabel('${F_{xy}}$ [N]',labelpad=-1, **font1)
plt.yticks(np.arange(0,26,8),**font1)
mpl.rcParams['patch.linewidth']=0.5
ax.legend(loc='lower center',prop=font1) #mode = "expand",ncol = 3
plt.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
ax.tick_params(axis='x',which='major',pad=0)
ax.tick_params(axis='y',which='major',pad=0)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0.5)
plt.savefig('plots_in_paper/force_xy.png')
plt.show()

plt.figure(2, figsize=(5/cm_2_inch,5*0.65/cm_2_inch), dpi=600)
ax = plt.gca()
plt.plot(time[n_start:n_start+n_duration]-time[n_start], Gt_fz[n_start:n_start+n_duration], linewidth=1.5, label='Ground truth',color='black') # The author of NeuroBEM shifted the beginning of time, we do the same
plt.plot(time[n_start:n_start+n_duration]-time[n_start], fz_mhe[n_start:n_start+n_duration], linewidth=0.6, label='NeuroMHE',color='orange')
plt.plot(time[n_start:n_start+n_duration]-time[n_start], Bemnn_fz[n_start:n_start+n_duration], linewidth=0.5, label='NeuroBEM',color='blue')
plt.xlabel('Time [s]', labelpad=-6.5,**font1)
plt.xticks(np.arange(0,43,14), **font1)
plt.ylabel('${F_{z}}$ [N]',labelpad=-6, **font1)
plt.yticks(np.arange(-2.5,17.5,7.5),**font1)
mpl.rcParams['patch.linewidth']=0.5
# leg = ax.legend(loc='best',prop=font1)
plt.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
ax.tick_params(axis='x',which='major',pad=0)
ax.tick_params(axis='y',which='major',pad=0)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0.5)
axes = plt.axes([0.2, 0.23, 0.18, 0.19])
axes.plot(time[n_start:80]-time[n_start],Gt_fz[n_start:80], linewidth=1.5, label='Ground truth',color='black')
axes.plot(time[n_start:80]-time[n_start],fz_mhe[n_start:80], linewidth=0.6, label='NeuroMHE',color='orange')
axes.plot(time[n_start:80]-time[n_start],Bemnn_fz[n_start:80], linewidth=0.5, label='NeuroBEM',color='blue')
axes.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
axes.tick_params(axis='x',which='major',pad=-0.1)
axes.tick_params(axis='y',which='major',pad=-0.1)
for axis in ['top', 'bottom', 'left', 'right']:
    axes.spines[axis].set_linewidth(0.5)
plt.savefig('plots_in_paper/force_z.png')
plt.show()

plt.figure(3, figsize=(5/cm_2_inch,5*0.65/cm_2_inch), dpi=600)
ax = plt.gca()
plt.plot(time[n_start:n_start+n_duration]-time[n_start], Gt_txy[n_start:n_start+n_duration], linewidth=1.5, label='Ground truth',color='black') # The author of NeuroBEM shifted the beginning of time, we do the same
plt.plot(time[n_start:n_start+n_duration]-time[n_start], txy_mhe[n_start:n_start+n_duration], linewidth=0.5, label='NeuroMHE',color='orange')
plt.plot(time[n_start:n_start+n_duration]-time[n_start], Bemnn_txy[n_start:n_start+n_duration], linewidth=0.25, label='NeuroBEM',color='blue')
plt.xlabel('Time [s]', labelpad=-6.5,**font1)
plt.xticks(np.arange(0,43,14), **font1)
plt.ylabel('${\\tau_{xy}}$ [Nm]',labelpad=-2, **font1)
plt.yticks(np.arange(0,0.12,0.05),**font1)
mpl.rcParams['patch.linewidth']=0.5
# leg = ax.legend(loc='best',prop=font1)
plt.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
ax.tick_params(axis='x',which='major',pad=0)
ax.tick_params(axis='y',which='major',pad=0)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0.5)
ax.yaxis.set_major_formatter(OOMFormatter(-1,"%1.1f"))
ax.yaxis.get_offset_text().set_fontsize(6)
plt.savefig('plots_in_paper/torque_xy.png')
plt.show()

plt.figure(4, figsize=(5/cm_2_inch,5*0.65/cm_2_inch), dpi=600)
ax = plt.gca()
plt.plot(time[n_start:n_start+n_duration]-time[n_start], Gt_tz[n_start:n_start+n_duration], linewidth=1.5, label='Ground truth',color='black') # The author of NeuroBEM shifted the beginning of time, we do the same
plt.plot(time[n_start:n_start+n_duration]-time[n_start], tz_mhe[n_start:n_start+n_duration], linewidth=0.5, label='NeuroMHE',color='orange')
plt.plot(time[n_start:n_start+n_duration]-time[n_start], Bemnn_tz[n_start:n_start+n_duration], linewidth=0.25, label='NeuroBEM',color='blue')
plt.xlabel('Time [s]', labelpad=-6.5,**font1)
plt.xticks(np.arange(0,43,14), **font1)
plt.ylabel('${\\tau_{z}}$ [Nm]',labelpad=-5, **font1)
plt.yticks(np.arange(-3e-2,4e-2,3e-2),**font1)
mpl.rcParams['patch.linewidth']=0.5
# leg = ax.legend(loc='best',prop=font1)
plt.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
ax.tick_params(axis='x',which='major',pad=0)
ax.tick_params(axis='y',which='major',pad=0)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(0.5)
ax.yaxis.set_major_formatter(OOMFormatter(-2,"%1.1f"))
ax.yaxis.get_offset_text().set_fontsize(6)
plt.savefig('plots_in_paper/torque_z.png')
plt.show()

rmse_fxy       = format(mean_squared_error(fxy_mhe[n_start:n_start+n_duration], Gt_fxy[n_start:n_start+n_duration], squared=False),'.3f')
rmse_fz        = format(mean_squared_error(fz_mhe[n_start:n_start+n_duration], Gt_fz[n_start:n_start+n_duration], squared=False),'.3f')
rmse_txy       = format(mean_squared_error(txy_mhe[n_start:n_start+n_duration], Gt_txy[n_start:n_start+n_duration], squared=False),'.3f')
rmse_tz        = format(mean_squared_error(tz_mhe[n_start:n_start+n_duration], Gt_tz[n_start:n_start+n_duration], squared=False),'.3f')
rmse_f         = format(mean_squared_error(f_mhe[n_start:n_start+n_duration], Gt_f[n_start:n_start+n_duration], squared=False),'.3f')
rmse_t         = format(mean_squared_error(t_mhe[n_start:n_start+n_duration], Gt_t[n_start:n_start+n_duration], squared=False),'.3f')
rmse_fxy_bemnn = format(mean_squared_error(Bemnn_fxy[n_start:n_start+n_duration], Gt_fxy[n_start:n_start+n_duration], squared=False),'.3f')
rmse_fz_bemnn  = format(mean_squared_error(Bemnn_fz[n_start:n_start+n_duration], Gt_fz[n_start:n_start+n_duration], squared=False),'.3f')
rmse_txy_bemnn = format(mean_squared_error(Bemnn_txy[n_start:n_start+n_duration], Gt_txy[n_start:n_start+n_duration], squared=False),'.3f')
rmse_tz_bemnn  = format(mean_squared_error(Bemnn_tz[n_start:n_start+n_duration], Gt_tz[n_start:n_start+n_duration], squared=False),'.3f')
rmse_f_bemnn   = format(mean_squared_error(Bemnn_f[n_start:n_start+n_duration], Gt_f[n_start:n_start+n_duration], squared=False),'.3f')
rmse_t_bemnn   = format(mean_squared_error(Bemnn_t[n_start:n_start+n_duration], Gt_t[n_start:n_start+n_duration], squared=False),'.3f')
print('rmse_fxy_bemnn=',rmse_fxy_bemnn,'rmse_fz_bemnn=',rmse_fz_bemnn,'rmse_txy_bemnn=',rmse_txy_bemnn,'rmse_tz_bemnn=',rmse_tz_bemnn,'rmse_f_bemnn=',rmse_f_bemnn,'rmse_t_bemnn=',rmse_t_bemnn)
print('rmse_fxy=',rmse_fxy,'rmse_fz=',rmse_fz,'rmse_txy=',rmse_txy,'rmse_tz=',rmse_tz,'rmse_f=',rmse_f,'rmse_t=',rmse_t)
