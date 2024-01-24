"""
This file plots figures showing wrench estimation of NeuroMHE compared to NeuroBEM
==============================================================================================
Wang Bingheng, at Control and Simulation Lab, NUS, Singapore
first version: 24 Dec. 2021
second version: 27 May. 2022
thrid version: 9 Dec. 2022 after receiving the reviewers' comments
fourth version: 24 July 2023 after receiving the reviewers' comments at the 2nd round
wangbingheng@u.nus.edu
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.pyplot import MultipleLocator
from sklearn.metrics import mean_squared_error
import matplotlib.ticker
import os
from matplotlib import rc

"""--------------Load data----------------"""
# NeuroBEM data ('2021-02-23-17-27-24_seg_2' Figure-8 trajectory)
Bemnn_fxy = np.load('evaluation_slow_better_cond_B1B2A3/Bemnn_fxy_j.npy')
Bemnn_fz  = np.load('evaluation_slow_better_cond_B1B2A3/Bemnn_fz_j.npy')
Bemnn_txy = np.load('evaluation_slow_better_cond_B1B2A3/Bemnn_txy_j.npy')
Bemnn_tz  = np.load('evaluation_slow_better_cond_B1B2A3/Bemnn_tz_j.npy')
Bemnn_f   = np.load('evaluation_slow_better_cond_B1B2A3/Bemnn_f_j.npy')
Bemnn_t   = np.load('evaluation_slow_better_cond_B1B2A3/Bemnn_t_j.npy')
# Ground truth data 
Gt_fxy    = np.load('evaluation_slow_better_cond_B1B2A3/Gt_fxy_j.npy')
Gt_fz     = np.load('evaluation_slow_better_cond_B1B2A3/Gt_fz_j.npy')
Gt_txy    = np.load('evaluation_slow_better_cond_B1B2A3/Gt_txy_j.npy')
Gt_tz     = np.load('evaluation_slow_better_cond_B1B2A3/Gt_tz_j.npy')
Gt_f      = np.load('evaluation_slow_better_cond_B1B2A3/Gt_f_j.npy')
Gt_t      = np.load('evaluation_slow_better_cond_B1B2A3/Gt_t_j.npy')
# NeuroMHE data
fxy_mhe   = np.load('evaluation_slow_better_cond_B1B2A3/fxy_mhe_j.npy')
fz_mhe    = np.load('evaluation_slow_better_cond_B1B2A3/fz_mhe_j.npy')
txy_mhe   = np.load('evaluation_slow_better_cond_B1B2A3/txy_mhe_j.npy')
tz_mhe    = np.load('evaluation_slow_better_cond_B1B2A3/tz_mhe_j.npy')
f_mhe     = np.load('evaluation_slow_better_cond_B1B2A3/f_mhe_j.npy')
t_mhe     = np.load('evaluation_slow_better_cond_B1B2A3/t_mhe_j.npy')
# Time
time      = np.load('evaluation_slow_better_cond_B1B2A3/Time_j.npy')
"""------------Plot figures---------------"""
font1 = {'family':'arial',
         'weight':'normal',
         'style':'normal', 'size':6}

font2 = {'family':'arial',
         'weight':'normal',
         'style':'normal', 'size':6}

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

n_start = 0 #3200
n_duration = len(time) #1200
# ----force and torque estimation by NeuroMHE----#
fig=plt.figure(1, figsize=(8/cm_2_inch,5*0.625/cm_2_inch), dpi=600)
plt1 = plt.subplot2grid((1, 3), (0, 0), colspan=2)
plt2 = plt.subplot2grid((1, 3), (0, 2), colspan=1)
plt1.plot(time[n_start:n_start+n_duration]-time[0], Gt_fxy[n_start:n_start+n_duration], linewidth=1.5, label='Ground truth',color='black') # The author of NeuroBEM shifted the beginning of time, we do the same
plt1.plot(time[n_start:n_start+n_duration]-time[0], fxy_mhe[n_start:n_start+n_duration], linewidth=0.5, label='NeuroMHE',color='orange')
plt1.plot(time[n_start:n_start+n_duration]-time[0], Bemnn_fxy[n_start:n_start+n_duration], linewidth=0.3, label='NeuroBEM',color='blue')
plt1.add_patch(Rectangle((9, 0.1), 5.5, 6.5,alpha=1, fill=None, facecolor='none',linewidth=0.3,linestyle='--',zorder=5))
plt1.set_xlabel('Time [s]', labelpad=0,**font1)
plt1.set_xticks(np.arange(0,25,8))
plt1.set_ylabel('${F_{xy}}$ [N]',labelpad=0, **font1)
plt1.set_ylim(-0.2,12.2)
plt1.set_yticks(np.arange(0,13,4))
mpl.rcParams['patch.linewidth']=0.5
plt1.legend(loc='upper center',prop=font2, bbox_to_anchor=(0.75, 1.18),labelspacing=0.15,ncol = 3,columnspacing=0.5,borderpad=0.3,handletextpad=0.4,handlelength=1.5) #mode = "expand",ncol = 3
plt1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
plt1.tick_params(axis='x',which='major',pad=0, length=1)
plt1.tick_params(axis='y',which='major',pad=0, length=1)
for axis in ['top', 'bottom', 'left', 'right']:
    plt1.spines[axis].set_linewidth(0.5)
for t in plt1.xaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in plt1.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
plt1.set_xlim(-1,25)
plt1.arrow(14.5,6,11,0,fc="k", ec="k",head_width=0.3,zorder=1, head_length=1,clip_on =False,linewidth=0.35)
n_start = 3600 #3200
n_duration = 2200

plt2.plot(time[n_start:n_start+n_duration]-time[0], Gt_fxy[n_start:n_start+n_duration], linewidth=1.5, label='Ground truth',color='black') # The author of NeuroBEM shifted the beginning of time, we do the same
plt2.plot(time[n_start:n_start+n_duration]-time[0], fxy_mhe[n_start:n_start+n_duration], linewidth=0.5, label='NeuroMHE',color='orange')
plt2.plot(time[n_start:n_start+n_duration]-time[0], Bemnn_fxy[n_start:n_start+n_duration], linewidth=0.3, label='NeuroBEM',color='blue')
plt2.set_xticks(np.arange(9,14.5,2))
plt2.set_ylim(0,6.2)
plt2.set_yticks(np.arange(0,6.5,2))
mpl.rcParams['patch.linewidth']=0.5
plt2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
plt2.tick_params(axis='x',which='major',pad=0, length=0.5)
plt2.tick_params(axis='y',which='major',pad=0, length=0.5)
for axis in ['top', 'bottom', 'left', 'right']:
    plt2.spines[axis].set_linewidth(0.5)
for t in plt2.xaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(5)
for t in plt2.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(5)
plt.savefig('plots_in_paper/force_xy.png',bbox_inches="tight", pad_inches=0.01, dpi=600)
plt.show()


n_start = 0 #3200
n_duration = len(time) #1200
fig=plt.figure(2, figsize=(8/cm_2_inch,5*0.625/cm_2_inch), dpi=600)
plt1 = plt.subplot2grid((3, 3), (0, 0), colspan=2,rowspan=3)
plt2 = plt.subplot2grid((3, 3), (0, 2), colspan=1,rowspan=2)
plt3 = plt.subplot2grid((3, 3), (2, 2), colspan=1,rowspan=1)
plt1.plot(time[n_start:n_start+n_duration]-time[n_start], Gt_fz[n_start:n_start+n_duration], linewidth=1.5, label='Ground truth',color='black') # The author of NeuroBEM shifted the beginning of time, we do the same
plt1.plot(time[n_start:n_start+n_duration]-time[n_start], fz_mhe[n_start:n_start+n_duration], linewidth=0.5, label='NeuroMHE',color='orange')
plt1.plot(time[n_start:n_start+n_duration]-time[n_start], Bemnn_fz[n_start:n_start+n_duration], linewidth=0.3, label='NeuroBEM',color='blue')
plt1.add_patch(Rectangle((-0.25,6.75), 1.25, 1.5,alpha=1, fill=None, facecolor='none',linewidth=0.3,linestyle='--',zorder=5))
plt1.add_patch(Rectangle((8.6, 27), 6.4, 9.5,alpha=1, fill=None, facecolor='none',linewidth=0.3,linestyle='--',zorder=5))
plt1.set_xlabel('Time [s]', labelpad=0,**font1)
plt1.set_xticks(np.arange(0,25,8))
plt1.set_xlim(-1,25)
plt1.set_ylabel('${F_{z}}$ [N]',labelpad=0, **font1)
plt1.set_yticks(np.arange(5,36,10))
plt1.set_ylim(5,36.5)
mpl.rcParams['patch.linewidth']=0.5
# leg = ax.legend(loc='best',prop=font1)
plt1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
plt1.tick_params(axis='x',which='major',pad=0, length=1)
plt1.tick_params(axis='y',which='major',pad=0, length=1)
for axis in ['top','bottom', 'left', 'right']:
    plt1.spines[axis].set_linewidth(0.5)
for t in plt1.xaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in plt1.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
plt1.arrow(1,6.75,24.5,0,fc="k", ec="k",head_width=0.6,zorder=1, head_length=1,clip_on =False,linewidth=0.35)
plt1.arrow(15,32,10.5,0,fc="k", ec="k",head_width=0.6,zorder=1, head_length=1,clip_on =False,linewidth=0.35)

n_start = 3500 #3200
n_duration = 2400
plt2.plot(time[n_start:n_start+n_duration]-time[0], Gt_fz[n_start:n_start+n_duration], linewidth=1.5, label='Ground truth',color='black') # The author of NeuroBEM shifted the beginning of time, we do the same
plt2.plot(time[n_start:n_start+n_duration]-time[0], fz_mhe[n_start:n_start+n_duration], linewidth=0.5, label='NeuroMHE',color='orange')
plt2.plot(time[n_start:n_start+n_duration]-time[0], Bemnn_fz[n_start:n_start+n_duration], linewidth=0.3, label='NeuroBEM',color='blue')
plt2.set_xticks(np.arange(8.5,15,3))
plt2.set_ylim(27,36.5)
plt2.set_yticks(np.arange(29,36.5,3))
plt2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
plt2.tick_params(axis='x',which='major',pad=-0.5, length=0.5)
plt2.tick_params(axis='y',which='major',pad=0, length=0.5)
for axis in ['top','bottom', 'left', 'right']:
    plt2.spines[axis].set_linewidth(0.5)
for t in plt2.xaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(5)
for t in plt2.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(5)

n_start = 0 #3200
n_duration = 45
plt3.plot(time[n_start:n_start+n_duration]-time[0], Gt_fz[n_start:n_start+n_duration], linewidth=1.5, label='Ground truth',color='black') # The author of NeuroBEM shifted the beginning of time, we do the same
plt3.plot(time[n_start:n_start+n_duration]-time[0], fz_mhe[n_start:n_start+n_duration], linewidth=0.5, label='NeuroMHE',color='orange')
plt3.plot(time[n_start:n_start+n_duration]-time[0], Bemnn_fz[n_start:n_start+n_duration], linewidth=0.3, label='NeuroBEM',color='blue')
plt3.set_xticks(np.arange(0,0.12,0.1))
# plt2.set_ylim(0,6.2)
plt3.set_yticks(np.arange(7.2,7.6,0.2))
plt3.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
plt3.tick_params(axis='x',which='major',pad=0, length=0.5)
plt3.tick_params(axis='y',which='major',pad=0, length=0.5)
for axis in ['top', 'bottom', 'left', 'right']:
    plt3.spines[axis].set_linewidth(0.5)
for t in plt3.xaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(5)
for t in plt3.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(5)
plt.savefig('plots_in_paper/force_z.png',bbox_inches="tight", pad_inches=0.01, dpi=600)
plt.show()

n_start = 0 #3200
n_duration = len(time) #1200
plt.figure(3, figsize=(8/cm_2_inch,5*0.6/cm_2_inch), dpi=600)
plt1 = plt.subplot2grid((1, 3), (0, 0), colspan=2)
plt2 = plt.subplot2grid((1, 3), (0, 2), colspan=1)
plt1.plot(time[n_start:n_start+n_duration]-time[0], Gt_txy[n_start:n_start+n_duration], linewidth=1.5, label='Ground truth',color='black') # The author of NeuroBEM shifted the beginning of time, we do the same
plt1.plot(time[n_start:n_start+n_duration]-time[0], txy_mhe[n_start:n_start+n_duration], linewidth=0.5, label='NeuroMHE',color='orange')
plt1.plot(time[n_start:n_start+n_duration]-time[0], Bemnn_txy[n_start:n_start+n_duration], linewidth=0.3, label='NeuroBEM',color='blue')
plt1.set_xlabel('Time [s]', labelpad=0,**font1)
plt1.set_xticks(np.arange(0,25,8))
plt1.set_ylabel('${\\tau_{xy}}$ [Nm]',labelpad=0, **font1)
plt1.set_yticks(np.arange(0,0.61,0.2))
plt1.set_xlim(-1,25)
plt1.add_patch(Rectangle((7.75, -0.02), 1, 0.627,alpha=1, fill=None, facecolor='none',linewidth=0.3,linestyle='--',zorder=5))
mpl.rcParams['patch.linewidth']=0.5
# leg = ax.legend(loc='best',prop=font1)
plt1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
plt1.tick_params(axis='x',which='major',pad=0, length=1)
plt1.tick_params(axis='y',which='major',pad=0, length=1)
for axis in ['top', 'bottom', 'left', 'right']:
    plt1.spines[axis].set_linewidth(0.5)
for t in plt1.xaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in plt1.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
plt1.yaxis.set_major_formatter(OOMFormatter(-1,"%1.1f"))
plt1.yaxis.get_offset_text().set_fontsize(5)
plt1.arrow(8.75,0.57,16.75,0,fc="k", ec="k",head_width=0.015,zorder=1, head_length=1,clip_on =False,linewidth=0.25)

n_start = 3100 #3200
n_duration = 400
plt2.plot(time[n_start:n_start+n_duration]-time[0], Gt_txy[n_start:n_start+n_duration], linewidth=1.5, label='Ground truth',color='black') # The author of NeuroBEM shifted the beginning of time, we do the same
plt2.plot(time[n_start:n_start+n_duration]-time[0], txy_mhe[n_start:n_start+n_duration], linewidth=0.5, label='NeuroMHE',color='orange')
plt2.plot(time[n_start:n_start+n_duration]-time[0], Bemnn_txy[n_start:n_start+n_duration], linewidth=0.3, label='NeuroBEM',color='blue')
plt2.set_xticks(np.arange(8,9,0.5))
plt2.set_yticks(np.arange(0,0.61,0.2))
mpl.rcParams['patch.linewidth']=0.5
# leg = ax.legend(loc='best',prop=font1)
plt2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
plt2.tick_params(axis='x',which='major',pad=0, length=0.5)
plt2.tick_params(axis='y',which='major',pad=0, length=0.5)
for axis in ['top', 'bottom', 'left', 'right']:
    plt2.spines[axis].set_linewidth(0.5)
for t in plt2.xaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(5)
for t in plt2.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(5)
plt2.yaxis.set_major_formatter(OOMFormatter(-1,"%1.1f"))
plt2.yaxis.get_offset_text().set_fontsize(5)

plt.savefig('plots_in_paper/torque_xy.png',bbox_inches="tight", pad_inches=0.01, dpi=600)
plt.show()

n_start = 0 #3200
n_duration = len(time) #1200
plt.figure(4, figsize=(8/cm_2_inch,5*0.6/cm_2_inch), dpi=600)
plt1 = plt.subplot2grid((1, 3), (0, 0), colspan=2)
plt2 = plt.subplot2grid((1, 3), (0, 2), colspan=1)
plt1.plot(time[n_start:n_start+n_duration]-time[0], Gt_tz[n_start:n_start+n_duration], linewidth=1.5, label='Ground truth',color='black') # The author of NeuroBEM shifted the beginning of time, we do the same
plt1.plot(time[n_start:n_start+n_duration]-time[0], tz_mhe[n_start:n_start+n_duration], linewidth=0.5, label='NeuroMHE',color='orange')
plt1.plot(time[n_start:n_start+n_duration]-time[0], Bemnn_tz[n_start:n_start+n_duration], linewidth=0.3, label='NeuroBEM',color='blue')
plt1.set_xlabel('Time [s]', labelpad=0,**font1)
plt1.set_xticks(np.arange(0,25,8))
plt1.set_ylabel('${\\tau_{z}}$ [Nm]',labelpad=-4, **font1)
plt1.set_yticks(np.arange(-0.06,0.15,0.06))
plt1.set_xlim(-1,25)
plt1.add_patch(Rectangle((7.75, -0.08), 1, 0.18,alpha=1, fill=None, facecolor='none',linewidth=0.3,linestyle='--',zorder=5))
mpl.rcParams['patch.linewidth']=0.5
# leg = ax.legend(loc='best',prop=font1)
plt1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
plt1.tick_params(axis='x',which='major',pad=0, length=1)
plt1.tick_params(axis='y',which='major',pad=0, length=1)
for axis in ['top', 'bottom', 'left', 'right']:
    plt1.spines[axis].set_linewidth(0.5)
for t in plt1.xaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
for t in plt1.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(6)
plt1.yaxis.set_major_formatter(OOMFormatter(-1,"%1.1f"))
plt1.yaxis.get_offset_text().set_fontsize(5)
plt1.arrow(8.75,0.09,16.75,0,fc="k", ec="k",head_width=0.005,zorder=1, head_length=1,clip_on =False,linewidth=0.1)

n_start = 3100 #3200
n_duration = 400
plt2.plot(time[n_start:n_start+n_duration]-time[0], Gt_tz[n_start:n_start+n_duration], linewidth=1.5, label='Ground truth',color='black') # The author of NeuroBEM shifted the beginning of time, we do the same
plt2.plot(time[n_start:n_start+n_duration]-time[0], tz_mhe[n_start:n_start+n_duration], linewidth=0.5, label='NeuroMHE',color='orange')
plt2.plot(time[n_start:n_start+n_duration]-time[0], Bemnn_tz[n_start:n_start+n_duration], linewidth=0.3, label='NeuroBEM',color='blue')
plt2.set_xticks(np.arange(8,9,0.5))
plt2.set_yticks(np.arange(-0.04,0.081,0.04))
mpl.rcParams['patch.linewidth']=0.5
# leg = ax.legend(loc='best',prop=font1)
plt2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
plt2.tick_params(axis='x',which='major',pad=0, length=0.5)
plt2.tick_params(axis='y',which='major',pad=0, length=0.5)
for axis in ['top', 'bottom', 'left', 'right']:
    plt2.spines[axis].set_linewidth(0.5)
for t in plt2.xaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(5)
for t in plt2.yaxis.get_major_ticks(): 
    t.label.set_font('arial') 
    t.label.set_fontsize(5)
plt2.yaxis.set_major_formatter(OOMFormatter(-1,"%1.1f"))
plt2.yaxis.get_offset_text().set_fontsize(5)


plt.savefig('plots_in_paper/torque_z.png',bbox_inches="tight", pad_inches=0.01, dpi=600)
plt.show()

