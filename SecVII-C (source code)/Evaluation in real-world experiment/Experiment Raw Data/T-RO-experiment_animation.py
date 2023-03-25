"""
Animation of the physical experiments for the T-RO submission (22-0575)
============================================================================
Wang, Bingheng, 27 Jan. 2023 at Control & Simulation Lab, NUS, Singapore
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
import matplotlib.ticker

font1 = {'family':'Times New Roman',
         'weight':'normal',
         'style':'normal', 'size':7}
cm_2_inch   = 2.54
sample_rate = 50
dt          = 1/sample_rate # note that we recorded the data at 50Hz


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

def play_animation_exp1_3d(position, position_ref, dt,save_option=0):
    fig = plt.figure(figsize=(4.5/cm_2_inch,4/cm_2_inch),dpi=300)
    ax = plt.axes(projection="3d")
    ax.set_xlabel('x [m]',labelpad=-10,**font1)
    ax.set_ylabel('y [m]',labelpad=-10,**font1)
    ax.set_zlabel('z [m]',labelpad=-8,**font1)
    ax.tick_params(axis='x',which='major',pad=-5)
    ax.tick_params(axis='y',which='major',pad=-5)
    ax.tick_params(axis='z',which='major',pad=-3)
    for t in ax.xaxis.get_major_ticks(): 
        t.label.set_font('Times New Roman') 
        t.label.set_fontsize(7)
    for t in ax.yaxis.get_major_ticks(): 
        t.label.set_font('Times New Roman') 
        t.label.set_fontsize(7)
    for t in ax.zaxis.get_major_ticks(): 
        t.label.set_font('Times New Roman') 
        t.label.set_fontsize(7)
    ax.view_init(30,60)
    ax.set_zlim(0, 2)
    ax.set_ylim(-1.25, 1.25)
    ax.set_xlim(-1.25, 1.25)
    ax.xaxis._axinfo["grid"].update({"linewidth":0.5,"linestyle":'--'})
    ax.yaxis._axinfo["grid"].update({"linewidth":0.5,"linestyle":'--'})
    ax.zaxis._axinfo["grid"].update({"linewidth":0.5,"linestyle":'--'})
    # leg=plt.legend(['Desired','Actual'],prop=font1,loc=(0.05,0.6),labelspacing = 0.1)
    # leg.get_frame().set_linewidth(0.5)
    ax.view_init(30,60)

    # data
    sim_horizon = np.size(position, 1)

    # animation
    line_traj, = ax.plot3D(position[0,:1], position[1,:1], position[2,:1], linewidth=1,color='orange')
    line_traj_ref, = ax.plot3D(position_ref[0, :1], position_ref[1, :1], position_ref[2, :1], linewidth=1.5,linestyle='--',color='black')
    
    # customize
    if position_ref is not None:
        leg=plt.legend(['Actual', 'Desired'],prop=font1,loc='lower right',labelspacing=0.15)
        leg.get_frame().set_linewidth(0.5)

    def update_traj(num):
        # trajectory
        line_traj.set_data_3d(position[0,:num], position[1,:num], position[2,:num])
        # trajectory ref
        num=sim_horizon-1
        line_traj_ref.set_data_3d(position_ref[0,:num], position_ref[1,:num],position_ref[2,:num])
        return  line_traj_ref,line_traj

    ani = animation.FuncAnimation(fig, update_traj, sim_horizon, interval=dt*sim_horizon, blit=True)
    if save_option != 0:
        # Writer = animation.writers['ffmpeg']
        # plt.rcParams['animation.ffmpeg_path'] = u'/home/nusuav/anaconda3/bin/ffmpeg'    
        writer = animation.FFMpegWriter()
        # writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=-1)
        ani.save('.mp4', writer=writer, dpi=600)
        print('save_success')
    plt.show()

def play_animation_exp2_pos(time, traj, traj_ref, traj_big, dw_start,dt,save_option=0): 
    fig, (ax1,ax2) = plt.subplots(2,sharex=True,figsize=(5/cm_2_inch,5*0.65/cm_2_inch),dpi=300)
    ax1.set_xticks(np.arange(0,25,8))
    ax1.tick_params(axis='y',which='major',pad=0)
    ax1.set_ylabel('$p_{xy}$ [m]',labelpad=-2,**font1)
    ax1.yaxis.set_major_formatter(OOMFormatter(-1,"%1.1f"))
    ax1.yaxis.get_offset_text().set_fontsize(6)
    ax1.add_patch(Rectangle((dw_start, 0), 9, 0.4,
             facecolor = 'blue',
             fill=True,
             alpha = 0.1,
             lw=0.1))
    ax1.add_patch(Rectangle((dw_start+9, 0), 2, 0.4,
             facecolor = 'navy',
             fill=True,
             alpha = 0.2,
             lw=0.1))
    
    ax2.set_xticks(np.arange(0,25,8))
    ax2.tick_params(axis='x',which='major',pad=-0.5)
    ax2.tick_params(axis='y',which='major',pad=0)
    ax2.set_xlabel('Time [s]',labelpad=-6,**font1)
    ax2.set_ylabel('$p_{z}$ [m]',labelpad=4,**font1)
    ax2.add_patch(Rectangle((dw_start, 0), 9, 2.2,
             facecolor = 'blue',
             fill=True,
             alpha = 0.1,
             lw=0.1))
    ax2.add_patch(Rectangle((dw_start+9, 0), 2, 2.2,
             facecolor = 'navy',
             fill=True,
             alpha = 0.2,
             lw=0.1))

    for t in ax1.yaxis.get_major_ticks(): 
        t.label.set_font('Times New Roman') 
        t.label.set_fontsize(7)
    for t in ax2.xaxis.get_major_ticks(): 
        t.label.set_font('Times New Roman') 
        t.label.set_fontsize(7)
    for t in ax2.yaxis.get_major_ticks(): 
        t.label.set_font('Times New Roman') 
        t.label.set_fontsize(7)
    ax1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
    ax2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(0.5)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax2.spines[axis].set_linewidth(0.5)  

    ax1.set_xlim(0, 25)
    ax1.set_ylim(0, 0.4)
    ax2.set_xlim(0, 25)
    ax2.set_ylim(0, 2.3) 
    # data
    sim_horizon = np.size(traj, 1)

    # animation
    line_traj_ref, = ax2.plot(time[0],traj_ref[2,:1],linewidth=0.5,color='black')
    line_traj_xy, = ax1.plot(time[0],traj[3,:1],linewidth=0.5,color='orange')
    line_traj_z, = ax2.plot(time[0],traj[2,:1], linewidth=0.5,color='orange')
    line_traj_big, = ax2.plot(time[0],traj_big[2,:1],linewidth=0.5,color='green')

    if traj_ref is not None:
        leg=ax2.legend(['Desired: ego quadrotor', 'Actual: ego quadrotor', 'Actual: 2nd quadrotor'],loc=(0.1,0.6),prop=font1,labelspacing=0.1)
        leg.get_frame().set_linewidth(0.5)

    def update_traj(num):
        line_traj_z.set_data(time[:num], traj[2,:num])
        line_traj_xy.set_data(time[:num],traj[3,:num])
        line_traj_big.set_data(time[:num],traj_big[2,:num])
        # trajectory ref
        num=sim_horizon-1
        line_traj_ref.set_data(time[:num], traj_ref[2,:num])

        return line_traj_z,line_traj_xy,line_traj_big,line_traj_ref

    ani = animation.FuncAnimation(fig, update_traj, sim_horizon, interval=dt*sim_horizon, blit=True)
    if save_option != 0:
        # Writer = animation.writers['ffmpeg']
        # plt.rcParams['animation.ffmpeg_path'] = u'/home/nusuav/anaconda3/bin/ffmpeg'    
        writer = animation.FFMpegWriter()
        # writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=-1)
        ani.save('.mp4', writer=writer, dpi=600)
        print('save_success')
    plt.show()


def play_animation_exp1_force(time, disf, disf_ref, tension, dt,save_option=0):
    fig, (ax1,ax2,ax3) = plt.subplots(3,sharex=True,figsize=(6/cm_2_inch,6*0.8/cm_2_inch),dpi=300)
    ax1.set_ylabel('$d_{x}$ [N]',labelpad=0,**font1)
    ax2.set_ylabel('$d_{y}$ [N]',labelpad=0,**font1)
    ax3.set_ylabel('$d_{z}$ [N]',labelpad=-0.5,**font1)
    ax3.set_xlabel('Time [s]',labelpad=-1.5,**font1) # -4 for 8
    ax1.tick_params(axis='y',which='major',pad=-0.25)
    ax2.tick_params(axis='y',which='major',pad=-0.25)
    ax3.tick_params(axis='x',which='major',pad=-1)
    ax3.tick_params(axis='y',which='major',pad=-0.25)
    ax3.set_xticks(np.arange(0,35,10))
    for t in ax1.yaxis.get_major_ticks(): 
        t.label.set_font('Times New Roman') 
        t.label.set_fontsize(7)
    for t in ax2.yaxis.get_major_ticks(): 
        t.label.set_font('Times New Roman') 
        t.label.set_fontsize(7)
    for t in ax3.xaxis.get_major_ticks(): 
        t.label.set_font('Times New Roman') 
        t.label.set_fontsize(7)
    for t in ax3.yaxis.get_major_ticks(): 
        t.label.set_font('Times New Roman') 
        t.label.set_fontsize(7)
    # leg=ax1.legend(['Ground truth', 'Estimation', 'Tension force'],loc='lower center',prop=font1,labelspacing=0.1)
    # leg.get_frame().set_linewidth(0.5)
    ax1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
    ax2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
    ax3.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(0.5)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax2.spines[axis].set_linewidth(0.5)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax3.spines[axis].set_linewidth(0.5)
    ax1.set_xlim(0, 35)
    ax1.set_ylim(-8, 8)
    ax2.set_xlim(0, 35)
    ax2.set_ylim(-8, 8)
    ax3.set_xlim(0, 35)
    ax3.set_ylim(-13, 2)
       
    # data
    sim_horizon = np.size(disf, 1)

    # animation
    line_traj_ref_x, = ax1.plot(time[0],disf_ref[0, :1], linewidth=0.5,color='black')
    line_traj_x, = ax1.plot(time[0],disf[0,:1], linewidth=0.5, color='orange')
    ten_traj_x, = ax1.plot(time[0],tension[0,:1], linewidth=0.5, color='blue')
    line_traj_ref_y, = ax2.plot(time[0],disf_ref[1, :1], linewidth=0.5,color='black')
    line_traj_y, = ax2.plot(time[0],disf[1,:1], linewidth=0.5, color='orange')
    ten_traj_y, = ax2.plot(time[0],tension[1,:1], linewidth=0.5, color='blue')
    line_traj_ref_z, = ax3.plot(time[0],disf_ref[2, :1], linewidth=0.5,color='black')
    line_traj_z, = ax3.plot(time[0],disf[2,:1], linewidth=0.5, color='orange')
    ten_traj_z, = ax3.plot(time[0],tension[2,:1], linewidth=0.5, color='blue')

    # customize
    if disf_ref is not None:
        leg=ax1.legend(['Ground truth', 'Estimation', 'Tension force'],loc='upper center',prop=font1,labelspacing=0.15)
        leg.get_frame().set_linewidth(0.5)

    def update_traj(num):
        line_traj_x.set_data(time[:num], disf[0,:num])
        line_traj_ref_x.set_data(time[:num], disf_ref[0,:num])
        ten_traj_x.set_data(time[:num],tension[0,:num])
        line_traj_y.set_data(time[:num], disf[1,:num])
        line_traj_ref_y.set_data(time[:num], disf_ref[1,:num])
        ten_traj_y.set_data(time[:num],tension[1,:num])
        line_traj_z.set_data(time[:num], disf[2,:num])
        line_traj_ref_z.set_data(time[:num], disf_ref[2,:num])
        ten_traj_z.set_data(time[:num],tension[2,:num])

        return line_traj_ref_x, line_traj_x, ten_traj_x, line_traj_ref_y, line_traj_y, ten_traj_y, line_traj_ref_z, line_traj_z, ten_traj_z

    ani = animation.FuncAnimation(fig, update_traj, sim_horizon, interval=dt*sim_horizon, blit=True)
    if save_option != 0:
        # Writer = animation.writers['ffmpeg']
        # plt.rcParams['animation.ffmpeg_path'] = u'/home/nusuav/anaconda3/bin/ffmpeg'    
        writer = animation.FFMpegWriter()
        # writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=-1)
        ani.save('.mp4', writer=writer, dpi=600)
        print('save_success')

    plt.show()

def play_animation_exp2_force(time, disf, disf_ref, dw_start, dt,save_option=0):
    fig, (ax1,ax2) = plt.subplots(2,sharex=True,figsize=(6/cm_2_inch,6*0.6/cm_2_inch),dpi=300)
    ax1.set_xticks(np.arange(0,25,8))
    ax1.tick_params(axis='y',which='major',pad=0)
    ax1.set_ylabel('$d_{xy}$ [N]',labelpad=4,**font1) 
    ax1.add_patch(Rectangle((dw_start, 0), 9, 1.8,
             facecolor = 'blue',
             fill=True,
             alpha = 0.1,
             lw=0.1))
    ax1.add_patch(Rectangle((dw_start+9, 0), 2, 1.8,
             facecolor = 'navy',
             fill=True,
             alpha = 0.2,
             lw=0.1))
    ax2.set_xticks(np.arange(0,25,8))
    ax2.tick_params(axis='x',which='major',pad=-0.5)
    ax2.tick_params(axis='y',which='major',pad=0)
    ax2.set_xlabel('Time [s]',labelpad=-5.5,**font1)
    ax2.set_ylabel('$d_{z}$ [N]',labelpad=0,**font1)
    ax2.add_patch(Rectangle((dw_start, -10), 9, 10,
             facecolor = 'blue',
             fill=True,
             alpha = 0.1,
             lw=0.1))
    ax2.add_patch(Rectangle((dw_start+9, -10), 2, 10,
             facecolor = 'navy',
             fill=True,
             alpha = 0.2,
             lw=0.1))
    ax1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
    ax2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
    for t in ax1.yaxis.get_major_ticks(): 
        t.label.set_font('Times New Roman') 
        t.label.set_fontsize(7)
    for t in ax2.xaxis.get_major_ticks(): 
        t.label.set_font('Times New Roman') 
        t.label.set_fontsize(7)
    for t in ax2.yaxis.get_major_ticks(): 
        t.label.set_font('Times New Roman') 
        t.label.set_fontsize(7)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(0.5)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax2.spines[axis].set_linewidth(0.5)
    
    ax1.set_xlim(0, 25)
    ax1.set_ylim(0, 2)
    ax2.set_xlim(0, 25)
    ax2.set_ylim(-8, 0.5) 
 
    # data
    sim_horizon = np.size(disf, 1)

    # animation
    line_traj_ref_xy, = ax1.plot(time[0],disf_ref[3, :1], linewidth=0.5,color='black')
    line_traj_xy, = ax1.plot(time[0],disf[3,:1], linewidth=0.5,color='orange')
    line_traj_ref_z, = ax2.plot(time[0],disf_ref[2, :1], linewidth=0.5,color='black')
    line_traj_z, = ax2.plot(time[0],disf[2,:1], linewidth=0.5,color='orange')
    
    # customize
    # if disf_ref is not None:
    #     leg=ax2.legend(['Ground truth', 'Estimation'],loc='upper center',prop=font1,labelspacing=0.15)
    #     leg.get_frame().set_linewidth(0.5)

    def update_traj(num):
        line_traj_xy.set_data(time[:num], disf[3,:num])
        line_traj_ref_xy.set_data(time[:num], disf_ref[3,:num])
        line_traj_z.set_data(time[:num], disf[2,:num])
        line_traj_ref_z.set_data(time[:num], disf_ref[2,:num])
        
        return line_traj_ref_xy, line_traj_xy, line_traj_ref_z, line_traj_z

    ani = animation.FuncAnimation(fig, update_traj, sim_horizon, interval=dt*sim_horizon, blit=True)
    if save_option != 0:
        # Writer = animation.writers['ffmpeg']
        # plt.rcParams['animation.ffmpeg_path'] = u'/home/nusuav/anaconda3/bin/ffmpeg'    
        writer = animation.FFMpegWriter()
        # writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=-1)
        ani.save('.mp4', writer=writer, dpi=600)
        print('save_success')
    plt.show()

# def play_animation_exp1_gamma(time, gamma, dt,save_option=0):
#     fig, (ax1,ax2) = plt.subplots(2,sharex=True,figsize=(8/cm_2_inch,8/cm_2_inch),dpi=300)
#     ax1.set_ylabel('$\gamma_{1}$',labelpad=0,**font1)
#     ax2.set_ylabel('$\gamma_{2}$',labelpad=0,**font1)
#     ax2.set_xlabel('Time [s]',labelpad=0,**font1)
#     ax1.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
#     ax2.grid(b=True,axis='both',linestyle='--', linewidth=0.5)
#     ax1.set_xlim(0, 100)
#     ax1.set_ylim(0.66, 0.75)
#     ax2.set_xlim(0, 100)
#     ax2.set_ylim(0.18, 0.28)
#     for t in ax1.yaxis.get_major_ticks(): 
#         t.label.set_font('Times New Roman') 
#         t.label.set_fontsize(7)
#     for t in ax2.xaxis.get_major_ticks(): 
#         t.label.set_font('Times New Roman') 
#         t.label.set_fontsize(7)
#     for t in ax2.yaxis.get_major_ticks(): 
#         t.label.set_font('Times New Roman') 
#         t.label.set_fontsize(7)

#     for axis in ['top', 'bottom', 'left', 'right']:
#         ax1.spines[axis].set_linewidth(0.5)
#     for axis in ['top', 'bottom', 'left', 'right']:
#         ax2.spines[axis].set_linewidth(0.5)
#     # data
#     sim_horizon = np.size(gamma, 1)

#     # animation
#     line_traj_1, = ax1.plot(time[0],gamma[0,:1], linewidth=0.5)
#     line_traj_2, = ax2.plot(time[0],gamma[1,:1], linewidth=0.5)
    
#     def update_traj(num):
#         line_traj_1.set_data(time[:num], gamma[0,:num])
#         line_traj_2.set_data(time[:num], gamma[1,:num])
        
#         return line_traj_1, line_traj_2
#     ani = animation.FuncAnimation(fig, update_traj, sim_horizon, interval=dt*sim_horizon, blit=True)
#     if save_option != 0:
#         # Writer = animation.writers['ffmpeg']
#         # plt.rcParams['animation.ffmpeg_path'] = u'/home/nusuav/anaconda3/bin/ffmpeg'    
#         writer = animation.FFMpegWriter()
#         # writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=-1)
#         ani.save('.mp4', writer=writer, dpi=600)
#         print('save_success')
#     plt.show()



"""
load data of Experiment 1
"""

# ----NeuroMHE-----#
# ref = np.load('ref_enu_neuromhe.npy')
# position_enu = np.load('pos_enu_neuromhe.npy')
# play_animation_exp1_3d(position_enu,ref,dt)

# time = np.load('Time_track_neuromhe.npy')
# tension = np.load('Tension_neuromhe.npy')
# disf_ref  = np.load('Gt_lpf_neuromhe.npy')
# disf = np.load('Df_Imhe_neuromhe.npy')
# play_animation_exp1_force(time, disf, disf_ref, tension, dt)

# ----DMHE-----#
# ref = np.load('ref_enu_dmhe.npy')
# position_enu = np.load('pos_enu_dmhe.npy')
# play_animation_exp1_3d(position_enu,ref,dt)

# time = np.load('Time_track_dmhe.npy')
# tension = np.load('Tension_dmhe.npy')
# disf_ref  = np.load('Gt_lpf_dmhe.npy')
# disf = np.load('Df_Imhe_dmhe.npy')
# play_animation_exp1_force(time, disf, disf_ref, tension, dt)

# ----L1-AC-----#
# ref = np.load('ref_enu_l1.npy')
# position_enu = np.load('pos_enu_l1.npy')
# play_animation_exp1_3d(position_enu,ref,dt)

# time = np.load('Time_track_l1.npy')
# tension = np.load('Tension_l1.npy')
# disf_ref  = np.load('Gt_lpf_l1.npy')
# disf = np.load('Df_Imhe_l1.npy')
# play_animation_exp1_force(time, disf, disf_ref, tension, dt)

# ----Baseline-----#
# ref = np.load('ref_enu_neuromhe.npy')
# position_enu = np.load('pos_enu_baseline.npy')
# play_animation_exp1_3d(position_enu,ref,dt)

"""
load data of Experiment 2
"""

# ----NeuroMHE-----#
# time = np.load('Time_dw_neuromhe.npy')
# dw_start = 8
# traj = np.load('pos_enu_dw_neuromhe.npy')
# traj_ref = np.load('ref_enu_dw_neuromhe.npy')
# traj_big = np.load('pos_big_dw.npy')
# play_animation_exp2_pos(time, traj, traj_ref, traj_big, dw_start,dt)

# disf = np.load('Df_Imhe_dw_neuromhe.npy')
# disf_ref = np.load('Gt_lpf_dw_neuromhe.npy')
# play_animation_exp2_force(time, disf, disf_ref, dw_start, dt)

# ----DMHE-----#
# time = np.load('Time_dw_dmhe.npy')
# dw_start = 8.6
# traj = np.load('pos_enu_dw_dmhe.npy')
# traj_ref = np.load('ref_enu_dw_dmhe.npy')
# traj_big = np.load('pos_big_dw.npy')
# play_animation_exp2_pos(time, traj, traj_ref, traj_big, dw_start,dt)

# disf = np.load('Df_Imhe_dw_dmhe.npy')
# disf_ref = np.load('Gt_lpf_dw_dmhe.npy')
# play_animation_exp2_force(time, disf, disf_ref, dw_start, dt)

# ----L1-AC-----#
# time = np.load('Time_dw_l1.npy')
# dw_start = 6.75
# traj = np.load('pos_enu_dw_l1.npy')
# traj_ref = np.load('ref_enu_dw_l1.npy')
# traj_big = np.load('pos_big_dw.npy')
# play_animation_exp2_pos(time, traj, traj_ref, traj_big, dw_start,dt)

# disf = np.load('Df_Imhe_dw_l1.npy')
# disf_ref = np.load('Gt_lpf_dw_l1.npy')
# play_animation_exp2_force(time, disf, disf_ref, dw_start, dt)

# ----Baseline-----#
time = np.load('Time_dw_baseline.npy')
dw_start = 8.5
traj = np.load('pos_enu_dw_baseline_lg.npy')
traj_ref = np.load('ref_enu_dw_baseline.npy')
traj_big = np.load('pos_big_dw.npy')
play_animation_exp2_pos(time, traj, traj_ref, traj_big, dw_start,dt)



