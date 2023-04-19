#!/usr/bin/env python3
import rospy
from enum import Enum
from std_msgs.msg import Int64, Header, Byte
from std_srvs.srv import SetBool
import math
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import Altitude, ExtendedState, HomePosition, State, \
                            WaypointList, PositionTarget, RCOut
from mavros_msgs.srv import CommandBool, ParamGet, SetMode, WaypointClear, \
                            WaypointPush
from pymavlink import mavutil
from sensor_msgs.msg import NavSatFix, Imu, BatteryState
from six.moves import xrange # six.moves
from threading import Thread
"""----Packages needed for Bingheng's code---"""
import UavEnv
import Robust_Flight
from casadi import *
import time
import numpy as np
import uavNN
import torch
from numpy import linalg as LA
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF

class uavTaskType(Enum):
    Idle = 0
    TakeOff = 1
    Mission = 2
    Land = 3

class TaskManager:
    def __init__(self):
        self.altitude = Altitude()
        self.extened_state = ExtendedState()
        self.global_position = NavSatFix()
        self.battery_state = BatteryState()
        self.imu_data = Imu()
        self.home_position = HomePosition()
        self.local_position = PoseStamped()
        self.state = State()
        self.local_velocity = TwistStamped()  # local_velocity initialize
        self.local_velocity_body = TwistStamped()

        self.pos = PoseStamped()
        self.position = PositionTarget() # thrust control commands
        self.pwm_out = RCOut()

        self.task_state = uavTaskType.Idle

        # ROS publisher
        self.pos_control_pub = rospy.Publisher('mavros/setpoint_raw/local', PositionTarget, queue_size = 10)
        self.position_pub = rospy.Publisher('mavros/setpoint_position/local', PoseStamped, queue_size = 1)

        # ROS subscribers

        self.local_pos_sub = rospy.Subscriber('mavros/local_position/pose', PoseStamped, self.local_position_callback)
        self.local_vel_sub = rospy.Subscriber('mavros/local_position/velocity_local', TwistStamped, self.local_velocity_callback) #vehicle velocity in ENU world frame
        self.local_vel_sub = rospy.Subscriber('mavros/local_position/velocity_body', TwistStamped, self.local_velocity_body_callback) # velocity in body frame
        self.state_sub = rospy.Subscriber('mavros/state', State, self.state_callback)
        self.cmd_sub = rospy.Subscriber('user/cmd', Byte, self.cmd_callback)
        self.pwm_sub = rospy.Subscriber('mavros/rc/out', RCOut, self.pwm_callback)
        self.battery_sub = rospy.Subscriber('mavros/battery', BatteryState, self.battery_callback)

        # send setpoints in seperate thread to better prevent failsafe
        self.pos_thread = Thread(target=self.send_pos_ctrl, args=())
        self.pos_thread.daemon = True
        self.pos_thread.start()

        # ROS services
        service_timeout = 30
        rospy.loginfo("Waiting for ROS services")
        try:
            rospy.wait_for_service('mavros/param/get',service_timeout)
            rospy.wait_for_service('mavros/cmd/arming',service_timeout)
            rospy.wait_for_service('mavros/mission/push',service_timeout)
            rospy.wait_for_service('mavros/mission/clear',service_timeout)
            rospy.wait_for_service('mavros/set_mode',service_timeout)
            rospy.loginfo("ROS services are up")
        except rospy.ROSException:
            rospy.logerr("failed to connect to services")
        self.get_param_srv = rospy.ServiceProxy('mavros/param/get', ParamGet)
        self.set_arming_srv = rospy.ServiceProxy('mavros/cmd/arming', CommandBool)
        self.set_mode_srv = rospy.ServiceProxy('mavros/set_mode', SetMode)

    def send_pos(self):
        rate = rospy.Rate(10)
        self.pos.header = Header()
        self.pos.header.frame_id = "base_footprint"

        while not rospy.is_shutdown():
            self.pos.header.stamp = rospy.Time.now()
            self.position_pub.publish(self.pos)
            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def send_pos_ctrl(self):
        rate = rospy.Rate(100)
        self.pos.header = Header()
        self.pos.header.frame_id = "base_footprint"
        # self.position.coordinate_frame = 1 # this is required for the new version of PX4 code!

        while not rospy.is_shutdown():
            self.pos.header.stamp = rospy.Time.now()
            self.pos_control_pub.publish(self.position)
            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def pwm_callback(self, data):
        self.pwm_out = data

    def battery_callback(self, data):
        self.battery_state = data
        #rospy.loginfo("voltage is {0}".format(self.battery_state.voltage))


    def cmd_callback(self, data):
        #self.task_state = data
        cmd = data.data
        rospy.loginfo("Command received: {0}".format(self.task_state))
        rospy.loginfo("Command received: {0}".format(data))
        if cmd == 1:
            rospy.loginfo("Taks state changed to {0}".format(self.task_state))
            self.task_state = uavTaskType.TakeOff
        elif cmd == 2:
            rospy.loginfo("Taks state changed to {0}".format(self.task_state))
            self.task_state = uavTaskType.Mission
        elif cmd == 3:
            rospy.loginfo("Taks state changed to {0}".format(self.task_state))
            self.task_state = uavTaskType.Land

    def local_velocity_callback(self, data): # local_velocity callback
        self.local_velocity = data

    def local_velocity_body_callback(self, data): # local_velocity callback
        self.local_velocity_body = data

    def local_position_callback(self, data):
        self.local_position = data

    def state_callback(self, data):
        if self.state.armed != data.armed:
            rospy.loginfo("armed state changed from {0} to {1}".format(self.state.armed, data.armed))

        if self.state.connected != data.connected:
            rospy.loginfo("connected changed from {0} to {1}".format(self.state.connected, data.connected))

        if self.state.mode != data.mode:
            rospy.loginfo("mode changed from {0} to {1}".format(self.state.mode, data.mode))

        if self.state.system_status != data.system_status:
            rospy.loginfo("system_status changed from {0} to {1}".format(
                mavutil.mavlink.enums['MAV_STATE'][
                    self.state.system_status].name, mavutil.mavlink.enums[
                        'MAV_STATE'][data.system_status].name))

        self.state = data

    #
    # Helper methods
    #

    def set_arm(self, arm, timeout):
        """arm: True to arm or False to disarm, timeout(int): seconds"""
        rospy.loginfo("setting FCU arm: {0}".format(arm))
        old_arm = self.state.armed
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        arm_set = False
        for i in xrange(timeout * loop_freq): # xrange
            if self.state.armed == arm:
                arm_set = True
                rospy.loginfo("set arm success | seconds: {0} of {1}".format(
                    i / loop_freq, timeout))
                break
            else:
                try:
                    res = self.set_arming_srv(arm)
                    if not res.success:
                        rospy.logerr("failed to send arm command")
                except rospy.ServiceException as e:
                    rospy.logerr(e)

            try:
                rate.sleep()
            except rospy.ROSException as e:
                rospy.logerr("fail to arm")

    def set_mode(self, mode, timeout):
        """mode: PX4 mode string, timeout(int): seconds"""
        rospy.loginfo("setting FCU mode: {0}".format(mode))
        old_mode = self.state.mode
        loop_freq = 1  # Hz
        rate = rospy.Rate(loop_freq)
        mode_set = False
        for i in xrange(timeout * loop_freq):
            if self.state.mode == mode:
                mode_set = True
                rospy.loginfo("set mode success | seconds: {0} of {1}".format(
                    i / loop_freq, timeout))
                break
            else:
                try:
                    res = self.set_mode_srv(0, mode)  # 0 is custom mode
                    if not res.mode_sent:
                        rospy.logerr("failed to send mode command")
                except rospy.ServiceException as e:
                    rospy.logerr(e)

            try:
                rate.sleep()
            except rospy.ROSException as e:
                rospy.logerr("fail to set mode")

"""------------Load system parameters for the proposed controller-------------"""
# Use the same parameters as those in the paper 'NeuroMHE'
uav_para = np.array([1.0, 0.029, 0.029, 0.055]) # need to be updated
wing_len = 0.5
# Sampling time-step for MHE
frequency = 25
dt_mhe = 1/frequency
uav = UavEnv.quadrotor(uav_para, dt_mhe)
uav.model()
# Initial states
horizon = 10
# First element in R_t
r11    = np.array([[100]])
# Define neural network for process noise
D_in, D_h, D_out = 3, 20, 14


"""------------Define parameterization model----------------------------------"""
def SetPara(epsilon, gmin,tunable_para, r11):
    p_diag = np.zeros((1, 6))
    for i in range(6):
        p_diag[0, i] = epsilon + tunable_para[0, i]**2
    P0 = np.diag(p_diag[0])

    gamma_r = gmin + (1 - gmin) * 1/(1+np.exp(-tunable_para[0, 6]))
    gamma_q = gmin + (1 - gmin) * 1/(1+np.exp(-tunable_para[0, 10]))

    r_diag = np.zeros((1, 3))
    for i in range(3):
        r_diag[0, i] = epsilon + tunable_para[0, i+7]**2
    R_t    = np.diag(r_diag[0])

    q_diag = np.zeros((1, 3))
    for i in range(3):
        q_diag[0, i] = epsilon + tunable_para[0, i+11]**2
    Q_t1   = np.diag(q_diag[0])

    weight_para = np.hstack((p_diag, np.reshape(gamma_r, (1,1)), r_diag, np.reshape(gamma_q,(1,1)), q_diag))
    return P0, gamma_r, gamma_q, R_t, Q_t1, weight_para

def convert(parameter):
    tunable_para = np.zeros((1,D_out))
    for i in range(D_out):
        tunable_para[0,i] = parameter[i,0]
    return tunable_para

"""------------Define reference trajectory-----------------------------------"""
# Load the reference trajectory polynomials
Coeffx_exp1, Coeffy_exp1, Coeffz_exp1 = np.zeros((12,8)), np.zeros((12,8)), np.zeros((12,8))
Coeffx_exp1_lf, Coeffy_exp1_lf, Coeffz_exp1_lf = np.zeros((12,8)), np.zeros((12,8)), np.zeros((12,8))
Coeffx_exp2, Coeffy_exp2, Coeffz_exp2 = np.zeros((3,8)), np.zeros((3,8)), np.zeros((3,8))
Coeffx_exp2_lo, Coeffy_exp2_lo, Coeffz_exp2_lo = np.zeros((7,8)), np.zeros((7,8)), np.zeros((7,8))

for i in range(12):
    Coeffx_exp1[i,:] = np.load('Ref_exp1/coeffx'+str(i+1)+'.npy')
    Coeffy_exp1[i,:] = np.load('Ref_exp1/coeffy'+str(i+1)+'.npy')
    Coeffz_exp1[i,:] = np.load('Ref_exp1/coeffz'+str(i+1)+'.npy')

for i in range(12):
    Coeffx_exp1_lf[i,:] = np.load('Ref_exp1_littlefast/coeffx'+str(i+1)+'.npy')
    Coeffy_exp1_lf[i,:] = np.load('Ref_exp1_littlefast/coeffy'+str(i+1)+'.npy')
    Coeffz_exp1_lf[i,:] = np.load('Ref_exp1_littlefast/coeffz'+str(i+1)+'.npy')
for i in range(3):
    Coeffx_exp2[i,:] = np.load('Ref_exp2/coeffx'+str(i+1)+'.npy')
    Coeffy_exp2[i,:] = np.load('Ref_exp2/coeffy'+str(i+1)+'.npy')
    Coeffz_exp2[i,:] = np.load('Ref_exp2/coeffz'+str(i+1)+'.npy')
for i in range(7):
    Coeffx_exp2_lo[i,:] = np.load('Ref_exp2_linearoscillation/coeffx'+str(i+1)+'.npy')
    Coeffy_exp2_lo[i,:] = np.load('Ref_exp2_linearoscillation/coeffy'+str(i+1)+'.npy')
    Coeffz_exp2_lo[i,:] = np.load('Ref_exp2_linearoscillation/coeffz'+str(i+1)+'.npy')


"""------------Quaternion to Rotation Matrix---------------------------------"""
def Quaternion2Rotation(quaternion):
    # convert a point from body frame to inertial frame
    quaternion = quaternion/LA.norm(quaternion)
    q0, q1, q2, q3 = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix from body to inertial frame
    R_b = np.array([[r00, r01, r02],
                    [r10, r11, r12],
                    [r20, r21, r22]])
    R_bh = np.array([[r00, r01, r02, r10, r11, r12, r20, r21, r22]])
    return R_b, R_bh 


# Controller
# ctrl_gain = 3*np.array([[4,4,4,3.6,3.6,3.6]]) # large gain for baseline,3 times for downwash, 4 times for cable (which causes a crash!)
ctrl_gain = np.array([[4,4,4,3.6,3.6,3.6]]) # tune this gain based on the normalization and the control performance, same for NeuroMHE, DMHE, and L1-AC
GeoCtrl   = Robust_Flight.Controller(uav_para, ctrl_gain, dt_mhe)

"""-------------------Define MHE----------------------------------------------"""
uavMHE = Robust_Flight.MHE(horizon, dt_mhe,r11)
uavMHE.SetStateVariable(uav.xp)
uavMHE.SetOutputVariable(uav.y)
uavMHE.SetControlVariable(uav.f)
uavMHE.SetNoiseVariable(uav.wf)
uavMHE.SetRotationVariable(uav.R_B)
uavMHE.SetModelDyn(uav.dymh,uav.dyukf)
uavMHE.SetCostDyn()

"""------------------Initialization-------------------------------------------"""
# Control force and torque list
print("===============================================")
print("Please choose which trajectory to evaluate")
print("'a': Trajectory for experiment 1 p0=[1,1,0.12]")
print("'b': little_fast Trajectory for experiment 1 p0=[1,1,0.12]")
print("'c': Trajectory for experiment 2 p0=[0,0,0.12]")
print("'d': Trajectory for experiment 2 p0=[-1.2,-1.2,0.1] linear oscillation")
traj       = input("enter 'a', or 'b',... without the quotation mark:")
print("Please choose which controller to evaluate")
print("'a': NeuroMHE               + PD Baseline Controller")
print("'b': DMHE                   + PD Baseline Controller")
print("'c': L1 Adaptive Control    + PD Baseline Controller")
print("'d': UKF                    + PD Baseline Controller")
print("'e': PD Baseline Controller alone")
controller = input("enter 'a', or 'b',... without the quotation mark:")
# print("Please choose thrust mode")
# print("'a': actual thrust")
# print("'b': calculated thrust command")
# thrust_mode = input("enter 'a', or 'b',... without the quotation mark:")
# print("===============================================")

mass = uav_para[0]
g = 9.78 # local gravity acceleration in Singapore
ctrl = []
# ctrl_f += [mass*g]
# Rotation list
R_B = []
# Force estimation list
Df_I = []
# Time list
Time = []
# Tunable parameters list
Gamma_r = []
Gamma_q = []

time = 0
k_time = 0
# initialization flag
flagi = 0
# Measurement list
Y_p = []
# Initial state prediction of L1-AC
z_hat  = np.zeros((3,1))
# Initial value of the low-pass filter used for L1-Adaptive control
sig_f_prev = 0
sig_fu_prev = np.zeros((2,1))
# Initial control
f      = uav_para[0]*9.78

if controller == 'a': 
    epsilon = np.load('trained_data/epsilon_trained.npy')
    gmin    = np.load('trained_data/gmin_trained.npy') 
    PATH1 = "trained_data/trained_nn_model.pt"
    model_QR = torch.load(PATH1)
elif controller == 'b':
    epsilon = np.load('trained_data/epsilon_trained_dmhe.npy')
    gmin    = np.load('trained_data/gmin_trained_dmhe.npy') 
    tunable_para = np.load('trained_data/tunable_para_trained.npy')



# Total trajectory time
if traj == 'a':
    T_end  = 33
elif traj == 'b':
    T_end  = 33
elif traj == 'c':
    T_end  = 24
else:
    T_end  = 28

"""---------------------------------Define model for UKF---------------------------"""
def DynUKF(x, dt, U):
    u = U[0]
    Rb = np.array([[U[1],U[2],U[3]],
                   [U[4],U[5],U[6]],
                   [U[7],U[8],U[9]]])
    x_next = uavMHE.MDyn_ukf_fn(s=x, c=u, R=Rb)['MDyn_ukff']
    x_next = np.reshape(x_next, (6))
    return x_next

def OutputUKF(x):
    uavMHE.SetArrivalCost(np.zeros((6,1)))
    uavMHE.diffKKT_general()
    H = uavMHE.H_fn(x0=x)['Hf']
    output = np.matmul(H, x)
    y = np.reshape(output, (3))
    return y

# UKF settings
sigmas = MerweScaledSigmaPoints(6, alpha=.1, beta=2., kappa=1)
ukf    = UKF(dim_x=6, dim_z=3, fx=DynUKF, hx=OutputUKF, dt=dt_mhe, points=sigmas)

# Covariance matrices for UKF
ukf.R = np.diag([0.01, 0.01, 0.01]) # measurement noise
ukf.Q = np.diag([0.01, 0.01, 0.01,0.2,0.2,0.2]) # process noise

factor = 0
# ref position
Ref_px = []
Ref_py = []
Ref_pz = []
px     = []
py     = []
pz     = []
weight = []
if __name__ == '__main__':
    rospy.init_node('Moving Horizon Estimator')
    uavTask = TaskManager()
    # initial time
    initial_time = rospy.get_time()
    uavTask.pos.pose.position.x = 0
    uavTask.pos.pose.position.y = 0
    uavTask.pos.pose.position.z = 0

    uavTask.set_mode("OFFBOARD", 5)
    uavTask.set_arm(True, 5)
    rate = rospy.Rate(frequency)
    while not rospy.is_shutdown():
        #rate = rospy.Rate(50) # the operating frequency at which the code runs
        print(uavTask.task_state)
        #uavTask.position_pub.publish(uavTask.pos)
        if uavTask.task_state == uavTaskType.TakeOff:
            #rospy.loginfo("Doing Takeoff")
            # current time
            current_time = rospy.get_time()
            # time = current_time-initial_time
            pwm = np.zeros(4)
            pwm[0] = uavTask.pwm_out.channels[0]
            pwm[1] = uavTask.pwm_out.channels[1]
            pwm[2] = uavTask.pwm_out.channels[2]
            pwm[3] = uavTask.pwm_out.channels[3]
            # rospy.loginfo("pwm 1 is {0}".format(pwm[0]))
            # rospy.loginfo("pwm 2 is {0}".format(pwm[1]))
            # rospy.loginfo("pwm 3 is {0}".format(pwm[2]))
            # rospy.loginfo("pwm 4 is {0}".format(pwm[3]))R_b
            if traj == 'a':
                if time <T_end:
                    ref_p, ref_v, ref_a, ref_j, ref_s = uav.ref_exp1(Coeffx_exp1, Coeffy_exp1, Coeffz_exp1, time, t_switch) 
                else:
                    ref_p, ref_v, ref_a, ref_j, ref_s = np.array([[1,-1,0.12]]).T, np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
                ref_pva = np.vstack((ref_p, ref_v))
            elif traj == 'b':
                if time <T_end:
                    ref_p, ref_v, ref_a, ref_j, ref_s = uav.ref_exp1(Coeffx_exp1_lf, Coeffy_exp1_lf, Coeffz_exp1_lf, time, t_switch) 
                else:
                    ref_p, ref_v, ref_a, ref_j, ref_s = np.array([[1,-1,0.12]]).T, np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
                ref_pva = np.vstack((ref_p, ref_v))
            elif traj == 'c':
                if time <T_end:
                    ref_p, ref_v, ref_a, ref_j, ref_s = uav.ref_exp2(Coeffx_exp2, Coeffy_exp2, Coeffz_exp2, time, t_switch) 
                else:
                    ref_p, ref_v, ref_a, ref_j, ref_s = np.array([[0,0,0.12]]).T, np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
                ref_pva = np.vstack((ref_p, ref_v))
            else:
                if time <T_end:
                    ref_p, ref_v, ref_a, ref_j, ref_s = uav.ref_exp2_linearoscillation(Coeffx_exp2_lo, Coeffy_exp2_lo, Coeffz_exp2_lo, time, t_switch) 
                else:
                    ref_p, ref_v, ref_a, ref_j, ref_s = np.array([[1.2,1.2,0.1]]).T, np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
                ref_pva = np.vstack((ref_p, ref_v))
            uavTask.position.position.x = ref_p[0,0]
            uavTask.position.position.y = ref_p[1,0]
            uavTask.position.position.z = ref_p[2,0]
            time += dt_mhe
            uavTask.position.type_mask = 3064 # flag for pid
            uavTask.pos_control_pub.publish(uavTask.position)

        elif uavTask.task_state == uavTaskType.Mission:
            #rospy.loginfo("Doing Mission")
            ## Controller will be used here ###
            
            # Quaternion (body frame:NWU, intertia frame: ENU)
            q0 = uavTask.local_position.pose.orientation.x
            q1 = uavTask.local_position.pose.orientation.y
            q2 = uavTask.local_position.pose.orientation.z
            q3 = uavTask.local_position.pose.orientation.w
            quaternion = np.array([q3, q0, q1, q2])
            R_b, R_bh = Quaternion2Rotation(quaternion) #rotation matrix from NWU body to ENU world
            x = uavTask.local_position.pose.position.x
            y = uavTask.local_position.pose.position.y
            z = uavTask.local_position.pose.position.z # ENU used in ROS
            vx_enu = uavTask.local_velocity.twist.linear.x 
            vy_enu = uavTask.local_velocity.twist.linear.y
            vz_enu = uavTask.local_velocity.twist.linear.z
            # inilialization of x_hatmh
            if flagi ==0:
                v_I0 = np.array([[vx_enu, vy_enu, vz_enu]]).T
                # Initial estimated force
                df_I0 = np.array([[0, 0, 0]]).T
                x_hatmh = np.vstack((v_I0, df_I0))
                xmhe_traj = x_hatmh
                noise_traj = np.zeros((1,3))
                # initial time
                initial_time = rospy.get_time()
                flagi = 1
            y_p = np.array([[x, y, z, vx_enu, vy_enu, vz_enu]]).T # used for feedback control in the baseline controller
            v   = np.array([[vx_enu, vy_enu, vz_enu]]).T
            nn_input = v 

            Y_p += [v]
            R_B += [R_b]
            #------------get reference trajectory------------#
            # current time
            current_time = rospy.get_time()
            time = current_time-initial_time
            t_switch = 0
            if traj == 'a':
                if time <T_end:
                    ref_p, ref_v, ref_a, ref_j, ref_s = uav.ref_exp1(Coeffx_exp1, Coeffy_exp1, Coeffz_exp1, time, t_switch) 
                else:
                    ref_p, ref_v, ref_a, ref_j, ref_s = np.array([[1,-1,0.12]]).T, np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
                ref_pva = np.vstack((ref_p, ref_v))
            elif traj == 'b':
                if time <T_end:
                    ref_p, ref_v, ref_a, ref_j, ref_s = uav.ref_exp1(Coeffx_exp1_lf, Coeffy_exp1_lf, Coeffz_exp1_lf, time, t_switch) 
                else:
                    ref_p, ref_v, ref_a, ref_j, ref_s = np.array([[1,-1,0.12]]).T, np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
                ref_pva = np.vstack((ref_p, ref_v))
            elif traj == 'c':
                if time <T_end:
                    ref_p, ref_v, ref_a, ref_j, ref_s = uav.ref_exp2(Coeffx_exp2, Coeffy_exp2, Coeffz_exp2, time, t_switch) 
                else:
                    ref_p, ref_v, ref_a, ref_j, ref_s = np.array([[0,0,0.12]]).T, np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
                ref_pva = np.vstack((ref_p, ref_v))
            else:
                if time <T_end:
                    ref_p, ref_v, ref_a, ref_j, ref_s = uav.ref_exp2_linearoscillation(Coeffx_exp2_lo, Coeffy_exp2_lo, Coeffz_exp2_lo, time, t_switch) 
                else:
                    ref_p, ref_v, ref_a, ref_j, ref_s = np.array([[1.2,1.2,0.1]]).T, np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1)), np.zeros((3,1))
                ref_pva = np.vstack((ref_p, ref_v))
            
            #-------------robust flight controller (including the baseline controller)-------------#
            if controller == 'a' or controller == 'b':
                if controller == 'a':
                    tunable_para    = convert(model_QR(nn_input))
                P0, gamma_r, gamma_q, R_t, Q_t1, weight_para = SetPara(epsilon, gmin,tunable_para,r11)
                print('gamma_r=',gamma_r,'gamma_q=',gamma_q)
                opt_sol      = uavMHE.MHEsolver(Y_p, R_B, x_hatmh, xmhe_traj, ctrl, noise_traj, weight_para, k_time)
                xmhe_traj    = opt_sol['state_traj_opt']
                costate_traj = opt_sol['costate_ipopt']
                noise_traj   = opt_sol['noise_traj_opt']
                if k_time>horizon:
                    # update m based on xmhe_traj
                    for ix in range(len(x_hatmh)):
                        x_hatmh[ix,0] = xmhe_traj[1, ix]
                df_I   = np.transpose(xmhe_traj[-1, 3:6])
                df_I   = np.reshape(df_I, (3, 1)) # MHE disturbance estimate
                Gamma_r += [gamma_r]
                Gamma_q += [gamma_q]
                weight  += [weight_para]
            elif controller == 'c':
                # Piecewise-constant adaptation law
                sig_hat_m, sig_hat_um, As = GeoCtrl.L1_adaptive_law(v,R_b,z_hat)
                z_hat = uav.predictor_L1(z_hat,R_b,v,f,sig_hat_m,sig_hat_um,As,dt_mhe)
                # Low-pass filter
                wf_coff = 5 
                time_constf = 1/wf_coff
                f_l1_lpf = GeoCtrl.lowpass_filter(time_constf,sig_hat_m,sig_f_prev) 
                sig_f_prev = f_l1_lpf
                fu_l1_lpf = GeoCtrl.lowpass_filter(time_constf,sig_hat_um,sig_fu_prev)
                sig_fu_prev = fu_l1_lpf
                u_ad = -f_l1_lpf
                df_B = np.vstack((fu_l1_lpf, f_l1_lpf)) # in body frame
                df_I = np.matmul(R_b,df_B) # transfer L1-AC force estimates from body frame to inertial frame
            elif controller == 'd':
                u_ukf = np.hstack((np.reshape(f,(1,1)),R_bh))
                u_ukf = np.reshape(u_ukf,(10))
                ukf.predict(U=u_ukf)
                y_m     = np.reshape(v,(3))
                ukf.update(z=y_m)
                Xukf = ukf.x.copy()
                df_I = np.reshape(Xukf[3:6],(3,1))
            else:
                df_I = np.zeros((3,1))
            Df_I += [df_I]
            # Position control
            if time > 1: # let the drone take off first so that there is no supporting force from the ground
                factor = 1
            Fd, fd     = GeoCtrl.position_ctrl(y_p,R_b, ref_p,ref_v,ref_a,factor*df_I) 

            f = fd # feed the control command directly to the MHE estimator
            #-----------control force normalization (required by the PX4 lower-level controller)------------#
            mass_for_normalization = 1.25 #1.1, larger this value, larger the control gain, unit: kg
            # force in inertial NED
            scale_factor = 4 # 4 motors
            fx = Fd[0, 0]/mass_for_normalization/(g*scale_factor)
            fy = Fd[1, 0]/mass_for_normalization/(g*scale_factor)
            fz = Fd[2, 0]/mass_for_normalization/(g*scale_factor)
            ctrl += [f]
            # send the normalized force to the PX4 lower-level controller, here our customized PX4 position controller is used.
            uavTask.position.position.x = fx
            uavTask.position.position.y = fy
            uavTask.position.position.z = fz
            rospy.loginfo("current dfx_mhe is {0}".format(df_I[0,0]))
            rospy.loginfo("current fx is {0}".format(fx))
            rospy.loginfo("current dfy_mhe is {0}".format(df_I[1,0]))
            rospy.loginfo("current fy is {0}".format(fy))
            rospy.loginfo("current dfz_mhe is {0}".format(df_I[2,0]))
            rospy.loginfo("current fz is {0}".format(fz))
            rospy.loginfo("current refx is {0}".format(ref_p[0,0]))
            rospy.loginfo("current x is {0}".format(x))
            rospy.loginfo("current refy is {0}".format(ref_p[1,0]))
            rospy.loginfo("current y is {0}".format(y))
            rospy.loginfo("current refz is {0}".format(ref_p[2,0]))
            rospy.loginfo("current z is {0}".format(z))
            rospy.loginfo("current time is {0}".format(time))
            uavTask.position.velocity.y = fd
            #----------save data (log in PX4)----------#
            if controller == 'a' or controller == 'b':
                uavTask.position.velocity.x = gamma_r
                uavTask.position.velocity.z = gamma_q
            uavTask.position.acceleration_or_force.x = df_I[0,0]
            uavTask.position.acceleration_or_force.y = df_I[1,0]
            uavTask.position.acceleration_or_force.z = df_I[2,0]
            uavTask.position.yaw = 0
            Time += [time]
            # time += dt_mhe # comment this line if rospy.get_time() is used above
            k_time += 1
            # 3576 to ask px4 only takes in position setpoints
            # 32768
            # uavTask.position.type_mask = 32768 # flag for proposed control mode
            uavTask.position.type_mask = 32768 # ask px4 to take acc_sp as thrust
            uavTask.pos_control_pub.publish(uavTask.position)

            
        elif uavTask.task_state == uavTaskType.Land:
            rospy.loginfo("Doing Land")
            uavTask.position.position.x = 0
            uavTask.position.position.y = 0
            uavTask.position.position.z = 0

            uavTask.position.yaw = 0

            uavTask.position.type_mask = 3064 # flag for pid
            uavTask.pos_control_pub.publish(uavTask.position)


        rate.sleep()
    rospy.spin()
