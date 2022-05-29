"""
This is the main function that trains NeuroMHE and evaluates its performance.
----------------------------------------------------------------------------
Wang, Bingheng at Control and Simulation Lab, ECE Dept. NUS, Singapore
First version: 19 Dec. 2020
Second version: 31 Aug. 2021
Third version: 10 May 2022
Should you have any question, please feel free to contact the author via:
wangbingheng@u.nus.edu
"""
import UavEnv
import Robust_Flight
from casadi import *
import time as TM
import numpy as np
import uavNN
import torch
from numpy import linalg as LA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

"""---------------------------------Learn or Evaluate?-------------------------------------"""
train = False

"""---------------------------------Load environment---------------------------------------"""
# Use the same parameters as those in the paper 'NeuroMHE'
uav_para = np.array([0.752, 0.00252, 0.00214, 0.00436])
wing_len = 0.5
# Sampling time-step for MHE
dt_sample = 1e-2
uav = UavEnv.quadrotor(uav_para, dt_sample)
uav.model()
# Initial states
horizon = 10 # previous 30
# Learning rate
lr_nn  = 1e-4
# First element in R_t
r11    = np.array([[100]])


"""---------------------------------Define parameterization model-----------------------------"""
# Define neural network for process noise
D_in, D_h, D_out = 18, 50, 49
model_QR = uavNN.Net(D_in, D_h, D_out)

"""---------------------------------Define reference trajectory-----------------------------"""
"""
The reference trajectory is generated using minisnap algorithm [2]
[2] Mellinger, D. and Kumar, V., 2011, May. 
    Minimum snap trajectory generation and control for quadrotors. 
    In 2011 IEEE international conference on robotics and automation (pp. 2520-2525). IEEE.
"""
# get the reference trajectory from a text file
# obtain the reference trajectory from polynomials in the for loop below
# the vertical figure-8 trajectory consists of 6 segments
# total time = 5s
# coeffy1  = np.array([[0,	0,	0,	0,	11.4417520989313,	-15.9295299392789,	7.86739580287094,	-1.37961796252333]])
# coeffy2  = np.array([[2,	3.66640777983499,	-1.60582696834201,	-4.46700362782327,	1.51841075725908,	2.30286766486403,	-1.78992993485645,	0.375074329814742]])
# coeffy3  = np.array([[2,	-3.47233499236823,	-1.84008475598486,	1.96431889710088,	-0.688598397760292,	-0.560151018093061,	0.835590373757921,	-0.238740105664274]])
# coeffy4  = np.array([[0,	-4.22796597185740,	7.38436256142371e-10,	0.753476510099850,	-3.68732153512958e-09,	0.693234548373297,	3.86506213278999e-09,	-0.238740109212597]])
# coeffy5  = np.array([[-2,	-3.47233499278689,	1.84008475532051,	1.96431889249457,	0.688598403936287,	-0.560151013386798,	-0.835590378359499,	0.375074330691937]])
# coeffy6  = np.array([[-2,	3.66640778937219,	1.60582700220108,	-4.46700360030695,	-1.51841076417020,	2.30286766098126,	1.78992993648766,	-1.37961796160896]])
# coeffz1  = np.array([[1.50000000000000,	0,	0,	0,	10.7217406154092,	-17.7132819791282,	9.86438930909612,	-1.87284794537709]])
# coeffz2  = np.array([[2.50000000000000,	0.396952803234457,	-4.16634331809132,	-2.50774923009029,	4.57149226798069,	2.14324702242955,	-3.24554630861773,	0.807946762678237]])
# coeffz3  = np.array([[0.500000000000000,	-0.274427867585458,	5.45552020231979,	0.577900584442352,	-5.11733055541407,	-0.363148812916062,	2.41008102995497,	-0.688594578143343]])
# coeffz4  = np.array([[1.50000000000000,	3.31894365018143,	4.36339178122580e-11,	-6.04573062259981,	-4.75075742318495e-09,	3.25197274149599,	6.40134321283130e-09,	-0.688594584378591]])
# coeffz5  = np.array([[2.50000000000000,	-0.274427871312888,	-5.45552019933877,	0.577900587288830,	5.11733056634362,	-0.363148807223697,	-2.41008103881412,	0.807946764852731]])
# coeffz6  = np.array([[0.500000000000000,	0.396952806998248,	4.16634334490378,	-2.50774925099112,	-4.57149228213592,	2.14324702179172,	3.24554631515831,	-1.87284794720144]])
# t1, t2, t3, t4, t5, t6 = 1, 1, 0.5, 0.5, 1, 1 
# T_end  = t1+t2+t3+t4+t5+t6

# total time = 10s
# coeffy1  = np.array([[0,	0,	0,	0,	0.715109506079672,	-0.497797810499814,	0.122928059387922,	-0.0107782653289662]])
# coeffy2  = np.array([[2,	1.83320388967584,	-0.401456742074237,	-0.558375453537513,	0.0949006722458052,	0.0719646145217878,	-0.0279676552176590,	0.00293026819945179]])
# coeffy3  = np.array([[2,	-1.73616749603011,	-0.460021189549377,	0.245539862161277,	-0.0430373997492550,	-0.0175047193360391,	0.0130560995745220,	-0.00186515707013232]])
# coeffy4  = np.array([[0,	-2.11398298638844,	2.39931992496140e-10,	0.0941845638268584,	-2.66192078091648e-10,	0.0216635796381875,	8.34796642572638e-11,	-0.00186515711135814]])
# coeffy5  = np.array([[-2,	-1.73616749658771,	0.460021188468167,	0.245539862047749,	0.0430374002794756,	-0.0175047191994634,	-0.0130560996961020,	0.00293026821623757]])
# coeffy6  = np.array([[-2,	1.83320389357685,	0.401456745973048,	-0.558375453845794,	-0.0949006729348390,	0.0719646146110639,	0.0279676553313809,	-0.0107782653558311]])
# coeffz1  = np.array([[1.50000000000000,	0,	0,	0,	0.670108788178773,	-0.553540061568246,	0.154131082868448,	-0.0146316245645094]])
# coeffz2  = np.array([[2.50000000000000,	0.198476402091168,	-1.04158582812769,	-0.313468654476541,	0.285718266540056,	0.0669764694342854,	-0.0507116610348912,	0.00631208407790541]])
# coeffz3  = np.array([[0.500000000000000,	-0.137213932933007,	1.36388004867623,	0.0722375731695036,	-0.319833159396692,	-0.0113484004411801,	0.0376575160560593,	-0.00537964513269534]])
# coeffz4  = np.array([[1.50000000000000,	1.65947182444572,	4.66042308744311e-10,	-0.755716327404562,	-4.06039731007509e-10,	0.101624148108360,	1.27817699187126e-10,	-0.00537964519610551]])
# coeffz5  = np.array([[2.50000000000000,	-0.137213933661702,	-1.36388005015638,	0.0722375734165580,	0.319833160189239,	-0.0113484002422211,	-0.0376575162447230,	0.00631208410477019]])
# coeffz6  = np.array([[0.500000000000000,	0.198476407512242,	1.04158583289536,	-0.313468657305748,	-0.285718267580650,	0.0669764696213481,	0.0507116612220829,	-0.0146316246129538]])
# t1, t2, t3, t4, t5, t6 = 2, 2, 1, 1, 2, 2 
# T_end  = t1+t2+t3+t4+t5+t6


# total time = 10s, horizontal figure-8 trajectory plus vertical take-off and landing
# coeffx1  = np.array([[0,	0,	0,	0,	1.06826912897616,	-1.16606314505495,	0.411015878637760,	-0.0475257941771612]])
# coeffx2  = np.array([[0,	-1.47694657020543,	-0.940115323009447,	1.05172307188418,	0.761368027085691,	-0.226039312282283,	-0.254345239844083,	0.0843553455105173]])
# coeffx3  = np.array([[-1,	0.777683528661472,	2.47915259210529,	-0.297665645998010,	-1.23157003911794,	0.0193515043713358,	0.336142178729666,	-0.0830941182282784]])
# coeffx4  = np.array([[1,	1.44866338376594,	-2.31259333939162,	-1.21588132216178,	0.999026025693195,	0.291228093956367,	-0.245516648867554,	0.0350738068425496]])
# coeffx5  = np.array([[0,	-2.59950593481131,	-1.22039818961625e-09,	2.00975398277686,	1.95096352617693e-09,	-0.445321855553080,	-9.70901472976740e-10,	0.0350738074296164]])
# coeffx6  = np.array([[-1,	1.44866338593274,	2.31259334330072,	-1.21588132405017,	-0.999026030340684,	0.291228094639323,	0.245516651036888,	-0.0830941189244367]])
# coeffx7  = np.array([[1,	0.777683523690794,	-2.47915259763839,	-0.297665641325746,	1.23157004605303,	0.0193515034504862,	-0.336142181433758,	0.0843553463141267]])
# coeffx8  = np.array([[0,	-1.47694655767031,	0.940115341424897,	1.05172307213739,	-0.761368037205774,	-0.226039312556122,	0.254345242764681,	-0.0475257948476967]])
# coeffy1  = np.array([[0,	0,	0,	0,	-0.782624893981478,	0.810733777509484,	-0.268231386397896,	0.0292603605692616]])
# coeffy2  = np.array([[0,	1.42292094010470,	1.36313431227419,	-0.362867956302477,	-0.576269343366177,	0.0498274285520530,	0.141413661572694,	-0.0381590426273137]])
# coeffy3  = np.array([[2,	1.58601413545097,	-1.36494630305560,	-0.676964304909416,	0.458506231028127,	0.0969695028153479,	-0.125699636818008,	0.0261203750579922]])
# coeffy4  = np.array([[2,	-1.42725414185960,	-1.01207347885496,	0.526976038125919,	-0.0279276801355175,	-0.108700441873889,	0.0571429885875103,	-0.00816328400878536]])
# coeffy5  = np.array([[0,	-2.23997097136558,	6.64349499856974e-10,	0.185405730386598,	-9.99772657391966e-10,	0.0627285254659844,	5.25637701527205e-10,	-0.00816328433175478]])
# coeffy6  = np.array([[-2,	-1.42725414304948,	1.01207347716520,	0.526976039605448,	0.0279276826033827,	-0.108700442346802,	-0.0571429897967761,	0.0261203754571315]])
# coeffy7  = np.array([[-2,	1.58601413883544,	1.36494630608603,	-0.676964308583729,	-0.458506235082311,	0.0969695034709689,	0.125699638403682,	-0.0381590431068173]])
# coeffy8  = np.array([[0,	1.42292093179500,	-1.36313432418462,	-0.362867955319322,	0.576269349588797,	0.0498274286507904,	-0.141413663343615,	0.0292603609789006]])
# coeffz1  = np.array([[0,	0,	0,	0,	0.997946014011672,	-0.864780109108952,	0.260314866772836,	-0.0269869078606388]])
# coeffz2  = np.array([[1.50000000000000,	0.642183418424516,	-0.891338449445481,	-0.0699259706578055,	0.412702728312573,	-0.00790196812853129,	-0.117501843276629,	0.0317820844771638]])
# coeffz3  = np.array([[1.50000000000000,	-0.221506787443172,	0.200976452267505,	0.264201352543249,	-0.276961804778573,	-0.0454892537680311,	0.104972748063318,	-0.0261927069465229]])
# coeffz4  = np.array([[1.50000000000000,	0.0842442266518492,	-0.0985384813654745,	-0.115828186112940,	0.153438404202603,	0.0343003887346630,	-0.0783762005623302,	0.0207598483372669]])
# coeffz5  = np.array([[1.50000000000000,	1.12482393888045e-09,	0.0779250794957442,	-1.34324363898470e-09,	-0.124107968754548,	4.43413512522542e-10,	0.0669427377986694,	-0.0207598484245890]])
# coeffz6  = np.array([[1.50000000000000,	-0.0842442286648966,	-0.0985384825115138,	0.115828189110890,	0.153438405581804,	-0.0343003896808215,	-0.0783762011735810,	0.0261927072085220]])
# coeffz7  = np.array([[1.50000000000000,	0.221506790924126,	0.200976455068527,	-0.264201356627131,	-0.276961808127529,	0.0454892546562376,	0.104972749286273,	-0.0317820848716374]])
# coeffz8  = np.array([[1.50000000000000,	-0.642183426513067,	-0.891338460817979,	0.0699259724104923,	0.412702733940253,	0.00790196806993205,	-0.117501844815164,	0.0269869082227578]])
# t1, t2, t3, t4, t5, t6, t7, t8 = 2, 1, 1, 1, 1, 1, 1, 2 

# total time = 10s, hybrid figure-8 trajectory plus vertical take-off and landing
coeffx1  = np.array([[0,	0,	0,	0,	0.152331494824417,	-0.154526440092534,	0.0496135785266626,	-0.00521661609324991]])
coeffx2  = np.array([[0,	-0.298744305676510,	-0.304466499879255,	0.0544619069915462,	0.123229299388696,	0.00264075039447410,	-0.0234190467789508,	0.00422646989917254]])
coeffx3  = np.array([[-0.500000000000000,	0.155878219816802,	0.588938955667651,	0.0213465647263575,	-0.148106144602473,	-0.00842996788046106,	0.0209588871623920,	-0.00327210811005434]])
coeffx4  = np.array([[0.500000000000000,	0.548007685927910,	-0.529179367383790,	-0.222016352490083,	0.109513767524510,	0.0255929083812182,	-0.0133982479931277,	0.00127602361505477]])
coeffx5  = np.array([[0,	-0.920599169199545,	-1.87465445833347e-10,	0.332620386064971,	1.44009013322612e-10,	-0.0346992077455723,	-3.51495712187929e-11,	0.00127602362958622]])
coeffx6  = np.array([[-0.500000000000000,	0.548007686740054,	0.529179367998795,	-0.222016352740123,	-0.109513767889182,	0.0255929084357713,	0.0133982480755826,	-0.00327210812914464]])
coeffx7  = np.array([[0.500000000000000,	0.155878218582443,	-0.588938956560076,	0.0213465654289180,	0.148106145174711,	-0.00842996798575140,	-0.0209588872804396,	0.00422646992825456]])
coeffx8  = np.array([[0,	-0.298744302001831,	0.304466501713942,	0.0544619054270734,	-0.123229300158115,	0.00264075060013480,	0.0234190469661898,	-0.00521661614812914]])

coeffy1  = np.array([[0,	0,	0,	0,	-0.135875093835255,	0.132156292555301,	-0.0401727225111496,	0.00403167484615656]])
coeffy2  = np.array([[0,	0.317528010662059,	0.379333246313780,	0.0293532636159754,	-0.0958065620272365,	-0.0112556905013975,	0.0162707253351386,	-0.00255761180885225]])
coeffy3  = np.array([[1,	0.612769369567035,	-0.334146162230932,	-0.153642027074412,	0.0667948443524554,	0.0143336795467161,	-0.0105841986577606,	0.00145043322179023]])
coeffy4  = np.array([[1,	-0.528794302485980,	-0.212482013376665,	0.112200055920328,	-0.0115868394225880,	-0.0123911386435023,	0.00464535017098875,	-0.000442414299515846]])
coeffy5  = np.array([[0,	-0.702584854735949,	1.51235212797456e-10,	-0.000950747255528086,	-1.08290930538104e-10,	0.00851293724325797,	2.60248132219883e-11,	-0.000442414310317114]])
coeffy6  = np.array([[-1,	-0.528794303034177,	0.212482012948838,	0.112200056167577,	0.0115868396882949,	-0.0123911386850187,	-0.00464535023234762,	0.00145043323612226]])
coeffy7  = np.array([[-1,	0.612769370599190,	0.334146162861386,	-0.153642027653544,	-0.0667948447741247,	0.0143336796304759,	0.0105841987470028,	-0.00255761183106151]])
coeffy8  = np.array([[0,	0.317528007740410,	-0.379333247859878,	0.0293532645380037,	0.0958065626216174,	-0.0112556906639969,	-0.0162707254791093,	0.00403167488856731]])

coeffz1  = np.array([[0,	0,	0,	0,	0.463672959786425,	-0.358987437200239,	0.0977834803363151,	-0.00929150084140096]])
coeffz2  = np.array([[1,	0.730375584761979,	-0.366697225844408,	-0.207997427074548,	0.139187172370347,	0.0339282561577513,	-0.0322975314433912,	0.00509822836285142]])
coeffz3  = np.array([[1.25000000000000,	-0.100915138733599,	0.0818341346374224,	0.113770336192889,	-0.0941643672990477,	-0.0158582366881220,	0.0212338663665501,	-0.00359742592511163]])
coeffz4  = np.array([[1.25000000000000,	-0.0794640115391700,	-0.173864087497295,	-0.0121591194915870,	0.0785959100072364,	0.00526818564940799,	-0.0165391058471361,	0.00303880988479067]])
coeffz5  = np.array([[1,	8.73063754624936e-10,	0.238922789237003,	-6.01759211349349e-10,	-0.0811281023220900,	8.14129397610944e-11,	0.0153683979431775,	-0.00303880989221208]])
coeffz6  = np.array([[1.25000000000000,	0.0794640093639612,	-0.173864087980318,	0.0121591207488013,	0.0785959103531395,	-0.00526818583688078,	-0.0165391059250553,	0.00359742594914942]])
coeffz7  = np.array([[1.25000000000000,	0.100915141671314,	0.0818341359633964,	-0.113770338104422,	-0.0941643681507745,	0.0158582369348030,	0.0212338665410333,	-0.00509822840982423]])
coeffz8  = np.array([[1,	-0.730375591915733,	-0.366697229603716,	0.207997430287229,	0.139187173709609,	-0.0339282565598957,	-0.0322975317621738,	0.00929150093740556]])
t1, t2, t3, t4, t5, t6, t7, t8 = 2, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2 


# Total simulation time
T_end  = t1+t2+t3+t4+t5+t6+t7+t8
# Total iterations
N      = int(T_end/dt_sample)
"""---------------------------------Define controller---------------------------------------"""
ctrl_gain = np.array([25,25,25, 15,15,15, 10,10,10, 0.3,0.3,0.3])
GeoCtrl   = Robust_Flight.Controller(uav_para, ctrl_gain, uav.x)

"""---------------------------------Define MHE----------------------------------------------"""
uavMHE = Robust_Flight.MHE(horizon, dt_sample,r11)
uavMHE.SetStateVariable(uav.xa)
uavMHE.SetOutputVariable()
uavMHE.SetControlVariable(uav.u)
uavMHE.SetNoiseVariable(uav.eta)
uavMHE.SetModelDyn(uav.dymh)
uavMHE.SetCostDyn()


"""---------------------------------Define NeuroMHE(or DMHE)----------------------------------------------"""
uavNMHE = Robust_Flight.KF_gradient_solver(GeoCtrl.ref, uav.xa, r11)

"""---------------------------------Parameterization of Neural Network Output-----------------------------"""
def SetPara(tunable_para, r11):
    epsilon = 1e-4
    gmin, gmax = 1e-2, 1 # 1e-2
    p_diag = np.zeros((1, 24))
    for i in range(24):
        p_diag[0, i] = epsilon + tunable_para[0, i]**2
    P0 = np.diag(p_diag[0])
    gamma_r = gmin + (gmax - gmin) * 1/(1+np.exp(-tunable_para[0, 24]))
    gamma_q = gmin + (gmax - gmin) * 1/(1+np.exp(-tunable_para[0, 42]))
    r_diag = np.zeros((1, 17))
    for i in range(17):
        r_diag[0, i] = epsilon + tunable_para[0, i+25]**2
    r      = np.hstack((r11, r_diag))
    r      = np.reshape(r, (1, 18))
    R_t    = np.diag(r[0])
    q_diag = np.zeros((1, 6))
    for i in range(6):
        q_diag[0, i] = epsilon + tunable_para[0, i+43]**2
    Q_t1   = np.diag(q_diag[0])
    weight_para = np.hstack((p_diag, np.reshape(gamma_r, (1,1)), r_diag, np.reshape(gamma_q,(1,1)), q_diag))
    return P0, gamma_r, gamma_q, R_t, Q_t1, weight_para

def convert(parameter):
    tunable_para = np.zeros((1,D_out))
    for i in range(D_out):
        tunable_para[0,i] = parameter[i,0]
    return tunable_para

def chainRule_gradient(tunable_para):
    tunable = SX.sym('tunable', 1, D_out)
    P = SX.sym('P', 1, 24)
    epsilon = 1e-4
    gmin, gmax = 1e-2, 1
    for i in range(24):
        P[0, i] = epsilon + tunable[0, i]**2
    gamma_r = gmin + (gmax - gmin) * 1/(1+exp(-tunable[0, 24]))
    R = SX.sym('R', 1, 17)
    for i in range(5):
        R[0, i] = epsilon + tunable[0, i+10]**2
    gamma_q = gmin + (gmax - gmin) * 1/(1+exp(-tunable[0, 42]))
    Q = SX.sym('Q', 1, 6)
    for i in range(6):
        Q[0, i] = epsilon + tunable[0, i+43]**2
    weight = horzcat(P, gamma_r, R, gamma_q, Q)
    w_jaco = jacobian(weight, tunable)
    w_jaco_fn = Function('W_fn',[tunable],[w_jaco],['tp'],['W_fnf'])
    weight_grad = w_jaco_fn(tp=tunable_para)['W_fnf'].full()
    return weight_grad

def Train():
    # load pre-trained nn model
    # the pretrained nn model is the neural network trained one time 
    # using the same training data to better initialize the network
    # parameters, this is necessary as an appropritate initialization guarantees convergence
    """
    Such initialization stage could be removed if a Lipschitz network is used [3]
    [3] Zhou, S., Pereida, K., Zhao, W. and Schoellig, A.P., 2021. 
        Bridging the Model-Reality Gap With Lipschitz Network Adaptation. 
        IEEE Robotics and Automation Letters, 7(1), pp.642-649.
    """
    PATH0 = "pretrained_nn_model.pt"
    model_QR = torch.load(PATH0)
    # load training data
    dis_f = np.load('Dis_f_for_training.npy') # dataset 3
    dis_t = np.load('Dis_t_for_training.npy')
    # mean loss list
    Loss = []
    # tracking performance in training
    Position = []
    # system state in training
    Fullstate = []
    # reference trajectory in training
    Ref_P = []
    # disturbance estimate in training
    Dis_f_mh = []
    Dis_t_mh = []
    # covariance in training
    Cov_f = []
    Cov_t = []
    # weighting matrices for the process noise in training
    Q_tp  = []
    # training iteration
    k_train  = 0
    # training iteration list
    K_iteration = []
    # initial mean loss
    mean_loss0 = 0
    # initial change of the mean loss
    delta_loss = 1e3
    # threshold
    eps = 1e-1
    # polynomial coefficients for the process noise covariance
    dpara = np.array([5,5,0.5,5,5,0.5,20,20,0.5, 0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01])

    while delta_loss >= eps:
        #for ktrain in range(10):
        # initial time
        time   = 0
        # initial states
        x0     = np.random.normal(0,0.01)
        y0     = np.random.normal(0,0.01)
        p      = np.array([[x0,y0,0]]).T
        v      = np.zeros((3,1))
        roll   = np.random.normal(0,0.01)
        pitch  = np.random.normal(0,0.01)
        yaw    = np.random.normal(0,0.01)
        Euler  = np.array([roll,pitch,yaw])
        R_h, R_bw = uav.dir_cosine(Euler)
        w      = np.zeros((3,1))
        # initial guess of state and disturbance force
        pg0    = np.zeros((3,1))
        vg0    = np.zeros((3,1))
        df0    = np.zeros((3,1))
        dtau0  = np.zeros((3,1))
        # initial reference attitude and angular velocity
        Rb_hd  = np.array([[1,0,0,0,1,0,0,0,1]]).T
        wd     = np.zeros((3,1))
        # filter priori
        m      = np.vstack((pg0,vg0,df0,Rb_hd,w,dtau0))
        xmhe_traj = m
        # total thruster list
        ctrl   = []
        # measurement list
        Y      = []
        # reference list
        Ref    = []
        # sum of loss
        sum_loss = 0.0
        # record the position tracking performance for each episode in training
        position = np.zeros((3, N))
        # record the system state for each episode in training
        State = np.zeros((D_in, N))
        # record the disturbance esitmate for each episode in training
        dis_f_mh = np.zeros((3, N))
        # dis_t = np.zeros((3, N))
        # record the covariance inverse of the process noise for each episode in training
        cov_f = np.zeros((3, N))
        cov_t = np.zeros((3, N))
        # record the weighting matrix of the process noise for each episode in training
        q_tp  = np.zeros((7, N))
        # record the reference position trajectory
        Ref_p = np.zeros((3, N))

        for k in range(N):
            # get reference
            if time <t1:
                ref_px, ref_vx, ref_ax, ref_jx, ref_sx = uav.polytraj(coeffx1,time,0)
                ref_py, ref_vy, ref_ay, ref_jy, ref_sy = uav.polytraj(coeffy1,time,0)
                ref_pz, ref_vz, ref_az, ref_jz, ref_sz = uav.polytraj(coeffz1,time,0)
            elif time >=t1 and time <t1+t2:
                ref_px, ref_vx, ref_ax, ref_jx, ref_sx = uav.polytraj(coeffx2,time,t1)
                ref_py, ref_vy, ref_ay, ref_jy, ref_sy = uav.polytraj(coeffy2,time,t1)
                ref_pz, ref_vz, ref_az, ref_jz, ref_sz = uav.polytraj(coeffz2,time,t1)
            elif time >=t1+t2 and time <t1+t2+t3:
                ref_px, ref_vx, ref_ax, ref_jx, ref_sx = uav.polytraj(coeffx3,time,t1+t2)
                ref_py, ref_vy, ref_ay, ref_jy, ref_sy = uav.polytraj(coeffy3,time,t1+t2)
                ref_pz, ref_vz, ref_az, ref_jz, ref_sz = uav.polytraj(coeffz3,time,t1+t2)
            elif time >=t1+t2+t3 and time < t1+t2+t3+t4:
                ref_px, ref_vx, ref_ax, ref_jx, ref_sx = uav.polytraj(coeffx4,time,t1+t2+t3)
                ref_py, ref_vy, ref_ay, ref_jy, ref_sy = uav.polytraj(coeffy4,time,t1+t2+t3)
                ref_pz, ref_vz, ref_az, ref_jz, ref_sz = uav.polytraj(coeffz4,time,t1+t2+t3)
            elif time >=t1+t2+t3+t4 and time <t1+t2+t3+t4+t5:
                ref_px, ref_vx, ref_ax, ref_jx, ref_sx = uav.polytraj(coeffx5,time,t1+t2+t3+t4)
                ref_py, ref_vy, ref_ay, ref_jy, ref_sy = uav.polytraj(coeffy5,time,t1+t2+t3+t4)
                ref_pz, ref_vz, ref_az, ref_jz, ref_sz = uav.polytraj(coeffz5,time,t1+t2+t3+t4)
            elif time >=t1+t2+t3+t4+t5 and time < t1+t2+t3+t4+t5+t6:
                ref_px, ref_vx, ref_ax, ref_jx, ref_sx = uav.polytraj(coeffx6,time,t1+t2+t3+t4+t5)
                ref_py, ref_vy, ref_ay, ref_jy, ref_sy = uav.polytraj(coeffy6,time,t1+t2+t3+t4+t5)
                ref_pz, ref_vz, ref_az, ref_jz, ref_sz = uav.polytraj(coeffz6,time,t1+t2+t3+t4+t5)
            elif time >=t1+t2+t3+t4+t5+t6 and time < t1+t2+t3+t4+t5+t6+t7:
                ref_px, ref_vx, ref_ax, ref_jx, ref_sx = uav.polytraj(coeffx7,time,t1+t2+t3+t4+t5+t6)
                ref_py, ref_vy, ref_ay, ref_jy, ref_sy = uav.polytraj(coeffy7,time,t1+t2+t3+t4+t5+t6)
                ref_pz, ref_vz, ref_az, ref_jz, ref_sz = uav.polytraj(coeffz7,time,t1+t2+t3+t4+t5+t6)
            else:
                ref_px, ref_vx, ref_ax, ref_jx, ref_sx = uav.polytraj(coeffx8,time,t1+t2+t3+t4+t5+t6+t7)
                ref_py, ref_vy, ref_ay, ref_jy, ref_sy = uav.polytraj(coeffy8,time,t1+t2+t3+t4+t5+t6+t7)
                ref_pz, ref_vz, ref_az, ref_jz, ref_sz = uav.polytraj(coeffz8,time,t1+t2+t3+t4+t5+t6+t7)

            ref_p = np.reshape(np.vstack((ref_px, ref_py, ref_pz)), (3,1))
            ref_v = np.reshape(np.vstack((ref_vx, ref_vy, ref_vz)), (3,1))
            ref_a = np.reshape(np.vstack((ref_ax, ref_ay, ref_az)), (3,1))
            ref_j = np.reshape(np.vstack((ref_jx, ref_jy, ref_jz)), (3,1))
            ref_s = np.reshape(np.vstack((ref_sx, ref_sy, ref_sz)), (3,1))
                
            b1_c  = np.array([[1, 0, 0]]).T # constant desired heading direction
            ref   = np.vstack((ref_p, ref_v, Rb_hd, wd))
            Ref_p[:, k:k+1] = ref_p
            Ref  += [ref] 
            # obtain the noisy measurement (position and velocity)
            state_m = np.vstack((p, v, R_h, w)) + np.reshape(np.random.normal(0,1e-3,18),(18,1)) # set the standard deviation of measurement noise to be 1e-3 since the accuracy of OptiTrack can be 0.2 mm
            Y      += [state_m]
            position[:,k:k+1] = p
            # generate the parameterized neural network output
            nn_output    = convert(model_QR(Y[-1]))
            P0, gamma_r, gamma_q, R_t, Q_t1, weight_para = SetPara(nn_output,r11)
            q_tp[:,k:k+1] = np.reshape(weight_para[0,42:49],(7,1))
            print('learning_iteration:', k_train, 'sample=', k, 'gamma1=', gamma_r,'gamma2=', gamma_q,'r1=', R_t[0, 0], 'r2=', R_t[1, 1], 'r3=', R_t[2,2], 'q1=', Q_t1[0,0], 'q2=', Q_t1[1,1], 'q3=', Q_t1[2,2])
            opt_sol      = uavMHE.MHEsolver(Y, m, xmhe_traj, ctrl, weight_para, k)
            xmhe_traj    = opt_sol['state_traj_opt']
            costate_traj = opt_sol['costate_traj_opt']
            noise_traj   = opt_sol['noise_traj_opt']
            if time>(horizon*dt_sample):
                # update m based on xmhe_traj
                for ix in range(len(m)):
                    m[ix,0] = xmhe_traj[1, ix]
            # else: # full-information estimation
            #     for ix in range(len(m)):
            #         m[ix,0] = xmhe_traj[0, ix]
            # obtain the coefficient matricres to establish the auxiliary MHE system
            auxSys       = uavMHE.GetAuxSys_general(xmhe_traj, costate_traj, noise_traj, weight_para, Y, ctrl)
            matF, matG, matH = auxSys['matF'], auxSys['matG'], auxSys['matH']
            matddLxx, matddLxw = auxSys['matddLxx'], auxSys['matddLxw']
            matddLxe, matddLee, matddLew = auxSys['matddLxe'], auxSys['matddLee'], auxSys['matddLew']
            # solve for the analytical gradient using a Kalman filter (Algorithm 1, Lemma 2)
            if k <= horizon:
                M = np.zeros((len(m), D_out))
            else:
                M = X_opt[1]
                # print('learning_iteration:', k_train, 'sample=', k,'matddLee=',matddLee[-1])
                # print('learning_iteration:', k_train, 'sample=', k,'matF=',matF[-1])
                # print('learning_iteration:', k_train, 'sample=', k,'matH=',matH[-1])
                # print('learning_iteration:', k_train, 'sample=', k,'matG=',matG[-1])
            gra_opt = uavNMHE.GradientSolver_general(M, matF, matG, matddLxx, matddLxw, matddLxe, matddLee, matddLew, weight_para)
            # gra_opt = uavNMHE.AuxMHESolver(matA, matB, matD, matE, matF, matH, matG, weight_para)
            X_opt   = gra_opt['state_gra_traj']
            # geometric tracking control
            df_Imh   = np.transpose(xmhe_traj[-1, 6:9])
            df_Imh   = np.reshape(df_Imh, (3, 1)) # MHE disturbance estimate
            dtau_mh  = np.transpose(xmhe_traj[-1, 21:24])
            dtau_mh  = np.reshape(dtau_mh, (3, 1))
            # dtau_mh  = np.zeros((3,1))
            # df_Imh   = np.zeros((3,1))
            # df       = np.zeros((3,1))
            # dtau = np.zeros((3,1))
            dis_f_mh[:,k:k+1] = df_Imh
            # dis_t[:,k:k+1] = dtau
            df       = dis_f[:,k]
            dtau     = dis_t[:,k]
            print('learning_iteration:', k_train, 'sample=', k, 'df_Imh=', df_Imh.T, 'df_true=', df.T, 'dt_mh=',dtau_mh.T, 'dt_true=',dtau.T)
            state_mh = np.reshape(np.hstack((xmhe_traj[-1, 0:6], xmhe_traj[-1, 9:21])), (18,1))
            # if k_train >2:
            feedback = state_mh # state_m for pre-training, state_mh for training
            # else:
                # feedback = state_m
            u, Rb_hd, wd  = GeoCtrl.geometric_ctrl(feedback,ref_p,ref_v,ref_a,ref_j,ref_s,b1_c,df_Imh, dtau_mh)
            ctrl    += [u]
            # update the system state based on the system dynamics model
            state    = np.vstack((p, v, R_h, w)) # current true state
            State[:,k:k+1] = state
            output   = uav.step(state, u, df, dtau, dt_sample)
            p        = output['p_new']
            v        = output['v_new']
            R_h      = output['R_new']
            w        = output['w_new']
            Euler    = output['Euler']
            print('sample=',k,'ref_p=',ref_p.T,'act_p=',p.T,'Attitude=',57.3*Euler.T, 'mean_loss0=',mean_loss0)
            # update the ground truth data of the disturbance force
            # df, vf_inv, dtau, vt_inv   = uav.dis(np.vstack((p, v, R_h, w)),Euler, df, dtau, dpara,dt_sample)
            # cov_f[:,k:k+1] = vf_inv
            # cov_t[:,k:k+1] = vt_inv
            # update the MHE tunable parameters using the analytical gradient (Algorithm 2)
            dldw, loss_track = uavNMHE.ChainRule(Ref, xmhe_traj,X_opt)
            # print('learning_iteration:', k_train, 'sample=', k,'dldw=',dldw)
            if k_train >0: # train the neural network from the second learning epoch
                weight_grad = chainRule_gradient(nn_output)
                dldp        = np.matmul(dldw, weight_grad)
                # print('learning_iteration:', k_train, 'sample=', k,'dldp=',dldp)
                loss_nn_p   = model_QR.myloss(model_QR(Y[-1]), dldp)
                optimizer_p = torch.optim.Adam(model_QR.parameters(), lr=lr_nn)
                model_QR.zero_grad()
                loss_nn_p.backward()
                optimizer_p.step()
            loss_track = np.reshape(loss_track,(1))
            sum_loss += loss_track
            # update time
            time += dt_sample
        mean_loss    = sum_loss/N
        Position    += [position]
        Dis_f_mh    += [dis_f_mh]
        Fullstate   += [State]
        Ref_P       += [Ref_p]
        # Dis_f       += [dis_f]
        # Dis_t       += [dis_t]
        Cov_f       += [cov_f]
        Cov_t       += [cov_t]
        Q_tp        += [q_tp]
        K_iteration += [k_train]
        if k_train == 0:
            if mean_loss/25 > 2: # 25 for pre-training, 50 for training
                eps = 2
            else:
                eps = mean_loss/25
        if k_train >1:
            delta_loss = abs(mean_loss - mean_loss0)
        mean_loss0 = mean_loss
        Loss += [mean_loss]
        print('learning_iteration:',k_train,'mean_loss=',mean_loss, 'eps=', eps)
        
        # if k_train==0:
        #     PATH1 = "pretrained_nn_model.pt"
        #     torch.save(model_QR,PATH1)
        # else:
        PATH2 = "trained_nn_model.pt"
        torch.save(model_QR, PATH2)
        k_train += 1
        np.save('Loss', Loss)
        np.save('K_iteration',K_iteration)
        np.save('Position_train',Position)
        np.save('Dis_f_mh_train',Dis_f_mh)
        # np.save('Dis_t_train',Dis_t)
        # np.save('Cov_f_train',Cov_f)
        # np.save('Cov_t_train',Cov_t)
        np.save('Q_tp_train',Q_tp)   
        FULLSTATE = np.zeros((D_in, k_train*N))
        REFP      = np.zeros((3, k_train*N))    
        for i in range(k_train):
            FULLSTATE[:,i*N:(i+1)*N] = Fullstate[i]
            REFP[:,i*N:(i+1)*N] = Ref_P[i]
        np.save('FULLSTATE',FULLSTATE)
        np.save('REFP',REFP)
    
    # uav.play_animation(wing_len, FULLSTATE, REFP, dt_sample)


"""---------------------------------Evaluation process-----------------------------"""
def Evaluate():
    # load the trained neural network model
    PATH1 = "trained_nn_model.pt"
    model_QR = torch.load(PATH1)
    # initial time
    time = 0
    # polynomial coefficients for the process noise covariance, slightly changed
    dpara = 1*np.array([10,10,1,10,10,1,30,30,1, 0.01,0.02,0.01,0.01,0.02,0.01,0.01,0.02,0.01])
    # load test data of the disturbance
    # dis_f = np.load('Dis_f_for_test2.npy') # dataset 2
    # dis_t = np.load('Dis_t_for_test2.npy')

    # initial states
    x0     = np.random.normal(0,0.01)
    y0     = np.random.normal(0,0.01)
    p      = np.array([[x0,y0,0]]).T
    v      = np.zeros((3,1))
    roll   = np.random.normal(0,0.01)
    pitch  = np.random.normal(0,0.01)
    yaw    = np.random.normal(0,0.01)
    Euler  = np.array([roll,pitch,yaw])
    R_h, R_bw = uav.dir_cosine(Euler)
    w      = np.zeros((3,1))
    # initial guess of state and disturbance force
    pg0    = np.zeros((3,1))
    vg0    = np.zeros((3,1))
    df     = np.zeros((3,1))
    dtau   = np.zeros((3,1))
    # initial reference attitude and angular velocity
    Rb_hd  = np.array([[1,0,0,0,1,0,0,0,1]]).T
    wd     = np.zeros((3,1))
    # filter priori
    m      = np.vstack((pg0,vg0,df,Rb_hd,w,dtau))
    xmhe_traj = m
    # total thruster list
    ctrl   = []
    # measurement list (position and velocity)
    Y      = []
    # reference list
    Ref    = []
    # sum of loss
    sum_loss = 0.0
    # record the position tracking performance  
    position = np.zeros((3, N))
    # record the position tracking error
    p_error  = np.zeros((3, N))
    # record the disturbance  
    dis_f = np.zeros((3, N))
    dis_t = np.zeros((3, N))
    # record the disturbance estimates 
    df_MH = np.zeros((3, N))
    dtau_MH = np.zeros((3, N))
    # record the covariance inverse of the process noise 
    cov_f = np.zeros((3, N))
    cov_t = np.zeros((3, N))
    # record the weighting matrix 
    tp    = np.zeros((D_out, N))
    # record the reference position trajectory
    Ref_p = np.zeros((3, N))
    # record the reference velocity trajectory
    Ref_v = np.zeros((3, N))
    # record the reference acceleration trajectory
    Ref_a = np.zeros((3, N))
    # record the reference jerk trajectory
    Ref_j = np.zeros((3, N))
    # record the reference snap trajectory
    Ref_s = np.zeros((3, N))
    # record the total thrust
    F_t   = np.zeros((1, N))
    # record time
    Time  = np.zeros(N)

    for k in range(N):
        Time[k] = time
        # get reference
        ref_px, ref_vx, ref_ax, ref_jx, ref_sx = np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
        if time <t1:
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = uav.polytraj(coeffx1,time,0)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = uav.polytraj(coeffy1,time,0)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = uav.polytraj(coeffz1,time,0)
        elif time >=t1 and time <t1+t2:
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = uav.polytraj(coeffx2,time,t1)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = uav.polytraj(coeffy2,time,t1)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = uav.polytraj(coeffz2,time,t1)
        elif time >=t1+t2 and time <t1+t2+t3:
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = uav.polytraj(coeffx3,time,t1+t2)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = uav.polytraj(coeffy3,time,t1+t2)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = uav.polytraj(coeffz3,time,t1+t2)
        elif time >=t1+t2+t3 and time < t1+t2+t3+t4:
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = uav.polytraj(coeffx4,time,t1+t2+t3)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = uav.polytraj(coeffy4,time,t1+t2+t3)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = uav.polytraj(coeffz4,time,t1+t2+t3)
        elif time >=t1+t2+t3+t4 and time <t1+t2+t3+t4+t5:
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = uav.polytraj(coeffx5,time,t1+t2+t3+t4)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = uav.polytraj(coeffy5,time,t1+t2+t3+t4)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = uav.polytraj(coeffz5,time,t1+t2+t3+t4)
        elif time >=t1+t2+t3+t4+t5 and time < t1+t2+t3+t4+t5+t6:
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = uav.polytraj(coeffx6,time,t1+t2+t3+t4+t5)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = uav.polytraj(coeffy6,time,t1+t2+t3+t4+t5)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = uav.polytraj(coeffz6,time,t1+t2+t3+t4+t5)
        elif time >=t1+t2+t3+t4+t5+t6 and time < t1+t2+t3+t4+t5+t6+t7:
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = uav.polytraj(coeffx7,time,t1+t2+t3+t4+t5+t6)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = uav.polytraj(coeffy7,time,t1+t2+t3+t4+t5+t6)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = uav.polytraj(coeffz7,time,t1+t2+t3+t4+t5+t6)
        else:
            ref_px, ref_vx, ref_ax, ref_jx, ref_sx = uav.polytraj(coeffx8,time,t1+t2+t3+t4+t5+t6+t7)
            ref_py, ref_vy, ref_ay, ref_jy, ref_sy = uav.polytraj(coeffy8,time,t1+t2+t3+t4+t5+t6+t7)
            ref_pz, ref_vz, ref_az, ref_jz, ref_sz = uav.polytraj(coeffz8,time,t1+t2+t3+t4+t5+t6+t7)

        ref_p = np.reshape(np.vstack((ref_px, ref_py, ref_pz)), (3,1))
        ref_v = np.reshape(np.vstack((ref_vx, ref_vy, ref_vz)), (3,1))
        ref_a = np.reshape(np.vstack((ref_ax, ref_ay, ref_az)), (3,1))
        ref_j = np.reshape(np.vstack((ref_jx, ref_jy, ref_jz)), (3,1))
        ref_s = np.reshape(np.vstack((ref_sx, ref_sy, ref_sz)), (3,1))
        b1_c  = np.array([[1, 0, 0]]).T # constant desired heading direction
        ref   = np.vstack((ref_p, ref_v, Rb_hd, wd))
        Ref  += [ref] 
        Ref_p[:,k:k+1] = ref_p
        Ref_v[:,k:k+1] = ref_v
        Ref_a[:,k:k+1] = ref_a
        Ref_j[:,k:k+1] = ref_j
        Ref_s[:,k:k+1] = ref_s
        # obtain the noisy measurement (position and velocity)
        state_m = np.vstack((p, v, R_h, w)) + np.reshape(np.random.normal(0,1e-3,18),(18,1)) # set the standard deviation of measurement noise to be 1e-3 since the accuracy of OptiTrack can be 0.2 mm
        Y      += [state_m]
        position[:,k:k+1] = p
        p_error[:,k:k+1] = p - ref_p
        # generate the parameterized neural network output
        nn_output    = convert(model_QR(Y[-1]))
        P0, gamma_r, gamma_q, R_t, Q_t1, weight_para = SetPara(nn_output,r11)
        tp[:,k:k+1]  = np.reshape(weight_para,(D_out,1))
        print('sample=', k, 'gamma1=', gamma_r,'gamma2=', gamma_q,'r1=', R_t[0, 0], 'r2=', R_t[1, 1], 'r3=', R_t[2,2], 'q1=', Q_t1[0,0], 'q2=', Q_t1[1,1], 'q3=', Q_t1[2,2])
        opt_sol      = uavMHE.MHEsolver(Y, m, xmhe_traj, ctrl, weight_para, k)
        xmhe_traj    = opt_sol['state_traj_opt']
        costate_traj = opt_sol['costate_traj_opt']
        noise_traj   = opt_sol['noise_traj_opt']
        if time>(horizon*dt_sample):
            # update m based on xmhe_traj
            for ix in range(len(m)):
                m[ix,0] = xmhe_traj[1, ix]
        else: # full-information estimation
            for ix in range(len(m)):
                m[ix,0] = xmhe_traj[0, ix]
        # obtain the coefficient matricres to establish the auxiliary MHE system
        auxSys       = uavMHE.GetAuxSys_general(xmhe_traj, costate_traj, noise_traj, weight_para, Y, ctrl)
        matF, matG, matH = auxSys['matF'], auxSys['matG'], auxSys['matH']
        matddLxx, matddLxw = auxSys['matddLxx'], auxSys['matddLxw']
        matddLxe, matddLee, matddLew = auxSys['matddLxe'], auxSys['matddLee'], auxSys['matddLew']
        # solve for the analytical gradient using a Kalman filter (Algorithm 1, Lemma 2)
        if k <= 1:
            M = np.zeros((len(m), D_out))
        else:
            M = X_opt[1]
        gra_opt = uavNMHE.GradientSolver_general(M, matF, matG, matddLxx, matddLxw, matddLxe, matddLee, matddLew, weight_para)
        # gra_opt = uavNMHE.AuxMHESolver(matA, matB, matD, matE, matF, matH, matG, weight_para)
        X_opt   = gra_opt['state_gra_traj']
        # geometric tracking control
        df_Imh   = np.transpose(xmhe_traj[-1, 6:9])
        df_Imh   = np.reshape(df_Imh, (3, 1)) # MHE disturbance estimate
        dtau_mh  = np.transpose(xmhe_traj[-1, 21:24])
        dtau_mh  = np.reshape(dtau_mh, (3, 1))
        # df_Imh   = df  
        # dtau_mh  = dtau
        df_MH[:,k:k+1] = df_Imh
        dtau_MH[:,k:k+1] = dtau_mh
        dis_f[:,k:k+1] = df
        dis_t[:,k:k+1] = dtau
        # df       = dis_f[:,k]
        # dtau     = dis_t[:,k]
        print('sample=', k, 'df_Imh=', df_Imh.T, 'df_true=', df.T, 'dtau_mh=', dtau_mh.T, 'dtau_true=',dtau.T)
        state_mh = np.reshape(np.hstack((xmhe_traj[-1, 0:6], xmhe_traj[-1, 9:21])), (18,1))
        feedback = state_mh
        u, Rb_hd, wd = GeoCtrl.geometric_ctrl(feedback,ref_p,ref_v,ref_a,ref_j,ref_s,b1_c,df_Imh,dtau_mh)
        ctrl    += [u]
        F_t[:,k:k+1] = u[0,0]
        # update the system state based on the system dynamics model
        state    = np.vstack((p, v, R_h, w)) # current true state
        output   = uav.step(state, u, df, dtau, dt_sample)
        p        = output['p_new']
        v        = output['v_new']
        R_h      = output['R_new']
        w        = output['w_new']
        Euler    = output['Euler']
        print('sample=',k,'ref_p=',ref_p.T,'act_p=',p.T,'Attitude=',Euler.T,'f=',u[0,0])
        # update the ground truth data of the disturbance force
        df, vf_inv, dtau, vt_inv   = uav.dis(np.vstack((p, v, R_h, w)),Euler, df, dtau, dpara,dt_sample)
        cov_f[:,k:k+1] = vf_inv
        cov_t[:,k:k+1] = vt_inv
        #compute the loss
        dldw, loss_track = uavNMHE.ChainRule(Ref, xmhe_traj,X_opt)
        loss_track = np.reshape(loss_track,(1))
        sum_loss += loss_track
        #update time
        time += dt_sample
    mean_loss    = sum_loss/N
    print('mean_loss=',mean_loss)
    np.save('Time_evaluation',Time)
    np.save('Position_evaluation',position)
    np.save('Error_evaluation',p_error)
    # np.save('Dist_f_evaluation',dis_f)
    np.save('Dist_f_MHE_evaluation',df_MH)
    # np.save('Dist_t_evaluation',dis_t)
    # np.save('Dist_t_MHE_evaluation',dtau_MH)
    # np.save('Cov_inv_f_training',cov_f)
    # np.save('Cov_inv_t_training',cov_t)
    np.save('Tunable_para_evaluation',tp)
    np.save('Reference_position',Ref_p)
    # np.save('Dis_f_for_training',dis_f)
    # np.save('Dis_t_for_training',dis_t)

    # compute RMSE of estimaton error and tracking error
    rmse_fx = mean_squared_error(df_MH[0,:], dis_f[0,:], squared=False)
    rmse_fy = mean_squared_error(df_MH[1,:], dis_f[1,:], squared=False)
    rmse_fz = mean_squared_error(df_MH[2,:], dis_f[2,:], squared=False)
    # rmse_tx = mean_squared_error(dtau_MH[0,:], dis_t[0,:], squared=False)
    # rmse_ty = mean_squared_error(dtau_MH[1,:], dis_t[1,:], squared=False)
    # rmse_tz = mean_squared_error(dtau_MH[2,:], dis_t[2,:], squared=False)
    rmse_px = mean_squared_error(position[0,:], Ref_p[0,:], squared=False)
    rmse_py = mean_squared_error(position[1,:], Ref_p[1,:], squared=False)
    rmse_pz = mean_squared_error(position[2,:], Ref_p[2,:], squared=False)
    rmse    = np.vstack((rmse_fx,rmse_fy,rmse_fz, rmse_px,rmse_py,rmse_pz))
    np.save('RMSE_evaluation',rmse)
    print('rmse_fx=',rmse_fx,'rmse_fy=',rmse_fy,'rmse_fz=',rmse_fz)
    # print('rmse_tx=',rmse_tx,'rmse_ty=',rmse_ty,'rmse_tz=',rmse_tz)
    print('rmse_px=',rmse_px,'rmse_py=',rmse_py,'rmse_pz=',rmse_pz)
    """
    Plot figures
    """
    # loss function
    # iteration = np.load('K_iteration.npy')
    # loss      = np.load('Loss.npy')
    # plt.figure(1)
    # plt.plot(iteration, loss, linewidth=1.5, marker='o')
    # plt.xlabel('Training episodes')
    # plt.ylabel('Mean loss')
    # plt.grid()
    # plt.savefig('./mean_loss_train.png')
    # plt.show()

    # # disturbance
    # plt.figure(2)
    # plt.plot(Time, dis_f[0,:], linewidth=1, linestyle='--')
    # plt.plot(Time, df_MH[0,:], linewidth=1)
    # plt.xlabel('Time [s]')
    # plt.ylabel('Disturbance force in x axis')
    # plt.legend(['Ground truth', 'MHE estimation'])
    # plt.grid()
    # plt.savefig('./dfx_evaluation.png')
    # plt.show()

    # plt.figure(3)
    # plt.plot(Time, dis_f[1,:], linewidth=1, linestyle='--')
    # plt.plot(Time, df_MH[1,:], linewidth=1)
    # plt.xlabel('Time [s]')
    # plt.ylabel('Disturbance force in y axis')
    # plt.legend(['Ground truth', 'MHE estimation'])
    # plt.grid()
    # plt.savefig('./dfy_evaluation.png')
    # plt.show()

    # plt.figure(4)
    # plt.plot(Time, dis_f[2,:], linewidth=1, linestyle='--')
    # plt.plot(Time, df_MH[2,:], linewidth=1)
    # plt.xlabel('Time [s]')
    # plt.ylabel('Disturbance force in z axis')
    # plt.legend(['Ground truth', 'MHE estimation'])
    # plt.grid()
    # plt.savefig('./dfz_evaluation.png')
    # plt.show()

    # plt.figure(5)
    # plt.plot(Time, dis_t[0,:], linewidth=1, linestyle='--')
    # plt.plot(Time, dtau_MH[0,:], linewidth=1)
    # plt.xlabel('Time [s]')
    # plt.ylabel('Disturbance torque in x axis')
    # plt.legend(['Ground truth', 'MHE estimation'])
    # plt.grid()
    # plt.savefig('./dtx_evaluation.png')
    # plt.show()

    # plt.figure(6)
    # plt.plot(Time, dis_t[1,:], linewidth=1, linestyle='--')
    # plt.plot(Time, dtau_MH[1,:], linewidth=1)
    # plt.xlabel('Time [s]')
    # plt.ylabel('Disturbance torque in y axis')
    # plt.legend(['Ground truth', 'MHE estimation'])
    # plt.grid()
    # plt.savefig('./dty_evaluation.png')
    # plt.show()

    # plt.figure(7)
    # plt.plot(Time, dis_t[2,:], linewidth=1, linestyle='--')
    # plt.plot(Time, dtau_MH[2,:], linewidth=1)
    # plt.xlabel('Time [s]')
    # plt.ylabel('Disturbance torque in z axis')
    # plt.legend(['Ground truth', 'MHE estimation'])
    # plt.grid()
    # plt.savefig('./dtz_evaluation.png')
    # plt.show()

    # # tracking error
    # plt.figure(8)
    # plt.plot(Time, p_error[0,:], linewidth=1)
    # plt.plot(Time, p_error[1,:], linewidth=1)
    # plt.plot(Time, p_error[2,:], linewidth=1)
    # plt.xlabel('Time [s]')
    # plt.ylabel('Tracking errors')
    # plt.legend(['x direction','y direction','z direction'])
    # plt.grid()
    # plt.savefig('./tracking_error.png')
    # plt.show()

    # # covariance inverse
    # plt.figure(9)
    # plt.plot(Time, cov_f[0,:], linewidth=1)
    # plt.plot(Time, cov_f[1,:], linewidth=1)
    # plt.plot(Time, cov_f[2,:], linewidth=1)
    # plt.xlabel('Time [s]')
    # plt.ylabel('Covariance inverse')
    # plt.legend(['x direction','y direction','z direction'])
    # plt.grid()
    # plt.savefig('./covariance_inverse.png')
    # plt.show()

    # # tunable parameters
    # plt.figure(10)
    # plt.plot(Time, tp[43,:], linewidth=1)
    # plt.plot(Time, tp[44,:], linewidth=1)
    # plt.plot(Time, tp[45,:], linewidth=1)
    # plt.xlabel('Time [s]')
    # plt.ylabel('Parameters in Q_t1')
    # plt.legend(['q1','q2','q3'])
    # plt.grid()
    # plt.savefig('./Q_t1.png')
    # plt.show()

    # plt.figure(11)
    # plt.plot(Time, tp[42,:], linewidth=1)
    # plt.xlabel('Time [s]')
    # plt.ylabel('Forgetting factor 2')
    # plt.grid()
    # plt.savefig('./forgetting_factor2.png')
    # plt.show()

    # # Trajectory
    # plt.figure(12)
    # ax = plt.axes(projection="3d")
    # ax.plot3D(position[0,:], position[1,:], position[2,:], linewidth=1.5)
    # ax.plot3D(Ref_p[0,:], Ref_p[1,:], Ref_p[2,:], linewidth=1, linestyle='--')
    # plt.legend(['Actual', 'Desired'])
    # plt.xlabel('x [m]')
    # plt.ylabel('y [m]')
    # # plt.zlabel('z [m]')
    # plt.grid()
    # plt.savefig('./tracking_3D.png')
    # plt.show()

    # # Control force
    # plt.figure(13)
    # plt.plot(Time, F_t[0,:], linewidth=1)
    # plt.xlabel('Time [s]')
    # plt.ylabel('Total thrust')
    # plt.grid()
    # plt.savefig('./total_f.png')
    # plt.show()

    # plt.figure(10)
    # plt.plot(Time, Ref_v[2,:], linewidth=1)
    # plt.xlabel('Time [s]')
    # plt.ylabel('reference velocity in z')
    # plt.grid()
    # plt.savefig('./ref_v_z.png')
    # plt.show()

    # plt.figure(11)
    # plt.plot(Time, Ref_a[2,:], linewidth=1)
    # plt.xlabel('Time [s]')
    # plt.ylabel('reference acceleration in z')
    # plt.grid()
    # plt.savefig('./ref_a_z.png')
    # plt.show()

    # plt.figure(12)
    # plt.plot(Time, Ref_j[2,:], linewidth=1)
    # plt.xlabel('Time [s]')
    # plt.ylabel('reference jerk in z')
    # plt.grid()
    # plt.savefig('./ref_j_z.png')
    # plt.show()

    # plt.figure(13)
    # plt.plot(Time, Ref_s[2,:], linewidth=1)
    # plt.xlabel('Time [s]')
    # plt.ylabel('reference snap in z')
    # plt.grid()
    # plt.savefig('./ref_s_z.png')
    # plt.show()
    return rmse


    

"""---------------------------------Main function-----------------------------"""
if train:
    Train()
    Evaluate()
else:
    # Evaluate()
    # Rmse_fx, Rmse_fy, Rmse_fz, Rmse_px, Rmse_py, Rmse_pz = [], [], [], [], [], []
    # for i in range(100):
    #     rmse = Evaluate()
    #     print('No.',i)
    #     Rmse_fx += [rmse[0,0]]
    #     Rmse_fy += [rmse[1,0]]
    #     Rmse_fz += [rmse[2,0]]
    #     Rmse_px += [rmse[3,0]]
    #     Rmse_py += [rmse[4,0]]
    #     Rmse_pz += [rmse[5,0]]
    #     np.save('Rmse_fx',Rmse_fx)
    #     np.save('Rmse_fy',Rmse_fy)
    #     np.save('Rmse_fz',Rmse_fz)
    #     np.save('Rmse_px',Rmse_px)
    #     np.save('Rmse_py',Rmse_py)
    #     np.save('Rmse_pz',Rmse_pz)
    Rmse_fx = np.load('Rmse_fx_neuroMHE.npy')
    Rmse_fy = np.load('Rmse_fy_neuroMHE.npy')
    Rmse_fz = np.load('Rmse_fz_neuroMHE.npy')
    Rmse_px = np.load('Rmse_px_neuroMHE.npy')
    Rmse_py = np.load('Rmse_py_neuroMHE.npy')
    Rmse_pz = np.load('Rmse_pz_neuroMHE.npy')
    RMSE_fx = np.zeros((len(Rmse_fx),1))
    RMSE_fy = np.zeros((len(Rmse_fy),1))
    RMSE_fz = np.zeros((len(Rmse_fz),1))
    RMSE_px = np.zeros((len(Rmse_px),1))
    RMSE_py = np.zeros((len(Rmse_py),1))
    RMSE_pz = np.zeros((len(Rmse_pz),1))
    print('len of rmse=',len(Rmse_fx))
    for i in range(len(Rmse_fx)):
        RMSE_fx[i,0] = Rmse_fx[i]
        RMSE_fy[i,0] = Rmse_fy[i]
        RMSE_fz[i,0] = Rmse_fz[i]
        RMSE_px[i,0] = Rmse_px[i]
        RMSE_py[i,0] = Rmse_py[i]
        RMSE_pz[i,0] = Rmse_pz[i]
    
    plt.figure(1)
    box_data = np.hstack((RMSE_fx,RMSE_fy,RMSE_fz))
    df = pd.DataFrame(data=box_data,columns = ['fx','fy','fz'])
    df.boxplot()
    plt.savefig('./rmse_force_box.png',dpi=600)
    plt.show()  
    plt.figure(2)
    box_data = np.hstack((RMSE_px,RMSE_py,RMSE_pz))
    df = pd.DataFrame(data=box_data,columns = ['px','py','pz'])
    df.boxplot()
    plt.savefig('./rmse_position_box.png',dpi=600)
    plt.show()
 



