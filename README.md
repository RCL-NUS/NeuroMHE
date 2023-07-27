# NeuroMHE
The ***Neural Moving Horizon Estimation (NeuroMHE)*** is an auto-tuning and adaptive optimal estimator. It fuses a neural network with an MHE to realize accurate estimation and fast online adaptation, leveraging the advantages from both advanced machine learning techniques and control-theoretic estimation algorithms. The neural network can be trained efficiently from the robot's trajectory tracking errors using reinforcement learning without the need for the ground truth data.

## Table of contents
1. [Project Overview](#project-Overview)
2. [Dependency Packages](#Dependency-Packages)
3. [How to Use](#How-to-Use)
4. [Contact Us](#Contact-Us)


## 1. Project Overview
The project consists of three folders, which correspond to the three experiments in the paper that show the following four features of our method.
1. NeuroMHE enjoys computationally efficient training and significantly improves the force estimation performance as compared to a state-of-the-art estimator [[1]](#1).
2. A stable NeuroMHE with a fast dynamic response can be trained directly from the trajectory tracking error using Algorithm 2 (i.e., the model-based policy gradient algorithm) .
3. NeuroMHE exhibits superior estimation and robust control performance than a fixed-weighting MHE (DMHE) [[2]](#2) and a state-of-the-art adaptive controller [[3]](#3) for handling dynamic noise covariances.
4. Finally, NeuroMHE is efficiently transferable to different challenging flight scenarios on a real quadrotor without extra parameter tuning, such as counteracting state-dependent cable forces and flying under the downwash flow.
* SecVII-A (source code): *A comprehensive comparison with the state-of-the-art NeuroBEM estimator* on its real agile test dataset to show the first feature. 
* SecVII-B (source code): *A robust trajectory tracking control scenario in simulation* to show the second and third features.
* SecVII-C (source code): *Physical experiments on a real quadrotor* to show the fourth feature.

## 2. Dependency Packages
Please make sure that the following packages have already been installed before running the source code.
* CasADi: version 3.5.5 Info: https://web.casadi.org/
* Numpy: version 1.23.0 Info: https://numpy.org/
* Pytorch: version 1.12.0+cu116 Info: https://pytorch.org/
* Matplotlib: version 3.3.0 Info: https://matplotlib.org/
* Python: version 3.9.12 Info: https://www.python.org/
* Scipy: version 1.8.1 Info: https://scipy.org/
* Pandas: version 1.4.2 Info: https://pandas.pydata.org/
* filterpy: version 1.4.5 Info: https://filterpy.readthedocs.io/en/latest/

## 3. How to Use
First and foremost, the training process for NeuroMHE is both efficient and straightforward to setup. The source code has been comprehensively annotated to facilitate ease of use. To reproduce the simulation results presented in the paper, simply follow the steps outlined below, sequentially, after downloading and decompressing all the necessary folders.

### SecVII-A
1. Download '**processed_data.zip**' and '**predictions.tat.xz**' from https://download.ifi.uzh.ch/rpg/NeuroBEM/. The former file is utilized for training NeuroMHE, whereas the latter serves the purpose of evaluation and comparison with NeuroBEM.
2. Relocate the folder '**bem+nn**' from the decomprassed archive '**predictions.tat.xz**' to the downloaded folder '**SecVII-A (source code)**', and place the decompressed '**processed_data.zip**' within the folder '**SecVII-A (source code)**' as well.
3. Run the Python file '**main_code_supervisedlearning.py**'. 
4. In the prompted terminal interface, you will be asked to select whether to train or evaluate NeuroMHE.
   * Training: type 'train' without the quotation mark in the terminal.
   * Evaluation: type 'evaluate' without the quotation mark in the terminal. Subsequently, you will be prompted to select a trajectory for evaluation. There are a total of 13 agile trajectories within the complete NeuroBEM test dataset, as shown below. _Note that you can skip the training process and directly evaluate the performance using the trained neural network model **Trained_model.pt** to reproduce the RMSE results in the following table_. The retained model is saved in the folder '**trained_data**' within the downloaded folder '**SecVII-A (source code)**'.

|                                         Trajectory Parameters of NeuroBEM Test Dataset                                           |
:----------------------------------------------------------------------------------------------------------------------------------:
![test dataset](https://github.com/RCL-NUS/NeuroMHE/assets/70559054/afbdc415-288b-4938-8bc9-7b18c59d6f40)

One advantage of NeuroBEM is that its accuracy only declines by 20% when the training dataset encompasses a limited portion of the velocity-range space compared to the test dataset. To assess the performance of our NeuroMHE in this scenario, we select two 10-second-long segments for training: 1) one from an agile figure-8 trajectory, covering a wide velocity range of 0.05 m/s to 16.38 m/s, referred to as the 'fast training set'; 2) the other from a relatively slow wobbly circle trajectory, with a limited velocity range of 0.19 m/s to 5.18 m/s, referred to as the 'slow training set'. The following figures present a comparison of the velocity-range space between the training sets and the partial test sets.
        Velocity-Range Space (World Frame): Training Sets        |        Velocity-Range Space (World Frame): Partial Test Sets
:---------------------------------------------------------------:|:--------------------------------------------------------------:
![3d_velocityspace_training](https://github.com/RCL-NUS/NeuroMHE/assets/70559054/7c0d344a-6ef5-4df5-8c5d-7be85084a09b) | ![3d_velocityspace_test](https://github.com/RCL-NUS/NeuroMHE/assets/70559054/421f9c2e-6583-44db-853d-502ca0475912)

We evaluate the performance of NeuroMHE, trained on the '**slow training set**', in comparison to NeuroBEM on its complete test dataset. The comparative results in terms of RMSE are summarized in the following table.

|                              Estimation Errors (RMSEs) Comparisons on the NeuroBEM Test Dataset                                | 
:--------------------------------------------------------------------------------------------------------------------------------:
![RMSE_slow_trainingset](https://github.com/RCL-NUS/NeuroMHE/assets/70559054/923667ae-d6fb-4683-9775-092c74a60434)

Notably, NeuroMHE demonstrates a significantly smaller RMSE in the overall force estimation than NeuroBEM across all of these trajectories, achieving a reduction of up to 62.7% (See the penultimate column). The only exception is the '3D Circle\_1' trajectory where both methods exhibit a similar RMSE value. Furthermore, NeuroMHE exhibits a comparable performance in the overall torque estimation to that of NeuroBEM. The torque estimation performance could potentially be improved by using inertia-normalized quadrotor dynamics, wherein the force and torque magnitudes are similar. These findings underscore the superior generalizability of NeuroMHE to previously unseen challenging trajectories. They also demonstrate the exceptional robustness of our approach with respect to the training dataset.

5. Check the RMSE results. The RMSEs can be computed under two conditions: A and B.
   * Under condition A, the force is expressed in the world frame (cond.A1), and the RMSEs of the planar and the overall disturbances are computed using the scalar error (cond.A2), e.g., $\Delta_{f}=\sqrt{d_{f_x}^2+d_{f_y}^2+d_{f_z}^2} - \sqrt{ \hat d_{f_x}^2 + \hat d_{f_y}^2 + \hat d_{f_z}^2}$. Additionally, a mass of 0.772 kg is used for obtaining the ground truth force $\mathbf d_{f}$ (cond.A3).
   * Under condition B, the force is expressed in the body frame (cond.B1), and the RMSEs of the planar and the overall disturbances are computed using the vector error (cond.B2), e.g., $\Delta_{f}=\sqrt{(d_{f_x}-\hat d_{f_x})^2 + (d_{f_y}-\hat d_{f_y})^2 + (d_{f_z}-\hat d_{f_z})^2}$. Furthermore, a mass of 0.752 kg is used for obtaining the ground truth force $\mathbf d_{f}$ (cond.B3).

   The use of different values for the quadrotor mass stems from two reasons. First, the mass was originally reported as 0.752 kg in [[1]](#1), but the authors subsequently updated it to be 0.772 kg (See https://rpg.ifi.uzh.ch/neuro_bem/Readme.html). Second, the residual forces (i.e., $\Delta_{f_x}$, $\Delta_{f_x}$, and $\Delta_{f_z}$) provided in columns 36-38 of the NeuroBEM dataset, considered equivalent to the force estimation error in our context (e.g., $\Delta_{f_x}=d_{f_x}-\hat d_{f_x}$), have been computed using the mass of 0.752 kg. As the provided residual forces of NeuroBEM are expressed in the body frame and convenient for computing the RMSEs using the vector error, we choose to present our RMSE comparison under condition B.
   
   Within the folder '**Check_RMSE**', run the Python file '**RMSE_Computation_Slow_Better_Cond.B.py**' to replicate the RMSE results in the above table and our paper.


### SecVII-B
1. Run the **main_code.py** Python file in the downloaded **SecVII-B (source code)** folder.
2. In the prompted terminal interface, you will be asked to select whether to train or evaluate NeuroMHE.
   * Training: type 'train' without the quotation mark in the terminal. Subsequently, you will be prompted to select whether to train NeuroMHE or DMHE.
   * Evaluation: type 'evaluate' without the quotation mark in the terminal. Subsequently, you will be prompted to select a controller for evaluation. There are 5 controllers, as shown below. Note again that, like the case in SecVII-A, you can also skip the training process and directly evaluate the performance using the trained network model **trained_nn_model.pt** that is saved in the **trained_data** folder within the downloaded **SecVII-B (source code)** folder.

     ![evaluation_mode](https://user-images.githubusercontent.com/70559054/227720537-e2910ce5-7128-4bed-864b-848c787a7413.png)
     
3. Run the **plotting.py** python file in the downloaded **SecVII-B (source code)** folder to plot figures and show the following animation demo.

     https://user-images.githubusercontent.com/70559054/227720768-c70ea330-114e-4058-aea6-619bbdc3f379.mp4
     
### SecVII-C
1. Run the **main_code.py** Python file in the **Training in simulation** folder within the downloaded **SecVII-C (source code)** folder.
2. In the prompted terminal interface, you will be asked to select whether to train or evaluate NeuroMHE.
   * Training: type 'train' without the quotation mark in the terminal. Subsequently, you will be prompted to select whether to train NeuroMHE or DMHE.
   * Evaluation: type 'evaluate' without the quotation mark in the terminal. Subsequently, you will be prompted to select a controller for evaluation. There are 4 controllers, as shown below.

     ![evaluation_mode](https://user-images.githubusercontent.com/70559054/227721390-17b9d275-23e2-4506-9bd3-c829363c981a.png)

3. Run the **T-RO-experiment-data-processing_22_Feb.py** Python file located in the **SecVII-C (source code)/Evaluation in real-world experiment/Experiment Raw Data** directory to generate the figures depicting the experimental results. A video demonstration showcasing the experiments is presented below.

      https://user-images.githubusercontent.com/70559054/227721900-338651dc-ce40-4288-ae8a-844ab37c50c1.mp4
      
Note that we modify the official v1.11.1 PX4 firmware to bypass the PX4's position and velocity controllers. The modified PX4 firmware is available at https://github.com/mamariomiamo/px4_modified/commit/d06d41265b8871c94f5fb110d99f8ec03d3c6907. The primary code implementing our method in the onboard computer, along with other state-of-the-art robust flight controllers, is located in the **offb_py_v1_neuromhe.py** Python file (lines 504-659). This file can be found in the **SecVII-C (source code)/Evaluation in real-world experiment/Code used in onboard computer** directory.

## 4. Contact Us
If you encounter a bug in your implementation of the code, please do not hesitate to inform me.
* Name: Mr. Bingheng Wang
* Email: wangbingheng@u.nus.edu

## References
<a id="1">[1]</a> 
L. Bauersfeld, E. Kaufmann, P. Foehn, S. Sun, and D. Scaramuzza, "NeuroBEM: Hybrid Aerodynamic Quadrotor Model", ROBOTICS: SCIENCE AND SYSTEM XVII,2021.

<a id="2">[2]</a>
B. Wang, Z. Ma, S. Lai, L. Zhao, and T. H. Lee, "Differentiable Moving Horizon Estimation for Robust Flight Control", in 2021 60th IEEE
Conference on Decision and Control (CDC). IEEE, 2021, pp. 3563–3568.

<a id="3">[3]</a> 
Z. Wu, S. Cheng, K. A. Ackerman, A. Gahlawat, A. Lakshmanan, P. Zhao, and N. Hovakimyan, "L1-Adaptive Augmentation for Geometrictracking Control of Quadrotors", in 2022 International Conference on
Robotics and Automation (ICRA). IEEE, 2022, pp. 1329–1336.

