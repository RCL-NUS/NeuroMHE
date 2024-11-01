# NeuroMHE
The ***Neural Moving Horizon Estimation (NeuroMHE)*** is an auto-tuning and adaptive optimal estimator. It fuses a neural network with an MHE to realize accurate estimation and fast online adaptation, leveraging the advantages from both advanced machine learning techniques and control-theoretic estimation algorithms. The neural network can be trained efficiently from the robot's trajectory tracking errors using reinforcement learning without the need for the ground truth data.

|                     A Diagram of the NeuroMHE-based Robust Flight Control System and Its Learning Piplines             |
:----------------------------------------------------------------------------------------------------------------------------------:
![diagram_enlarged](https://github.com/RCL-NUS/NeuroMHE/assets/70559054/49a00744-cef7-47c2-b83b-2b91c448967f)

Please find out more details in 
   * our paper "Neural Moving Horizon Estimation for Robust Flight Control," [IEEE Xplore](https://ieeexplore.ieee.org/document/10313083), [arXiv](https://arxiv.org/abs/2206.10397)
   * YouTube video: https://www.youtube.com/watch?v=L5SrVr5ha-k

For learning estimation policy using the second-order trust-region method, kindly refer to our another respository https://github.com/BinghengNUS/TR-NeuroMHE

## Citation
If you find this work helpful in your publications, we would appreciate citing our paper (published in IEEE T-RO)

```
@ARTICLE{10313083,
  author={Wang, Bingheng and Ma, Zhengtian and Lai, Shupeng and Zhao, Lin},
  journal={IEEE Transactions on Robotics}, 
  title={Neural Moving Horizon Estimation for Robust Flight Control}, 
  year={2024},
  volume={40},
  number={},
  pages={639-659},
  doi={10.1109/TRO.2023.3331064}}
 ```


## Table of contents
1. [Project Overview](#project-Overview)
2. [Dependency Packages](#Dependency-Packages)
3. [How to Use](#How-to-Use)
      1. [SecVII-A: Accurate Estimation](#SecVII-A-Accurate-Estimation)
      2. [SecVII-B: Online Learning](#SecVII-B-Online-Learning)
      3. [SecVII-C: Real-world Experiments](#SecVII-C-Real-world-Experiments)
      4. [Applications to other robots](#Applications-to-other-robots)
4. [Acknowledgement](#Acknowledgement)
5. [Contact Us](#Contact-Us)


## 1. Project Overview
The project consists of three folders, which correspond to the three experiments in the paper that show the following four advantages of our method.
1. NeuroMHE enjoys computationally efficient training and significantly improves the force estimation performance as compared to a state-of-the-art estimator [[1]](#1).
2. A stable NeuroMHE with a fast dynamic response can be trained directly from the trajectory tracking error using Algorithm 2 (i.e., the model-based policy gradient algorithm) .
3. NeuroMHE exhibits superior estimation and robust control performance than a fixed-weighting MHE (DMHE) [[2]](#2) and a state-of-the-art adaptive controller [[3]](#3) for handling dynamic noise covariances.
4. Finally, NeuroMHE is efficiently transferable to different challenging flight scenarios on a real quadrotor without extra parameter tuning, such as counteracting state-dependent cable forces and flying under the downwash flow.
* SecVII-A (source code): *A comprehensive comparison with the state-of-the-art NeuroBEM estimator* on its real agile test dataset to show the first advantage. 
* SecVII-B (source code): *A robust trajectory tracking control scenario in simulation* to show the second and third advantages.
* SecVII-C (source code): *Physical experiments on a real quadrotor* to show the fourth advantage.

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
* scikit-learn: version 1.0.2 Info: https://scikit-learn.org/stable/whats_new/v1.0.html

## 3. How to Use
First and foremost, the training process for NeuroMHE is both efficient and straightforward to setup. The source code has been comprehensively annotated to facilitate ease of use. To reproduce the simulation results presented in the paper, simply follow the steps outlined below, sequentially, after downloading and decompressing all the necessary folders.

### SecVII-A: Accurate Estimation 
1. Download '**processed_data.zip**' and '**predictions.tat.xz**' from https://download.ifi.uzh.ch/rpg/NeuroBEM/. The former file is utilized for training NeuroMHE, whereas the latter serves the purpose of evaluation and comparison with NeuroBEM.
2. Relocate the folder '**bem+nn**' from the decomprassed archive '**predictions.tat.xz**' to the downloaded folder '**SecVII-A (source code)**', and place the decompressed '**processed_data.zip**' within the folder '**SecVII-A (source code)**' as well.
3. Run the Python file '**main_code_supervisedlearning.py**'. 
4. In the prompted terminal interface, you will be asked to select whether to train or evaluate NeuroMHE.
   * Training: type 'train' without the quotation mark in the terminal.
   * Evaluation: type 'evaluate' without the quotation mark in the terminal. Subsequently, you will be prompted to select a trajectory for evaluation. There are a total of 13 agile trajectories within the complete NeuroBEM test dataset, as shown below. Note that you can skip the training process and directly evaluate the performance using the trained neural network model '**Trained_model.pt**' to reproduce the RMSE results presented in our paper. The retained model is saved in the folder '**trained_data**'.

|                                         Trajectory Parameters of NeuroBEM Test Dataset                                           |
:----------------------------------------------------------------------------------------------------------------------------------:
![test dataset](https://github.com/RCL-NUS/NeuroMHE/assets/70559054/afbdc415-288b-4938-8bc9-7b18c59d6f40)

Please note that training performance may vary depending on the computer used. This variation arises from the gradient information's dependence on the MHE estimation trajectory, which is computed in the forward pass by a numerical solver. The solver's accuracy is influenced by its parameters and the computer's specifications. To uphold the training performance, we may slightly adjust the solver's parameters in the Python file '**Uav_mhe_supervisedlearning.py**'. Here are two training examples using different sets of solver parameters on different computers.
        Training on a workstation       |      Training on a laptop
:---------------------------------------------------------------:|:--------------------------------------------------------------:
![Mean_loss_train_reproduction_photo](https://github.com/RCL-NUS/NeuroMHE/assets/70559054/45c59927-40e7-4ec2-836f-5ad3fbdf72d7) | ![Mean_loss_train_reproduction_photo_mulaptop](https://github.com/RCL-NUS/NeuroMHE/assets/70559054/88e0a8dc-3239-4932-9f4e-3bc7e86798df)
opts['ipopt.tol']=1e-8 | opts['ipopt.tol']=1e-6
opts['ipopt.max_iter']=3e3 | opts['ipopt.max_iter']=1e3
opts['ipopt.acceptable_tol']=1e-7 | opts['ipopt.acceptable_tol']=1e-5


One advantage of NeuroBEM is that its accuracy only declines by 20% when the training dataset encompasses a limited portion of the velocity-range space compared to the test dataset. To assess the performance of our NeuroMHE in this scenario, we select two 10-second-long segments for training: 1) one from an agile figure-8 trajectory, covering a wide velocity range of 0.05 m/s to 16.38 m/s, referred to as the 'fast training set'; 2) the other from a relatively slow wobbly circle trajectory, with a limited velocity range of 0.19 m/s to 5.18 m/s, referred to as the 'slow training set'. The following figures present a comparison of the velocity-range space in the world frame between the training sets and the partial test sets.
        Velocity-Range Space: Training Sets        |      Velocity-Range Space: Partial Test Sets
:---------------------------------------------------------------:|:--------------------------------------------------------------:
![3d_velocityspace_training](https://github.com/RCL-NUS/NeuroMHE/assets/70559054/7c0d344a-6ef5-4df5-8c5d-7be85084a09b) | ![3d_velocityspace_test](https://github.com/RCL-NUS/NeuroMHE/assets/70559054/421f9c2e-6583-44db-853d-502ca0475912)

We evaluate the performance of NeuroMHE, trained on the '**slow training set**', in comparison to NeuroBEM on its complete test dataset. The comparative results in terms of RMSE are summarized in Table V of our paper.

5. Run the Python file '**plotting.py**' to reproduce the figures in Section VII-A of our paper.
6. Check the RMSE results. In the folder '**Check_RMSE**',
   * Run the MATLAB file '**RMSE_vector_error_slowtrainingset.m**' to replicate the RMSE results presented in Table V of our paper.
   * Additionally, we provide the RMSE results of NeuroMHE trained on the fast training set. To access these results, locate the MATLAB files '**RMSE_scalar_error_v1.m**' and '**RMSE_vector_error.m**' in the same folder and run them respectively.
   The file names themselves imply the distinction between these two datasets: the former is derived using scalar error (i.e., $e_{f}=\left | \sqrt{d_{f_x}^2+d_{f_y}^2+d_{f_z}^2} - \sqrt{ \hat d_{f_x}^2 + \hat d_{f_y}^2 + \hat d_{f_z}^2}\right |$) and expresses the force in the world frame, while the latter is attained through vector error (i.e., $e_{f}=\sqrt{(d_{f_x}-\hat d_{f_x})^2 + (d_{f_y}-\hat d_{f_y})^2 + (d_{f_z}-\hat d_{f_z})^2}$) and presents the force in the body frame. You can also run the corresponding Python files for the RMSE reproduction. These files have the same names as the MATLAB counterparts but end with '**.py**'.
   
   Note that the residual force data provided in the NeuroBEM dataset (columns 36-38 in the file 'predictions.tar.xz') was computed using the initially reported mass of 0.752 kg [[1]](#1) instead of the later revised value of 0.772 kg (See the NeuroBEM's website). As a result, we refrain from utilizing this data to compute NeuroBEM's RMSE. The rest of the dataset remains unaffected, as the residual force data is provided purely for users' convenience. It can be calculated from the other provided data including the mass value, as explained on https://rpg.ifi.uzh.ch/neuro_bem/Readme.html.

   To demonstrate the mass verification, in the subfolder '**MATLAB_code_for_mass_verification**', run the MATLAB file '**residual_force_XXX.m**' where '**XXX**' represents the name of the test trajectory, such as '**3D_Circle_1**'. Additionally, we provide the RMSE values calculated with the initially reported mass solely for reference. To replicate these results, run the MATLAB file '**RMSE_vector_error_slowtrainingset_m_0752.m**'.


### SecVII-B: Online Learning
1. Run the Python file '**main_code.py**' in the downloaded folder '**SecVII-B (source code)**'.
2. In the prompted terminal interface, you will be asked to select whether to train or evaluate NeuroMHE.
   * Training: type 'train' without the quotation mark in the terminal. Subsequently, you will be prompted to select whether to train NeuroMHE or DMHE.
   * Evaluation: type 'evaluate' without the quotation mark in the terminal. Subsequently, you will be prompted to select a controller for evaluation. There are 5 controllers, as shown below. Note again that, like the case in SecVII-A, you can also skip the training process and directly evaluate the performance using the trained neural network model '**trained_nn_model.pt**' that is saved in the folder '**trained_data**'.

     ![evaluation_mode](https://user-images.githubusercontent.com/70559054/227720537-e2910ce5-7128-4bed-864b-848c787a7413.png)
     
3. Run the Python file '**plotting.py**' to reproduce the figures in Section VII-B of our paper and show the following animation demo.

|                                                 Efficient Online Training Process                                                |
:----------------------------------------------------------------------------------------------------------------------------------:
![synthetic data training](https://github.com/RCL-NUS/NeuroMHE/assets/70559054/209d4f1f-8abe-4db8-abf1-2dc920b6597f)


     
### SecVII-C: Real-world Experiments
1. Run the Python file '**main_code.py**' in the folder '**Training in simulation**' within the downloaded folder '**SecVII-C (source code)**'.
2. In the prompted terminal interface, you will be asked to select whether to train or evaluate NeuroMHE.
   * Training: type 'train' without the quotation mark in the terminal. Subsequently, you will be prompted to select whether to train NeuroMHE or DMHE.
   * Evaluation: type 'evaluate' without the quotation mark in the terminal. Subsequently, you will be prompted to select a controller for evaluation. There are 4 controllers, as shown below.

     ![evaluation_mode](https://user-images.githubusercontent.com/70559054/227721390-17b9d275-23e2-4506-9bd3-c829363c981a.png)

3. Run the Python file '**T-RO-experiment-data-processing_22_Feb.py**' located in the directory '**SecVII-C (source code)/Evaluation in real-world experiment/Experiment Raw Data**' to generate the figures depicting the experimental results. A video demonstration showcasing the experiments is presented below.

https://github.com/RCL-NUS/NeuroMHE/assets/70559054/d2f26efe-321c-4213-a47e-4a5f4297675c
   
Note that we modify the official v1.11.1 PX4 firmware to bypass the PX4's position and velocity controllers. The modified PX4 firmware is available at https://github.com/mamariomiamo/px4_modified/commit/d06d41265b8871c94f5fb110d99f8ec03d3c6907. The primary code implementing our method in the onboard computer, along with other state-of-the-art robust flight controllers, is located in the Python file '**offb_py_v1_neuromhe.py**' (lines 504-659). This file can be found in the directory '**SecVII-C (source code)/Evaluation in real-world experiment/Code used in onboard computer**'.

### Applications to other robots
Please note that although we demonstrated the effectiveness of our approach using a quadrotor, the proposed method is general and can be applied to robust adaptive control for other robotic systems. Only minor modifications in our code are needed for such applications. To illustrate, we can take the source code in the folder '**SecVII-B (source code)**' as an example and proceed as follows:
   * Update the robotic dynamics model in the Python file '**UavEnv.py**';
   * Update the robotic controller in the Python file '**Robust_Flight.py**';
   * Update the simulation environment for training and evaluation in the Python file '**main_code.py**'.


## 4. Acknowledgement
We thank Leonard Bauersfeld for the help in using the flight dataset of NeuroBEM.

## 5. Contact Us
If you encounter a bug in your implementation of the code, please do not hesitate to inform me.
* Name: Dr. Bingheng Wang
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

