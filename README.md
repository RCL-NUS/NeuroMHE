# NeuroMHE
The ***Neural Moving Horizon Estimation (NeuroMHE)*** is an auto-tuning and adaptive optimal estimator. It fuses a neural network with an MHE to realize accurate estimation and fast online adaptation, leveraging the advantages from both advanced machine learning techniques and control-theoretic estimation and control algorithms. The neural network can be trained efficiently from the robot's trajectory tracking errors using reinforcement learning without the need for the ground truth data.

## Table of contents
1. [Project Overview](#project-Overview)
2. [Dependency Packages](#Dependency-Packages)
3. [How to Use](#How-to-Use)
4. [Contact Us](#Contact-Us)


## 1. Project Overview
The project consists of three folders, which correspond to the three experiments in the paper that show the following four features of our method.
1. NeuroMHE enjoys computationally efficient training and significantly improves the force estimation performance as compared to a state-of-the-art estimator [[1]](#1).
2. A stable NeuroMHE with a fast dynamic response can be trained directly from the trajectory tracking error using Algorithm 2 (i.e., the model-based policy gradient algorithm) .
3. NeuroMHE exhibits superior estimation and robust control performance than a fixed-weighting MHE and a state-of-the-art adaptive controller [[2]](#2) for handling dynamic noise covariances.
4. Finally, NeuroMHE is efficiently transferable to different challenging flight scenarios on a real quadrotor without extra parameter tuning, such as counteracting state-dependent cable forces and flying under the downwash flow.
* SecVII-A (source code): A comprehensive comparison with the state-of-the-art NeuroBEM estimator on its real agile test dataset to show the first feature. 
* SecVII-B (source code): A robust trajectory tracking control scenario in simulation to show the second and third features.
* SecVII-C (source code): Physical experiments on a real quadrotor to show the fourth feature.

## 2. Dependency Packages
Please make sure that the following packages have already been installed before running the source code.
* CasADi: version 3.5.5 Info: https://web.casadi.org/
* Numpy: version 1.23.0 Info: https://numpy.org/
* Pytorch: version 1.12.0+cu116 Info: https://pytorch.org/
* Matplotlib: version 3.3.0 Info: https://matplotlib.org/
* Python: version 3.9.12 Info: https://www.python.org/


## 3. How to Use
First and foremost, the training process for NeuroMHE is both efficient and straightforward to setup. The source code has been comprehensively annotated to facilitate ease of use. To reproduce the simulation results presented in the paper, simply follow the steps outlined below, sequentially, after downloading and decompressing all the necessary folders.

### SecVII-A
1. Download **processed_data.zip** and **predictions.tat.xz** from https://download.ifi.uzh.ch/rpg/NeuroBEM/. The former file is utilized for training NeuroMHE, whereas the latter serves the purpose of evaluation and comparison with NeuroBEM.
2. Relocate the **bem+nn** folder from the decomprassed **predictions.tat.xz** archive to the downloaded **SecVII-A (source code)** folder, and place the decompressed **processed_data.zip** within the **SecVII-A (source code)** folder as well.



## 4. Contact Us

## References
<a id="1">[1]</a> 
L. Bauersfeld, E. Kaufmann, P. Foehn, S. Sun, and D. Scaramuzza, "NeuroBEM: Hybrid Aerodynamic Quadrotor Model", ROBOTICS: SCIENCE AND SYSTEM XVII,2021.

<a id="2">[2]</a> 
Z. Wu, S. Cheng, K. A. Ackerman, A. Gahlawat, A. Lakshmanan, P. Zhao, and N. Hovakimyan, "L1-Adaptive Augmentation for Geometrictracking Control of Quadrotors", in 2022 International Conference on
Robotics and Automation (ICRA). IEEE, 2022, pp. 1329–1336.
