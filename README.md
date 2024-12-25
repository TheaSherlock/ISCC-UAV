# Integrated Sensing, Computation, and Communication for UAV-assisted Federated Edge Learning
This repository contains the implementation of "Integrated Sensing, Computation, and Communication for UAV-assisted Federated Edge Learning" (accepted by the IEEE Transactions on Wireless Communications, 2024).

### Paper Abstract
Federated edge learning (FEEL) enables privacy preserving model training through periodic communication be tween edge devices and the server. Unmanned Aerial Vehicle (UAV)-mounted edge devices are particularly advantageous for FEEL due to their flexibility and mobility in efficient data collection. In UAV-assisted FEEL, sensing, computation, and communication are coupled and compete for limited onboard resources, and UAV deployment also affects sensing and communication performance. Therefore, the joint design of UAV deployment and resource allocation is crucial to achieving the optimal training performance. In this paper, we address the problem of joint UAV deployment design and resource allocation for FEEL via a concrete case study of human motion recognition based on wireless sensing. We first analyze the impact of UAV deployment on the sensing quality and identify a threshold value for the sensing elevation angle that guarantees a satisfactory quality of data samples. Due to the non-ideal sensing channels, we consider the probabilistic sensing model, where the successful sensing probability of each UAV is determined by its position. Then, we derive the upper bound of the FEEL training loss as a function of the sensing probability. Theoretical results suggest that the convergence rate can be improved if UAVs have a uniform successful sensing probability. Based on this analysis, we formulate a training time minimization problem by jointly optimizing UAV deployment, integrated sensing, computation, and communication (ISCC) resources under a desirable optimality gap constraint. To solve this challenging mixed-integer nonconvex problem, we apply the alternating optimization technique, and propose the bandwidth, batch size, and position optimization (BBPO) scheme to optimize these three decision variables alternately.

### Requirements
python>=3.9   
torch==0.4.1   
torchvision==0.2.1

### Installation
git clone https://github.com/username/repository.git  
cd repository  
pip install -r FLLS_BSPO/requirements.txt 

### Directory Structure
├── data/               # Dataset files: constains the training data with different UAV elevation angles, e.g., $55^\circ$   
├── models/             # Model implementations  
├── utils/              # Utility functions  
├── log/     
├── save/                  # Save results  
├── main_fed_spec.py       # Main function     
└── README.md 

### Citation
@article{Yao2024paper,  
  title={Integrated Sensing, Computation, and Communication for UAV-assisted Federated Edge Learning},  
  author={Yao Tang, Guangxu Zhu, Wei Xu, Man Hon Cheung, Tat-Ming Lok, and Shuguang Cui},  
  journal={IEEE Trans. Wirel. Commun.},  
  year={2024},  
  publisher={IEEE}  
}  
