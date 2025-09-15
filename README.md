# Cross-Embodiment Active Visual Tracking via Context-enhanced Adaptation


## Installation

Our model relies on **DEVA** as the vision foundation model and **Gym-Unrealcv** as the evaluation environment, which requires three additional packages:  
- [Grounded-Segment-Anything](https://github.com/hkchengrex/Grounded-Segment-Anything)  
- [DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA)  
- [Gym-Unrealcv](https://github.com/zfw1226/gym-unrealcv)
- [mbrl](https://github.com/facebookresearch/mbrl-lib)  

> ⚠️ We provide a **modified version of DEVA** in this repository to adapt it to our task.

### Prerequisites
- Python **3.9**
- PyTorch **2.0.1+** and corresponding `torchvision`

---

### 1. Clone our repository
```bash
git clone [https://github.com/wukui-muc/Offline_RL_Active_Tracking](https://github.com/carponter/Adaptive_tracking.git
cd Offline_RL_Active_Tracking
```

### 2. Install Grounded-Segment-Anything
```bash
git clone https://github.com/hkchengrex/Grounded-Segment-Anything

export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda/
# Example: /usr/local/cuda
# If CUDA is installed in conda, set to:
# {path_to_conda}/env/{conda_env_name}/lib/
# Example: ~/anaconda3/envs/offline_evt/lib/

cd Grounded-Segment-Anything
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO
pip install --upgrade "diffusers[torch]"
```

### 3. Install DEVA
Directly install the modified DEVA provided in this repository:
```bash
cd ../Tracking-Anything-with-DEVA   # go to the DEVA directory
pip install -e .
bash scripts/download_models.sh     # download the pretrained models
```

### 4. Install Gym-Unrealcv
```bash
cd ..
git clone https://github.com/zfw1226/gym-unrealcv.git
cd gym-unrealcv
pip install -e .
```

### 5. Install mbrl
```bash
pip install mbrl
```
