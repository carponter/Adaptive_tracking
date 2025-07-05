# Empowering Embodied Visual Tracking with Visual Foundation Models and Offline RL

![Example Image](Overview/overview_v7.png)  

[FangWei Zhong](https://fangweizhong.xyz/), [Kui Wu](), [Hai Ci](), [Chu-ran Wang ](), [Hao Chen]()

Peking University, BeiHang University, National University of Singapore and The Hong Kong Polytechnic University.

ECCV 2024

[[arXiV]](https://arxiv.org/abs/2404.09857)  [[Project Page]](https://sites.google.com/d/1TlnjsKbF2IgvdM9-aMJLShlnVBlS9ttN/p/1NZNTU2LmzgeXYYwuFn4w4r9pZaw-gdYN/edit?pli=1)  

## Installation
Our model rely on the DEVA as the vision foundation model and Gym-Unrealcv as the evaluation environment, which requires to install three additional packages: Grounded-Segment-Anything, DEVA and Gym-Unrealcv. Note that we modified the original DEVA to adapt to our task, we provide the modified version in the repository.
**Prerequisite:**
- Python 3.9
- PyTorch 2.0.1+ and corresponding torchvision
- gym_unrealcv(https://github.com/zfw1226/gym-unrealcv)
- Grounded-Segment-Anything (https://github.com/hkchengrex/Grounded-Segment-Anything)
- DEVA (https://github.com/hkchengrex/Tracking-Anything-with-DEVA)

**Clone our repository:**
```bash
git clone https://github.com/wukui-muc/Offline_RL_Active_Tracking.git
```

**Install Grounded-Segment-Anything:**  
```bash
cd Offline_RL_Active_Tracking
git clone https://github.com/hkchengrex/Grounded-Segment-Anything

export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda/
# /path/to/cuda/ is the path to the cuda installation directory, e.g., /usr/local/cuda
# if you install the cuda in conda, it should be {path_to_conda}/env/{conda_env_name}/lib/, e.g., ~/anaconda3/env/offline_evt/lib/

cd Grounded-Segment-Anything
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO
pip install --upgrade diffusers[torch]
```
**Install DEVA:**  
Directly install the modified DEVA in the repository  
(If you encounter the `File "setup.py" not found` error, upgrade your pip with `pip install --upgrade pip`)
```bash
cd ../Tracking-Anything-with-DEVA # go to the DEVA directory
pip install -e .
bash scripts/download_models.sh #download the pretrained models
```

**Install Gym-Unrealcv:**
```bash
cd ..
git clone https://github.com/zfw1226/gym-unrealcv.git
cd gym-unrealcv
pip install -e .
```
Before running the environments, you need to prepare unreal binaries. You can load them from clouds by running load_env.py
```bash
python load_env.py -e {ENV_NAME}

# To run the demo evaluation script, you need to load the UrbanCityMulti environment and textures by running:
python load_env.py -e UrbanCityMulti
python load_env.py -e Textures
sudo chmod -R 777 ./   #solve the permission problem
```

## Quick Start

### Training

```bash
python train_offline --buffer_path {Data-Path}
```

### Evaluation

```bash
python Eval_tracking_agent.py --env UnrealTrackGeneral-UrbanCity-ContinuousColor-v0 --chunk_size 1 --amp --min_mid_term_frames 5 --max_mid_term_frames 10 --detection_every 20 --prompt person.obstacles 
```

## Citation

```bibtex
@inproceedings{zhong2024empowering,
  title={Empowering embodied visual tracking with visual foundation models and offline rl},
  author={Zhong, Fangwei and Wu, Kui and Ci, Hai and Wang, Churan and Chen, Hao},
  booktitle={European Conference on Computer Vision},
  pages={139--155},
  year={2024},
  organization={Springer}
}
```

## References

Thanks for the previous works that we build upon:  
DEVA: https://github.com/hkchengrex/Tracking-Anything-with-DEVA  
Grounded Segment Anything: https://github.com/IDEA-Research/Grounded-Segment-Anything  
Segment Anything: https://github.com/facebookresearch/segment-anything  
XMem: https://github.com/hkchengrex/XMem  
Title card generated with OpenPano: https://github.com/ppwwyyxx/OpenPano

