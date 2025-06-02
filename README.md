<h1 align="center">STeCa</h1>

This repository contains the code for the paper ["STeCa: Step-level Trajectory Calibration for LLM Agent Learning"](https://arxiv.org/abs/2502.14276)

<p align="center">
  <img src=assets/framework.png width=700/>
</p>


### ✨ Key Advantages:

1. 🎯 **Real-time Calibration**: Identifies and corrects suboptimal behaviors through step-level reward comparison during exploration
2. 🔄 **Automated Trajectory Enhancement**: Leverages LLM-driven reflection to automatically construct calibrated trajectories without human intervention
3. 🚀 **Robust Task Completion**: Combines calibrated and successful trajectories for reinforced training, significantly improving agent performance on complex long-horizon tasks (ALFWorld and VirtualHome)



## 🎉News
- [2025.04.15] 🔥 Our paper is accepted in ACL25 Findings.
- [2025.02.19] 🚀 STeCa Repo launched!

## 📝Contents

- [Setup](#setup)
- [Usage](#method)
  - [Base Agent SFT Training](#base-agent-sft-training)
  - [Explored Trajectories Collection](#environment-exploration)
  - [Data Construction](#data-construction)
  - [Reinforced Training](#rl-training)
  - [Evaluation](#evaluation)
- [Expert Trajectories Collection](#etc)


## ⚙️ Setup

```
# Create virtual environment for agentic evaluation
conda create -n STeCa python=3.9
conda activate STeCa
pip install -r requirements.txt

# Download data for ALFWorld environment
cd eval_agent/data/alfworld
gdown https://drive.google.com/uc?id=1y7Vqeo0_xm9d3I07vZaP6qbPFtyuJ6kI
unzip alfworld_data.zip

# Download data for VirtualHome environment
cd ../..
gdown https://drive.google.com/uc?id=1kZKWkWhtJ-DneqfS1Nb_FybR1RBxPeef
unzip virtualhome_master.zip

# Download expert trajectories for ALFWorld and VirtualHome environment
gdown https://drive.google.com/uc?id=1tqZyqzyE7NnJdUJlwwfcwnPUFMafKdfA
unzip data.zip
```

## ⛏️ Usage 

### 🤖 SFT Training
```
bash sft/my_sft.sh
```
### 🗺️ Data Construction
Firstly, please run following scripts to conduct acquisite step rewards:
```
# For ALFWorld environment

# Collect step rewards for expert step
bash sampling/alfworld/expert_generate_response.sh
bash sampling/alfworld/expert_mc_explore.sh

# Collect step rewards for explored step
bash sampling/alfworld/explore_generate_response.sh
bash sampling/alfworld/explore_mc_explore.sh
```

```
# For VirtualHome environment

# Collect step rewards for expert step
bash sampling/virtualhome/expert_generate_response.sh
bash sampling/virtualhome/expert_mc_explore.sh

# Collect step rewards for explored step
bash sampling/virtualhome/explore_generate_response.sh
bash sampling/virtualhome/explore_mc_explore.sh
```

Then, run the following script to construct the calibration dataset, explored successful trajectory datase and the expert sub-trajectory dataset.
```
# For ALFWorld environment
# Collecting expert sub-trajectory dataset
python data_collection/data_expert_alfworld.py
# Collecting calibration trajectory
python data_collection/data_cali_alfworld.py
# Collecting explored successful trajectory dataset
python data_collection/data_explored_alfworld.py

# For VirtualHome environment
# Collecting expert sub-trajectory dataset
python data_collection/data_org_vh.py
# Collecting calibration trajectory
python data_collection/data_cali_vh.py
# Collecting explored successful trajectory dataset
python data_collection/data_explored_vh.py
```

### 💪 Reinforced Training
```
bash Reinfoced_train/steca_train.sh
```

### 📊 Evaluation
```
# ALFWorld
bash eval/my_eval_alfworld.sh
# VirtualHome
bash eval/my_eval_vh.sh
```

## 🙏 Acknowledgments

This codebase is built from [ETO](https://github.com/Yifan-Song793/ETO) and [IPR](https://github.com/WeiminXiong/IPR).

## 📖 Citation

If you find this repo helpful, please cite our paper:

```
@article{wang2025steca,
  title={Steca: Step-level trajectory calibration for llm agent learning},
  author={Wang, Hanlin and Wang, Jian and Leong, Chak Tou and Li, Wenjie},
  journal={arXiv preprint arXiv:2502.14276},
  year={2025}
}
```