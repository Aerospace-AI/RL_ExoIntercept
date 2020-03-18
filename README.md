# Code for  "[Reinforcement Learning for Angle-Only Intercept Guidance of Maneuvering Targets](https://arxiv.org/abs/1906.02113)"
# If you use this code in your research, please cite:

@article{gaudet2020reinforcement,
  title={Reinforcement learning for angle-only intercept guidance of maneuvering targets},
  author={Gaudet, Brian and Furfaro, Roberto and Linares, Richard},
  journal={Aerospace Science and Technology},
  volume={99},
  pages={105746},
  year={2020},
  publisher={Elsevier}
}


## Notes:
### Since this was ported to a standalone repository, the path in the notebook needs to be adjusted from:
sys.path.append('../../../RL_lib/Agents')   
sys.path.append('../../../RL_lib/Policies/PPO')   
sys.path.append('../../../RL_lib/Policies/Common')   
sys.path.append('../../../RL_lib/Utils')   

### To:
sys.path.append('../../RL_lib/Agents')   
sys.path.append('../../RL_lib/Policies/PPO')   
sys.path.append('../../RL_lib/Policies/Common')   
sys.path.append('../../RL_lib/Utils')   

## virtual environment should include:
### Pytorch 
### Python 3.5
### Matplotlib 2.1.1  (newer versions mess up the plotting in notebook)
### Scipy
### scikit-learn
### jupyter

### 1.) Download this repository
### 2.) cd to appropriate directory in Experiments
### 3.) open jupyter from the virtual environment
### 4.) run the notebook
