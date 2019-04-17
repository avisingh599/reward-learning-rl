# reward-learning-rl

This repository is the official implementation of the following paper: 

**End-to-End Robotic Reinforcement Learning without Reward Engineering** <br/>
Avi Singh, Larry Yang, Kristian Hartikainen, Chelsea Finn, Sergey Levine <br/>
arXiv 2019 <br/>
[Website](https://sites.google.com/view/reward-learning-rl/) | [Video](https://www.youtube.com/watch?v=9pWJzb4G-CA) | [Arxiv](https://arxiv.org/abs/1904.07854)

| Visual Draping             |  Visual Pushing | Visual Bookshelf | 
:-------------------------:|:-------------------------:|:-------------------------:
![](https://imgur.com/CbwNUZv.gif)  |  ![](https://imgur.com/NPQBnBW.gif) |  ![](https://imgur.com/bl6Qszu.gif) <br/> ![](https://imgur.com/MpNlZxK.gif)


| Visual Door Opening      |   Visual Pusher | Visual Picker | 
:-------------------------:|:-------------------------:|:-------------------------:
![](https://imgur.com/zPI1zAa.gif) | ![](https://imgur.com/pns0uwY.gif) | ![](https://imgur.com/UYbpTLt.gif) |

We propose a method for end-to-end learning of robotic skills in the real world using deep reinforcement learning. We learn these policies directly on pixel observations, and we do so without any hand-engineered or task-specific reward functions, and instead learn the rewards for such tasks from a small number of user-provided goal examples (around 80), followed by a modest number of active queries (around 25-75).  

This implementation is based on [softlearning](https://github.com/rail-berkeley/softlearning).

# Getting Started

## Prerequisites

The environment can be run either locally using conda or inside a docker container. For conda installation, you need to have [Conda](https://conda.io/docs/user-guide/install/index.html) installed. For docker installation you will need to have [Docker](https://docs.docker.com/engine/installation/) and [Docker Compose](https://docs.docker.com/compose/install/) installed. Also, most of our environments currently require a [MuJoCo](https://www.roboti.us/license.html) license.

## Conda Installation

1. [Download](https://www.roboti.us/index.html) and install MuJoCo 1.50 from the MuJoCo website. We assume that the MuJoCo files are extracted to the default location (`~/.mujoco/mjpro150`).

2. Copy your MuJoCo license key (mjkey.txt) to ~/.mujoco/mjkey.txt:

3. Clone `reward-learning-rl`
```
git clone https://github.com/avisingh599/reward-learning-rl.git ${REWARD_LEARNING_PATH}
```

4. Create and activate conda environment, install softlearning to enable command line interface.
```
cd ${REWARD_LEARNING_PATH}
conda env create -f environment.yml
conda activate softlearning
pip install -e ${REWARD_LEARNING_PATH}
```

The environment should be ready to run. See examples section for examples of how to train and simulate the agents.

Finally, to deactivate and remove the conda environment:
```
conda deactivate
conda remove --name softlearning --all
```

## Docker Installation

### docker-compose
To build the image and run the container:
```
export MJKEY="$(cat ~/.mujoco/mjkey.txt)" \
    && docker-compose \
        -f ./docker/docker-compose.dev.cpu.yml \
        up \
        -d \
        --force-recreate
```

You can access the container with the typical Docker [exec](https://docs.docker.com/engine/reference/commandline/exec/)-command, i.e.

```
docker exec -it softlearning bash
```

See examples section for examples of how to train and simulate the agents.

Finally, to clean up the docker setup:
```
docker-compose \
    -f ./docker/docker-compose.dev.cpu.yml \
    down \
    --rmi all \
    --volumes
```

## Examples
### Training an agent
```
softlearning run_example_local examples.classifier_rl \
--n_goal_examples 10 \
--task=Image48SawyerDoorPullHookEnv-v0 \
--algorithm VICERAQ \
--num-samples 5 \
--n_epochs 300 \
--active_query_frequency 10
```
The tasks used in the paper were `Image48SawyerPushForwardEnv-v0`, `Image48SawyerDoorPullHookEnv-v0` and `Image48SawyerPickAndPlace3DEnv-v0`.  For the algorithm, you can experiment with `VICERAQ`, `VICE`,  `RAQ`, `SACClassifier`, and `SAC`. The `--num-samples` flag specifies the number of random seeds launched. All results in the paper were averaged across five random seeds. The hyperparameters are stored in `examples/classifier_rl/variants.py`. 

`examples.classifier_rl.main` contains several different environments. For more information about the agents and configurations, run the scripts with `--help` flag: `python ./examples/classifier_rl/main.py --help`. 

If this codebase helps you in your academic research, you are encouraged to cite our paper. Here is an example bibtex:
```
@article{singh2019,
  title={End-to-End Robotic Reinforcement Learning without Reward Engineering},
  author={Avi Singh and Larry Yang and Kristian Hartikainen and Chelsea Finn and Sergey Levine},
  journal={arXiv preprint arXiv:1904.07854},
  year={2019}
}
```
