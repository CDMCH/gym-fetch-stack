# Gym Fetch Stack

Mujoco Block Stacking Gym Reinforcement Learning Environments.  
(Modified from [OpenAI Robotics Gym Environments](https://github.com/openai/gym/tree/master/gym/envs/robotics))
![](/media/stack4.png)

These environments are made for use with [DDPG with Curiosity Driven Exploration and Multi-Criteria Hindsight Experience Replay](https://github.com/CDMCH/ddpg-with-curiosity-and-multi-criteria-her)

### Setup
(You need a Mujoco License. Follow the instructions to set up Mujoco [here](https://github.com/openai/mujoco-py).)

In the gym_fetch_stack root dir, use
```
pip install -e .
```
In your python code, use:
```python
import gym
import gym_fetch_stack
env = gym.make("FetchStack2Stage3-v1")
```

## Curriculum Stages
We use three curriculum stages to train an agent to stack blocks:
- Stage 1: basic manipulation tasks without having to create stacks
- Stage 2: stacking blocks where the environment is initialized at various stages of completion
- Stage 3: stacking blocks where all blocks all initialized away from their target locations

A video example of the different stages can found [here](https://www.youtube.com/watch?v=Mrd_9cxydRQ&feature=youtu.be).

## Environments Available:
### Incremental Rewards (Sparse reward for each correctly placed block):
(FetchStack*i* has _i_ blocks in the environment to stack)

- FetchStack2Stage1-v1
- FetchStack3Stage1-v1
- FetchStack4Stage1-v1

- FetchStack2Stage2-v1
- FetchStack3Stage2-v1
- FetchStack4Stage2-v1

- FetchStack2Stage3-v1
- FetchStack3Stage3-v1
- FetchStack4Stage3-v1

### Binary Rewards (Single Sparse reward for completed stack):

- FetchStack2SparseStage1-v1
- FetchStack3SparseStage1-v1
- FetchStack4SparseStage1-v1

- FetchStack2SparseStage2-v1
- FetchStack3SparseStage2-v1
- FetchStack4SparseStage2-v1

- FetchStack2SparseStage3-v1
- FetchStack3SparseStage3-v1
- FetchStack4SparseStage3-v1
