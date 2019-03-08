# Gym Fetch Stack

Mujoco Block Stacking Gym Reinforcement Learning Environments.  
(Modified from [OpenAI Robotics Gym Environments](https://github.com/openai/gym/tree/master/gym/envs/robotics))
![](/media/stack4.png)

### Setup
(You need a Mujoco License. Follow the instructions from OpenAI gym.)

In the gym_fetch_stack root dir, use
```
pip install -e .
```
In your python code, use:
```python
import gym
import gym_fetch_stack
env = gym.make("FetchStack2-v1")
```

## Environments Available:
### Incremental Rewards (Sparse reward for each correctly placed block):
(FetchStack*i* has _i_ blocks in the environment to stack)
- FetchStack1-v1 
- FetchStack2-v1
- FetchStack3-v1
- FetchStack4-v1

### Sparse Rewards (Single Sparse reward for completed stack):
- FetchStack1Sparse-v1
- FetchStack2Sparse-v1
- FetchStack3Sparse-v1
- FetchStack4Sparse-v1

### Dense Rewards (Reward is negative total distance between blocks and goals):
- FetchStack1Dense-v1
- FetchStack2Dense-v1
- FetchStack3Dense-v1
- FetchStack4Dense-v1
