from gym.envs.registration import register

from gym_fetch_stack.envs import *
import sys
from functools import reduce


def str_to_class(str):
    return reduce(getattr, str.split("."), sys.modules[__name__])

def class_exist(className):
    try:
        cls = str_to_class(class_name)
    except AttributeError:
        cls = None
    return True if cls else False


# Robotics
# ----------------------------------------

for num_blocks in [1, 2, 3, 4, 5, 6]:

    # Default reward type is incremental

    for reward_type in ['sparse', 'incremental', 'dense']:
        if reward_type == 'dense':
            suffix = 'Dense'
        elif reward_type == 'sparse':
            suffix = 'Sparse'
        elif reward_type == 'incremental':
            suffix = ''

        kwargs = {
            'reward_type': reward_type,
        }

        # Fetch
        register(
            id='FetchStack{}{}Stage2-v1'.format(num_blocks, suffix),
            entry_point='gym_fetch_stack.envs:FetchStack{}Env'.format(num_blocks),
            kwargs=kwargs,
            max_episode_steps=50 * num_blocks,
        )

        for trainer_type in ['Easy']:
            class_name = 'FetchStack{}Trainer{}Env'.format(num_blocks, trainer_type)
            if class_exist(class_name):

                register(
                    id='FetchStack{}{}Stage1-v1'.format(num_blocks, suffix, trainer_type),
                    entry_point='gym_fetch_stack.envs:{}'.format(class_name),
                    kwargs=kwargs,
                    max_episode_steps=50 * num_blocks,
                )

        class_name = 'FetchStack{}TestEnv'.format(num_blocks)
        if class_exist(class_name):
            register(
                id='FetchStack{}{}Stage3-v1'.format(num_blocks, suffix),
                entry_point='gym_fetch_stack.envs:{}'.format(class_name),
                kwargs=kwargs,
                max_episode_steps=50 * num_blocks,
            )
