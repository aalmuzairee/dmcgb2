

import numpy as np
import os

import dm_control.suite.wrappers
import envs.dcs.suite as dcs
import envs.dmc.dmc_wrapper as dmc


DIFFICULTY_NUM_VIDEOS = {'0.025': 2, '0.05': 2, '0.1': 4, '0.15': 6, '0.2': 8, '0.3': None, '0.4': None, '0.5': None}
DEFAULT_BACKGROUND_PATH = "$HOME/davis/"


def make(name, frame_stack, action_repeat, seed, intensity=0 , dir_paths=None):
    domain_name, task_name = name.split('_', 1)
    domain_name = dict(cup='ball_in_cup').get(domain_name, domain_name)

    # Distracting Control Suite
    paths = []
    if dir_paths:
        for path in dir_paths:
            loaded_path = os.path.join(path, 'DAVIS/JPEGImages/480p')
            if os.path.exists(loaded_path):
                paths.append(loaded_path)
    
    assert len(paths) > 0, f"Need the DAVIS dataset to run the distracting control suite, but no dataset path was given."
    
    task_kwargs = {}
    task_kwargs['random'] = seed

    # zoom in camera for quadruped, for pixel wrapper
    camera_id = dict(quadruped=2).get(domain_name, 0)
    render_kwargs = dict(height=84, width=84, camera_id=camera_id)

    env = dcs.load(
        domain_name,
        task_name,
        task_kwargs=task_kwargs,
        visualize_reward=False,
        environment_kwargs=None,
        render_kwargs=render_kwargs,
        difficulty=intensity,
        dynamic=True,
        background_dataset_paths=paths
    )
    # Wrappers
    env = dmc.ActionDTypeWrapper(env, np.float32)
    env = dmc.ActionRepeatWrapper(env, action_repeat)
    env = dm_control.suite.wrappers.action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    env = dmc.FrameStackWrapper(env, frame_stack, pixels_key="pixels") # Stack
    env = dmc.ExtendedTimeStepWrapper(env)


    return env
