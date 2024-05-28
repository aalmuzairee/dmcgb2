import numpy as np

import dm_control
import dm_control.suite.wrappers


import envs.dmc.dmc_wrapper as dmc
import envs.dmcgb.dmcgb_geometric as dmcgb_geometric
import envs.dmcgb.dmcgb_photometric as dmcgb_photometric


dmcgb_photometric_modes = dmcgb_photometric.dmcgb_photometric_modes
dmcgb_geometric_modes = dmcgb_geometric.dmcgb_geometric_modes
valid_modes = dmcgb_photometric_modes + dmcgb_geometric_modes


def make(name, frame_stack, action_repeat, seed, mode="color_hard"):
    domain_name, task_name = name.split('_', 1)
    domain_name = dict(cup='ball_in_cup').get(domain_name, domain_name)
    assert mode in valid_modes , f'Specified mode "{mode}" is not supported'

    env = dm_control.suite.load(domain_name, task_name,task_kwargs={'random': seed},visualize_reward=False)
    env._domain_name = domain_name
    # Wrappers
    env = dmc.ActionDTypeWrapper(env, np.float32)
    env = dmc.ActionRepeatWrapper(env, action_repeat)
    env = dm_control.suite.wrappers.action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    # ------ DMCGB ------ #
    env = dmcgb_geometric.ShiftWrapper(env, mode, seed) 
    env = dmcgb_geometric.RotateWrapper(env, mode, seed) 
    env = dmcgb_photometric.ColorVideoWrapper(env, mode, seed, video_render_size=256) 
    # ------------------- #
    camera_id = dict(quadruped=2).get(domain_name, 0)
    render_kwargs = dict(height=84, width=84, camera_id=camera_id)
    env = dm_control.suite.wrappers.pixels.Wrapper(env, pixels_only=True, render_kwargs=render_kwargs)
    env = dmc.FrameStackWrapper(env, frame_stack, pixels_key="pixels")
    env = dmc.ExtendedTimeStepWrapper(env)
    return env


