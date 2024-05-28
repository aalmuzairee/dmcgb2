# Makes the environment, supports [dmc, dmcgb, dcs]
import envs.dmc.dmc_wrapper as dmc # DeepMind Control Suite
import envs.dmcgb.dmcgb_wrapper as dmcgb # DeepMind Control Generalization Benchmark
import envs.dcs.dcs_wrapper as dcs # Distracting Control Suite


def make(name, frame_stack, action_repeat, seed, mode="train", intensity=0, dir_paths=None):
    mode = mode.lower()
    # Deepmind Control Suite
    if mode == "train":
        env = dmc.make(name=name, frame_stack=frame_stack, action_repeat=action_repeat, seed=seed)
    # Deepmind Control Generalization Benchmark
    elif mode in dmcgb.valid_modes:
        env = dmcgb.make(name=name, frame_stack=frame_stack, action_repeat=action_repeat, seed=seed, mode=mode)
    # Distracting Control Suite
    elif mode == "dcs":
        env = dcs.make(name=name, frame_stack=frame_stack, action_repeat=action_repeat, seed=seed, intensity=intensity, dir_paths=dir_paths)
    # logging
    env._mode = mode 
    if mode == "dcs":
        env._mode = f"{mode}_{intensity}"
    return env