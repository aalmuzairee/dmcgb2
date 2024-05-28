# Everything is written in train.py, but we provide this complimentary script for choosing to evaluate manually 

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
import hydra
import torch

torch.backends.cudnn.benchmark = True


@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    # Turn off wandb for local testing
    cfg.use_wandb=False
    cfg.strong_augs=['all']
    
    # Choose eval envs for singular environment testing
    cfg.eval_modes = ['color_hard']
    cfg.num_eval_episodes = 10

    # Choose test envs for comprehensive testing
    cfg.test_modes = ['dmcgb_geo', 'dmcgb_photo', 'dcs']
    cfg.num_test_episodes = 100


    # Load Checkpoint
    from train import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    else:
        print("WARNING: No Snapshot found")

    # Eval
    workspace.eval()

    # Test
    workspace.test()


if __name__ == '__main__':
    main()