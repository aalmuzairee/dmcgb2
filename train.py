
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
from tqdm import tqdm
from dm_env import specs
import shutil

from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import VideoRecorder
from logger import Logger
import utils
import env_factory


torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        self.cfg = cfg

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # create train env
        self.train_env = env_factory.make(self.cfg.task, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed, mode="train")
        # create eval envs
        self.eval_envs = []
        self.eval_envs.append(env_factory.make(self.cfg.task, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed, mode="train"))
        for each_mode in self.cfg.eval_modes:
            self.eval_envs.append(env_factory.make(self.cfg.task, self.cfg.frame_stack,
                                self.cfg.action_repeat, self.cfg.seed, mode=each_mode))

        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))
        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')
        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer' , self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)
        self._replay_iter = None

        # create agent
        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.cfg.agent_cfg)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        # create logger
        self.logger = Logger(self.work_dir, self.cfg)
        # create video
        self.video_recorder = VideoRecorder(save_video=self.cfg.save_video, video_dir=self.cfg.video_dir)

        # add places365 dataset directory for overlay aug
        utils.add_aug_directory(self.cfg.datasets)


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    # generalization testing after training
    def test(self):
        self.video_recorder.save_video = self.cfg.save_test_video
        self.cfg.num_eval_episodes = self.cfg.num_test_episodes
        for each_test_mode in self.cfg.test_modes:
            print("Running generalization testing for the following set:", each_test_mode)
            if each_test_mode == "dmcgb_photo":
                dmcgb_set = ["color_easy", "color_hard", "video_easy", "video_hard", "color_video_easy", "color_video_hard"]
                for each_mode in dmcgb_set:
                    self.eval_envs = [env_factory.make(self.cfg.task, self.cfg.frame_stack, self.cfg.action_repeat, self.cfg.seed, mode=each_mode)]
                    self.eval()
            elif each_test_mode == "dmcgb_geo":
                dmcgb_set = ["rotate_easy", "rotate_hard", "shift_easy", "shift_hard", "rotate_shift_easy", "rotate_shift_hard"]
                for each_mode in dmcgb_set:
                    self.eval_envs = [env_factory.make(self.cfg.task, self.cfg.frame_stack, self.cfg.action_repeat, self.cfg.seed, mode=each_mode)]
                    self.eval()
            elif each_test_mode == "dcs":
                dcs_set = ['0.025','0.05','0.1','0.15','0.2','0.3','0.4', '0.5']
                for each_intensity in dcs_set:
                    self.eval_envs = [env_factory.make(self.cfg.task, self.cfg.frame_stack, self.cfg.action_repeat, self.cfg.seed, 
                                                        mode="dcs", intensity=float(each_intensity), dir_paths=self.cfg.datasets)]
                    self.eval()
            else:
                print("Test mode: ", {each_test_mode}, " not found, proceeding to the next.")

    # evaluation during training
    def eval(self):
        metrics = {}
        for each_env in self.eval_envs:
            step, episode, total_reward = 0, 0, 0 
            for episode in tqdm(range(self.cfg.num_eval_episodes), leave=False):
                time_step = each_env.reset()
                self.video_recorder.init(each_env, enabled=(episode == 0))
                while not time_step.last():
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(time_step.observation,
                                                self.global_step,
                                                eval_mode=True)
                    time_step = each_env.step(action)
                    self.video_recorder.record(each_env)
                    total_reward += time_step.reward
                    step += 1

                self.video_recorder.save(file_name=f'{each_env._mode}_{self.global_frame}',
                                            wandb=self.logger._wandb)
            # final increment
            episode += 1
            
            elapsed_time, total_time = self.timer.reset()
            metrics.update({
                'frame': self.global_frame,
                'fps': (step * self.cfg.action_repeat / elapsed_time),
                'episode_reward': total_reward / episode,
                'episode_length': step * self.cfg.action_repeat / episode,
                'episode': self.global_episode,
                'step': self.global_step,
                'total_time': total_time,
                'mode': each_env._mode,
            })

            # logging
            if each_env._mode[:3] == "dcs":
                mode, intensity = each_env._mode.split('_',1)
                metrics['mode'] = mode
                metrics['intensity'] = float(intensity)
            # Log local
            self.logger.log_local(metrics, category="eval")
            # Log wandb
            curr_mode = metrics.pop('mode')
            curr_rew = metrics.pop('episode_reward')
            metrics[f'{curr_mode}_reward'] = curr_rew
            self.logger.log_wandb(metrics, category="eval")  



    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        metrics = {}
        agent_metrics = {}
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                # log stats
                elapsed_time, total_time = self.timer.reset()
                episode_frame = episode_step * self.cfg.action_repeat
                metrics.update(agent_metrics)
                metrics.update({
                    'frame': self.global_frame,
                    'fps': (episode_frame / elapsed_time),
                    'total_time': total_time,
                    'episode_reward': episode_reward,
                    'episode_length': episode_frame,
                    'episode': self.global_episode,
                    'step': self.global_step
                })
                self.logger.log(metrics, category="train")
                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                # save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # evaluate
            if eval_every_step(self.global_step):
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)

            # update the agent
            if not seed_until_step(self.global_step):
                agent_metrics.update(self.agent.update(self.replay_iter, self.global_step))

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            episode_step += 1
            self._global_step += 1

        # finished training
        if self.cfg.save_snapshot:
            self.save_snapshot()
        if self.cfg.save_final_video_once:
            self.video_recorder.save_video = True        
        self.eval()


    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v

    # Saving model snapshot to wandb, close wandb, delete buffer files
    def finish(self):
        snapshot = self.work_dir / 'snapshot.pt'
        self.logger.finish(snapshot)
        try:
            shutil.rmtree("buffer")
        except Exception as e:
            pass

@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    from train import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    
    # train
    workspace.train()

    # generalization test 
    workspace.test()

    # finish
    workspace.finish()


if __name__ == '__main__':
    main()