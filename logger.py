import os
import datetime
import re
import numpy as np
import pandas as pd
from termcolor import colored
from omegaconf import OmegaConf


CONSOLE_FORMAT = [
	('frame', 'F', 'int'), 
	('step', 'S', 'int'),
	('episode', 'E', 'int'),
	('episode_reward', 'R', 'float'),
	('episode_length', 'L', 'int'),
	('total_time', 'T', 'time'),
	('fps', 'FPS', 'float'),
	('mode', 'M', 'str'), 
	('intensity', 'I', 'float'), # Distracting Control Suite
]


CAT_TO_COLOR = {
	"train": "white",
	"eval": "green",
}

def make_dir(dir_path):
	try:
		os.makedirs(dir_path)
	except OSError:
		pass
	return dir_path


def print_run(cfg):
	"""Pretty-printing of run information. Call at start of training."""
	prefix, color, attrs = '  ', 'green', ['bold']
	def limstr(s, maxlen=32):
		return str(s[:maxlen]) + '...' if len(str(s)) > maxlen else s
	def pprint(k, v):
		print(prefix + colored(f'{k.capitalize()+":":<16}', color, attrs=attrs), limstr(v))
	kvs = [('task', cfg.task),
		   ('algorithm', cfg.agent),
		   ('augmentations', cfg.get('strong_augs', "None")),
		   ('experiment', cfg.exp_name),
		   ('seed', cfg.seed),
		   ('train frames', f'{int(cfg.num_train_frames):,}'),
		   ('observations', 'x'.join([str(s) for s in cfg.agent_cfg.obs_shape])),
		   ('actions', cfg.agent_cfg.action_shape[0]),
		   ('use_wandb', cfg.use_wandb)]
	
	w = np.max([len(limstr(str(kv[1]))) for kv in kvs]) + 21
	div = '-'*w
	print(div)
	for k,v in kvs:
		pprint(k, v)
	print(div)
	

def cfg_to_group(cfg, return_list=False):
	"""Return a wandb-safe group name for logging. Optionally returns group name as list."""
	lst = [cfg.task, cfg.agent, re.sub('[^0-9a-zA-Z]+', '-', cfg.exp_name)]
	return lst if return_list else '-'.join(lst)




class Logger:
	"""Primary logging object. Logs either locally or using wandb."""

	def __init__(self, work_dir, cfg):
		self._cfg = cfg
		self._log_dir = make_dir(work_dir / "logs")
		self._save_csv = cfg.save_csv
		self._group = cfg_to_group(cfg)
		self._seed = cfg.seed
		self._eval = {}
		print_run(cfg)
		self.project = cfg.get("wandb_project", "none")
		self.entity = cfg.get("wandb_entity", "none")
		if not cfg.use_wandb or self.project == "none" or self.entity == "none":
			print(colored("Wandb disabled.", "blue", attrs=["bold"]))
			self._wandb = None
			return
		os.environ["WANDB_SILENT"] = "true" if cfg.wandb_silent else "false"
		import wandb
		wandb.init(
			project=self.project,
			entity=self.entity,
			name=str(cfg.seed),
			group=self._group,
			tags=cfg_to_group(cfg, return_list=True) + [f"seed:{cfg.seed}"],
			dir=self._log_dir,
			config=OmegaConf.to_container(cfg, resolve=True),
		)
		print(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
		self._wandb = wandb


	def finish(self, model_path):
		if self._wandb:
			if self._cfg.save_snapshot:
				artifact = self._wandb.Artifact(self._group+'-'+str(self._seed), type='model')
				artifact.add_file(model_path)
				self._wandb.log_artifact(artifact)
			self._wandb.finish()

	def _format(self, key, value, ty):
		if ty == "int":
			return f'{colored(key+":", "blue")} {int(value):,}'
		elif ty == "float":
			return f'{colored(key+":", "blue")} {value:.02f}'
		elif ty == "time":
			value = str(datetime.timedelta(seconds=int(value)))
			return f'{colored(key+":", "blue")} {value}'
		elif ty == "str":
			value = str(value)
			return f'{colored(key+":", "blue")} {value}'
		else:
			raise f"invalid log format type: {ty}"

	def _print(self, d, category):
		category = colored(category, CAT_TO_COLOR[category])
		pieces = [f" {category:<14}"]
		for k, disp_k, ty in CONSOLE_FORMAT:
			if (k in d) and d[k] is not None:
				pieces.append(f"{self._format(disp_k, d[k], ty):<22}")
		print("   ".join(pieces))


	def log(self, d, category="train"):
		assert category in CAT_TO_COLOR.keys(), f"invalid category: {category}"
		self.log_wandb(d, category)
		self.log_local(d, category)


	def log_wandb(self, d, category="train", add_step=False):
		assert category in CAT_TO_COLOR.keys(), f"invalid category: {category}"
		if self._wandb:
			step_val = d["frame"] if add_step else None
			full_metrics = {}
			for k, v in d.items():
				full_metrics[f'{category}/{k}'] = v
			self._wandb.log(full_metrics, step=step_val)

	def log_local(self, d, category="train"):
		assert category in CAT_TO_COLOR.keys(), f"invalid category: {category}"
		if category == "eval" and self._save_csv:
			keys = ["frame", "episode_reward"]
			curr_mode = d["mode"]
			if curr_mode not in self._eval.keys():
				self._eval[curr_mode] = []
			self._eval[curr_mode].append(np.array([d[keys[0]], d[keys[1]]]))
			pd.DataFrame(np.array(self._eval[curr_mode])).to_csv(
				self._log_dir / f"{curr_mode}.csv", header=keys, index=None
			)
		self._print(d, category)
