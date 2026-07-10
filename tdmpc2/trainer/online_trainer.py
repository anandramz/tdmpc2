import os
from pathlib import Path
from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from trainer.base import Trainer


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()
		self._tds = None
		self._resumed = False
		# Agent updates start once `seed_steps` of experience have been collected.
		# On resume the replay buffer is empty (it is not checkpointed), so we
		# re-collect that much data before updating again.
		self._update_start = self.cfg.seed_steps
		if self.cfg.get('resume', False):
			self.load_checkpoint()

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		elapsed_time = time() - self._start_time
		return dict(
			step=self._step,
			episode=self._ep_idx,
			elapsed_time=elapsed_time,
			steps_per_second=self._step / elapsed_time
		)

	def _checkpoint_path(self):
		"""Path of the resume checkpoint. Point `checkpoint_dir` at Google Drive to survive disconnects."""
		d = self.cfg.get('checkpoint_dir', None)
		d = Path(d) if d else Path(self.cfg.work_dir) / 'checkpoints'
		return d / 'latest.pt'

	def _local_checkpoint_path(self):
		"""Fallback path on local disk, used if the primary (e.g. Drive) write fails."""
		return Path(self.cfg.work_dir) / 'checkpoints' / 'latest.pt'

	def save_checkpoint(self):
		"""Snapshot model, both optimizers, the reward scale, and progress (no replay buffer).

		Writes atomically (tmp file + os.replace) so an interrupted write cannot leave a
		corrupt `latest.pt`, and never raises: if the primary destination fails (e.g. a
		stale Google Drive mount midway through a long run) we fall back to local disk
		rather than killing the training run.
		"""
		state = {
			'model': self.agent.model.state_dict(),
			'optim': self.agent.optim.state_dict(),
			'pi_optim': self.agent.pi_optim.state_dict(),
			'scale': self.agent.scale.state_dict(),
			'step': self._step,
			'ep_idx': self._ep_idx,
			'wandb_run_id': getattr(self.logger, 'run_id', None),
		}
		paths = [self._checkpoint_path()]
		if self._local_checkpoint_path() != paths[0]:
			paths.append(self._local_checkpoint_path())
		saved = None
		for fp in paths:
			try:
				fp.parent.mkdir(parents=True, exist_ok=True)
				tmp = fp.with_suffix('.tmp')
				torch.save(state, tmp)
				os.replace(tmp, fp)
				print(f'Saved checkpoint at step {self._step:,} -> {fp}')
				saved = fp
				break
			except Exception as e:
				print(f'Warning: could not write checkpoint to {fp}: {e}')
		if saved is None:
			print('Warning: checkpoint was not saved at this step; training continues.')
			return
		# Mirror the checkpoint to wandb as an artifact so it survives the VM
		# (fail-soft: a network hiccup must not kill training).
		if self.cfg.get('checkpoint_wandb', True):
			self.logger.log_checkpoint(saved)

	def load_checkpoint(self):
		"""Restore from the latest checkpoint, if one exists.

		Looks for a local file first; on a fresh VM (no local file) falls back to
		downloading the latest wandb checkpoint artifact for this run config.
		"""
		fp = self._checkpoint_path()
		if not fp.exists() and self.cfg.get('checkpoint_wandb', True):
			downloaded = self.logger.download_checkpoint(fp.parent)
			if downloaded is not None:
				fp = downloaded
		if not fp.exists():
			print(f'No checkpoint found at {fp}; starting from scratch.')
			return False
		ckpt = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
		self.agent.model.load_state_dict(ckpt['model'])
		self.agent.optim.load_state_dict(ckpt['optim'])
		self.agent.pi_optim.load_state_dict(ckpt['pi_optim'])
		self.agent.scale.load_state_dict(ckpt['scale'])
		self._step = int(ckpt['step'])
		self._ep_idx = int(ckpt['ep_idx'])
		self._resumed = True
		self._update_start = self._step + self.cfg.seed_steps
		print(f'Resumed from {fp} at step {self._step:,}')
		run_id = ckpt.get('wandb_run_id')
		if run_id:
			print(f'To continue the same wandb run, set WANDB_RUN_ID={run_id} WANDB_RESUME=allow')
		return True

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		gamma = float(self.agent.discount)
		ep_rewards, ep_disc_returns, ep_successes, ep_lengths = [], [], [], []
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, disc_return, disc, t = self.env.reset(), False, 0, 0, 1.0, 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done:
				torch.compiler.cudagraph_mark_step_begin()
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				disc_return += disc * reward
				disc *= gamma
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			ep_disc_returns.append(disc_return)
			ep_successes.append(info['success'])
			ep_lengths.append(t)
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			undiscounted_return=np.nanmean(ep_rewards),
			discounted_return=np.nanmean(ep_disc_returns),
			episode_success=np.nanmean(ep_successes),
			episode_length= np.nanmean(ep_lengths),
		)

	def to_td(self, obs, action=None, reward=None, terminated=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		if terminated is None:
			terminated = torch.tensor(float('nan'))
		td = TensorDict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
			terminated=terminated.unsqueeze(0),
		batch_size=(1,))
		return td

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, True, False
		while self._step <= self.cfg.steps:
			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			# Reset environment
			if done:
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				if self._step > 0 and self._tds is not None:
					if info['terminated'] and not self.cfg.episodic:
						raise ValueError('Termination detected but you are not in episodic mode. ' \
						'Set `episodic=true` to enable support for terminations.')
					rewards = torch.cat([td['reward'] for td in self._tds[1:]])
					gamma = float(self.agent.discount)
					discounts = gamma ** torch.arange(len(rewards), dtype=rewards.dtype)
					undiscounted_return = rewards.sum()
					discounted_return = (discounts * rewards).sum()
					train_metrics.update(
						episode_reward=undiscounted_return,
						undiscounted_return=undiscounted_return,
						discounted_return=discounted_return,
						episode_success=info['success'],
						episode_length=len(self._tds),
						episode_terminated=info['terminated'])
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					self._ep_idx = self.buffer.add(torch.cat(self._tds))

				obs = self.env.reset()
				self._tds = [self.to_td(obs)]

			# Collect experience
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds)==1)
			else:
				action = self.env.rand_act()
			obs, reward, done, info = self.env.step(action)
			self._tds.append(self.to_td(obs, action, reward, info['terminated']))

			# Update agent
			if self._step >= self._update_start:
				if self._step == self._update_start and not self._resumed:
					num_updates = self.cfg.seed_steps
					print('Pretraining agent on seed data...')
				else:
					num_updates = 1
				for _ in range(num_updates):
					_train_metrics = self.agent.update(self.buffer)
				train_metrics.update(_train_metrics)

			self._step += 1

			# Periodic checkpoint so an interrupted run can resume
			if self.cfg.get('checkpoint_freq', 0) and self._step % self.cfg.checkpoint_freq == 0:
				self.save_checkpoint()

		self.save_checkpoint()
		self.logger.finish(self.agent)
