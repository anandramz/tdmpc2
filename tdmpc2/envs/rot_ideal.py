"""TD-MPC2 environment factory for the *idealized* SO(3) rotation benchmark.

Lets a TD-MPC2 agent train on the physics-free orientation task from
`rotations.envs.rot.RotEnv`: the state is a single quaternion, the goal is a
random quaternion, and the agent must rotate the state to match the goal. There
is no robot and no simulator - only the rotation dynamics - so this is the
cleanest place to compare *action* rotation representations (euler / tangent /
quat / matrix / r6) with everything else held fixed.

The agent is untouched: it emits a normalized [-1, 1] action of size
`RotType(cfg.rot_type).dim`, which `RotEnv` decodes/projects onto SO(3). The
observation is `RotType(cfg.rot_obs_type)` for both the current and desired
orientation, concatenated into a flat Box.

`RotEnv` is a JAX `VectorEnv` that auto-resets *inside* `step` and therefore
never reports `truncated=True` to a single-env consumer. We run it with
`num_envs=1`, push its internal time limit out of the way, and enforce the
episode horizon (`cfg.rot_max_steps`) here so TD-MPC2 sees proper episode
boundaries. See SO3_INTEGRATION_PLAN.md for the design rationale.
"""

import os
import sys
from pathlib import Path

# The `rotations` package lives at the repo root, two levels above this package
# (tdmpc2/tdmpc2/envs/ -> tdmpc2/).
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

# Keep JAX (pulled in by the rotations package) from grabbing all GPU memory
# next to torch.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import gymnasium as gym
import numpy as np

from rotations.envs.rot import RotEnv
from rotations.rotations import RotType


# cfg.task -> nothing extra; the idealized env has a single task identity.
ROT_IDEAL_TASKS = {"rot-orient"}

# quat_plus is excluded for the *action*: its w-component needs an asymmetric
# [0, 1] bound that TD-MPC2's uniform tanh + clamp(-1, 1) cannot produce. The
# other reps are symmetric in [-1, 1] and fine.
SUPPORTED_ROT_TYPES = {
    RotType.euler,
    RotType.tangent,
    RotType.quat,
    RotType.matrix,
    RotType.r6,
}

# Effectively-infinite internal horizon so RotEnv's in-step auto-reset never
# fires; the adapter owns the real horizon.
_NO_INTERNAL_TIMELIMIT = 1 << 30


class RotIdealTDMPCWrapper(gym.Env):
    """Adapt the single-env slice of `RotEnv` to TD-MPC2's classic-gym API.

    `RotEnv` is a `gymnasium.vector.VectorEnv`, so it can't be wrapped with a
    plain `gym.Wrapper` (and TD-MPC2's `TensorWrapper` asserts its target is a
    `gymnasium.Env`). We therefore hold it as a member and present a single-env
    `gym.Env`: `reset() -> obs` and `step(a) -> (obs, reward, done, info)` with a
    flat Box `obs` and `info['success']`/`info['terminated']`. `RotEnv`
    (num_envs=1) returns batched JAX arrays and a `{observation, desired_goal,
    achieved_goal}` Dict; we squeeze the batch dim, keep `observation +
    desired_goal`, and convert to numpy float32.
    """

    def __init__(self, env, max_episode_steps, tol, keys=("observation", "desired_goal")):
        super().__init__()
        self.env = env
        self.max_episode_steps = max_episode_steps
        self.tol = float(tol)
        self.keys = list(keys)
        spaces = env.single_observation_space.spaces
        lows = np.concatenate([np.broadcast_to(spaces[k].low, spaces[k].shape) for k in self.keys])
        highs = np.concatenate([np.broadcast_to(spaces[k].high, spaces[k].shape) for k in self.keys])
        self.observation_space = gym.spaces.Box(
            low=lows.astype(np.float32), high=highs.astype(np.float32), dtype=np.float32
        )
        self.action_space = env.single_action_space
        self._t = 0

    def _flatten(self, obs):
        # obs[k] is a JAX array of shape (1, dim); squeeze the batch dim.
        return np.concatenate(
            [np.asarray(obs[k], dtype=np.float32).reshape(-1) for k in self.keys], axis=-1
        )

    def reset(self, task_idx=None):
        obs, _info = self.env.reset()
        self._t = 0
        return self._flatten(obs)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(1, -1)
        obs, reward, _terminated, _truncated, info = self.env.step(action)
        self._t += 1
        truncated = self._t >= self.max_episode_steps
        angle = float(np.asarray(info["angle"]).reshape(-1)[0])
        info = {"success": float(angle <= self.tol), "terminated": 0.0}
        return self._flatten(obs), float(np.asarray(reward).reshape(-1)[0]), bool(truncated), info


def make_env(cfg):
    """Make an idealized SO(3) rotation environment for TD-MPC2.

    Reads `cfg.rot_type` (action representation), `cfg.rot_obs_type` (observation
    representation), `cfg.rot_control_mode` (rel | abs | rel_scale), and optional
    `cfg.rot_step_len`, `cfg.rot_reward_type`, `cfg.rot_tol`, `cfg.rot_max_steps`.
    """
    if cfg.task not in ROT_IDEAL_TASKS:
        raise ValueError("Unknown task:", cfg.task)
    assert cfg.get("obs", "state") == "state", "rot-orient only supports state observations."

    rot_type = RotType(cfg.rot_type)
    if rot_type not in SUPPORTED_ROT_TYPES:
        raise ValueError(
            f"rot_type={rot_type.value!r} is not supported for rot-orient. "
            f"Supported: {sorted(t.value for t in SUPPORTED_ROT_TYPES)}."
        )

    max_steps = int(cfg.get("rot_max_steps", 50))
    kwargs = dict(
        num_envs=1,
        tol=float(cfg.get("rot_tol", 0.1)),
        action_type=cfg.rot_type,
        obs_type=cfg.get("rot_obs_type", "matrix"),
        reward_type=cfg.get("rot_reward_type", "sparse"),
        max_steps=_NO_INTERNAL_TIMELIMIT,
        seed=int(cfg.get("seed", 0)),
        device=cfg.get("rot_device", "cpu"),
        control_mode=cfg.get("rot_control_mode", "rel"),
    )
    if cfg.get("rot_step_len", None) is not None:
        kwargs["step_len"] = float(cfg.rot_step_len)

    env = RotEnv(**kwargs)
    env = RotIdealTDMPCWrapper(env, max_episode_steps=max_steps, tol=kwargs["tol"])
    return env
