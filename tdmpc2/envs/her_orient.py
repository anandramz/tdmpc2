"""TD-MPC2 environment factory for the vendored her_orient SO(3) benchmark.

Lets a TD-MPC2 agent train on the FR3 reach/pick orientation tasks while varying
the *action* rotation representation (euler / tangent / quat / matrix) via the
`RotationWrapper` from `benchmarks/her_orient`. The agent is untouched: it emits a
normalized [-1, 1] action of size `3 + RotType(cfg.rot_type).dim` (+1 for the
gripper on pick-and-place), and the wrapper decodes/projects it onto SO(3) and
converts it to the quaternion the MuJoCo env expects.

The reward is the env's own sparse task reward (0 on success, -1 otherwise); the
observation representation is fixed by the env (rotation matrix). See
SO3_INTEGRATION_PLAN.md for the design rationale.
"""

import sys
from pathlib import Path

# The vendored benchmark code (`benchmarks.*`) and the `rotations` package live at
# the repo root, two levels above this package (tdmpc2/tdmpc2/envs/ -> tdmpc2/).
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

import gymnasium as gym
import numpy as np

# Importing the package runs the gymnasium.register(...) calls for the her_orient envs.
import benchmarks.her_orient.envs  # noqa: F401  (registers Reach/ReachOrient/PickAndPlace*-v0)
from benchmarks.her_orient.envs import RotationWrapper
from rotations.rotations import RotType


# cfg.task -> registered gymnasium id. Only the *Orient* variants carry a rotation
# in the action and are wrappable by RotationWrapper (action shape 7 or 8).
HER_ORIENT_TASKS = {
    "her-reach-orient": "ReachOrient-v0",
    "her-pnp-orient": "PickAndPlaceOrient-v0",
}

# RotationWrapper asserts these four; r6 / quat_plus are not supported here.
SUPPORTED_ROT_TYPES = {RotType.euler, RotType.tangent, RotType.quat, RotType.matrix}


class FlattenGoalObs(gym.ObservationWrapper):
    """Concatenate the goal-conditioned Dict observation into a single Box.

    TD-MPC2's state encoder wants a flat Box; her_orient returns
    {observation, desired_goal, achieved_goal}. We keep observation + desired_goal
    (achieved_goal is already contained in observation).
    """

    def __init__(self, env, keys=("observation", "desired_goal")):
        super().__init__(env)
        self.keys = list(keys)
        spaces = env.observation_space.spaces
        lows = np.concatenate([np.broadcast_to(spaces[k].low, spaces[k].shape) for k in self.keys])
        highs = np.concatenate([np.broadcast_to(spaces[k].high, spaces[k].shape) for k in self.keys])
        self.observation_space = gym.spaces.Box(
            low=lows.astype(np.float32), high=highs.astype(np.float32), dtype=np.float32
        )

    def observation(self, obs):
        return np.concatenate([np.asarray(obs[k], dtype=np.float32) for k in self.keys], axis=-1)


class HerOrientTDMPCWrapper(gym.Wrapper):
    """Adapt the gymnasium API to what TD-MPC2's TensorWrapper expects.

    TensorWrapper calls `reset() -> obs` and `step(a) -> (obs, reward, done, info)`
    with `info['success']` and `info['terminated']` populated. her_orient returns
    `reset() -> (obs, info)` and `step(a) -> (obs, reward, terminated, truncated, info)`.
    """

    def __init__(self, env, max_episode_steps):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, task_idx=None):
        obs, _info = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward)
        done = bool(terminated) or bool(truncated)
        info = dict(info)
        # Sparse task reward is 0.0 exactly on success, -1.0 otherwise.
        info["success"] = float(reward == 0.0)
        info["terminated"] = float(bool(terminated))
        return obs, reward, done, info


def make_env(cfg):
    """Make a her_orient SO(3) environment for TD-MPC2.

    Reads `cfg.rot_type` (action representation), `cfg.rot_relative` (delta vs
    absolute) and optional `cfg.rot_scale`.
    """
    if cfg.task not in HER_ORIENT_TASKS:
        raise ValueError("Unknown task:", cfg.task)
    assert cfg.get("obs", "state") == "state", "her_orient SO(3) tasks only support state observations."

    rot_type = RotType(cfg.rot_type)
    if rot_type not in SUPPORTED_ROT_TYPES:
        raise ValueError(
            f"rot_type={rot_type.value!r} is not supported for her_orient. "
            f"Supported: {sorted(t.value for t in SUPPORTED_ROT_TYPES)}."
        )

    render_mode = "rgb_array" if cfg.get("save_video", False) else None
    made = gym.make(HER_ORIENT_TASKS[cfg.task], disable_env_checker=True, render_mode=render_mode)
    max_steps = getattr(made.spec, "max_episode_steps", None) or 50

    env = RotationWrapper(
        made,
        rot_type=cfg.rot_type,
        relative=cfg.get("rot_relative", False),
        rot_scale=cfg.get("rot_scale", None),
    )
    env = FlattenGoalObs(env)
    env = HerOrientTDMPCWrapper(env, max_episode_steps=max_steps)
    return env
