"""TD-MPC2 environment factory for the vendored robosuite SO(3) benchmark.

Lets a TD-MPC2 agent train on robosuite manipulation tasks while varying the
*action* rotation representation (euler / tangent / quat / matrix / r6) via the
`RotationGymWrapper` from `benchmarks/robosuite`. The agent is untouched: it emits
a normalized [-1, 1] action of size `3 + RotType(cfg.rot_type).dim` (+1 for the
gripper), and the wrapper decodes/projects it onto SO(3) and converts it to the
relative-tangent command robosuite's OSC controller expects.

The reward is robosuite's own (shaped) task reward; the observation representation
is fixed per run by `cfg.rot_obs_type`. This is the "SO3 wrapper + task reward"
design. The separate `robosuite.py` (geodesic-tracking) is left untouched.

See SO3_INTEGRATION_PLAN.md for the design rationale.
"""

import os
import sys
from pathlib import Path

# The vendored benchmark code (`benchmarks.*`) and the `rotations` package live at
# the repo root, two levels above this package (tdmpc2/tdmpc2/envs/ -> tdmpc2/).
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

# Keep JAX (pulled in transitively by the rotations package) from grabbing all GPU
# memory next to torch.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import gymnasium as gym
import robosuite as suite

from benchmarks.robosuite.utils.utils import (
    RotationGymWrapper,
    load_manipulator_controller_config,
)
from rotations.rotations import RotType


# cfg.task -> robosuite suite.make() identity. Robots default to a single Panda.
ROBOSUITE_TASKS = {
    "robosuite-lift": dict(env_name="Lift", robots=["Panda"]),
    "robosuite-stack": dict(env_name="Stack", robots=["Panda"]),
    "robosuite-door": dict(env_name="Door", robots=["Panda"]),
    "robosuite-pickplace-can": dict(env_name="PickPlaceCan", robots=["Panda"]),
    "robosuite-nut-round": dict(env_name="NutAssemblyRound", robots=["Panda"]),
}

# Mirrors benchmarks/robosuite/config/*.toml [env.kwargs]. has_offscreen_renderer
# is toggled on only when video saving is requested (see make_env).
DEFAULT_ENV_KWARGS = dict(
    gripper_types="default",
    use_camera_obs=False,
    use_object_obs=True,
    reward_scale=1.0,
    reward_shaping=True,
    has_renderer=False,
    has_offscreen_renderer=False,
    control_freq=20,
    horizon=500,
    ignore_done=False,
    hard_reset=False,
)

# quat_plus is excluded: its w-component needs an asymmetric [0, 1] bound that
# TD-MPC2's uniform tanh + clamp(-1, 1) cannot produce (and robosuite's wrapper
# would silently treat it as plain quat). r6/matrix are symmetric and fine.
SUPPORTED_ROT_TYPES = {
    RotType.euler,
    RotType.tangent,
    RotType.quat,
    RotType.matrix,
    RotType.r6,
}


def _resolve_base_env(env):
    """Walk down to the underlying robosuite RobotEnv (for _check_success)."""
    base = env
    while hasattr(base, "env"):
        base = base.env
    return base


class RobosuiteTDMPCWrapper(gym.Wrapper):
    """Adapt the gymnasium-style robosuite wrapper to TD-MPC2's classic-gym API.

    TensorWrapper expects `reset() -> obs` and `step(a) -> (obs, reward, done, info)`
    with `info['success']` and `info['terminated']`. RotationGymWrapper returns
    `reset() -> (obs, info)` and `step(a) -> (obs, reward, terminated, truncated, info)`,
    where terminated is always False (robosuite only truncates at the horizon).
    """

    def __init__(self, env, max_episode_steps):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._base_env = _resolve_base_env(env)

    def reset(self, task_idx=None):
        obs, _info = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated) or bool(truncated)
        info = dict(info)
        try:
            success = bool(self._base_env._check_success())
        except Exception:
            success = bool(info.get("success", False))
        info["success"] = float(success)
        info["terminated"] = float(bool(terminated))
        return obs, float(reward), done, info

    def render(self, *args, **kwargs):
        # VideoRecorder calls env.render() with no args and expects an (H, W, 3)
        # frame. robosuite renders via the offscreen renderer (enabled in make_env
        # when save_video=true) and returns vertically-flipped frames.
        frame = self._base_env.sim.render(width=384, height=384, camera_name="agentview")
        return frame[::-1]


def make_env(cfg):
    """Make a robosuite SO(3) environment for TD-MPC2.

    Reads `cfg.rot_type` (action representation), `cfg.rot_obs_type` (fixed obs
    representation), `cfg.rot_control_mode` and optional `cfg.rot_step_len`.
    """
    if cfg.task not in ROBOSUITE_TASKS:
        raise ValueError("Unknown task:", cfg.task)
    assert cfg.get("obs", "state") == "state", "robosuite SO(3) tasks only support state observations."

    rot_type = RotType(cfg.rot_type)
    if rot_type not in SUPPORTED_ROT_TYPES:
        raise ValueError(
            f"rot_type={rot_type.value!r} is not supported for TD-MPC2 robosuite. "
            f"Supported: {sorted(t.value for t in SUPPORTED_ROT_TYPES)}."
        )

    task_cfg = ROBOSUITE_TASKS[cfg.task]
    env_kwargs = dict(DEFAULT_ENV_KWARGS)
    env_kwargs["robots"] = task_cfg["robots"]
    # The RotationGymWrapper asserts this exact controller config.
    env_kwargs["controller_configs"] = load_manipulator_controller_config(keep_rot_scale=False)
    if cfg.get("save_video", False):
        env_kwargs["has_offscreen_renderer"] = True

    base = suite.make(task_cfg["env_name"], **env_kwargs)

    wrapper_kwargs = dict(
        action_type=cfg.rot_type,
        obs_type=cfg.get("rot_obs_type", "matrix"),
        control_mode=cfg.get("rot_control_mode", "rel"),
    )
    if cfg.get("rot_step_len", None) is not None:
        wrapper_kwargs["step_len"] = cfg.rot_step_len

    env = RotationGymWrapper(base, **wrapper_kwargs)
    env = RobosuiteTDMPCWrapper(env, max_episode_steps=env_kwargs["horizon"])
    return env
