"""Smoke test for the robosuite SO(3) integration into TD-MPC2.

Run from the inner package dir (where train.py lives), on a machine with robosuite:

    cd tdmpc2/tdmpc2
    python test_robosuite_so3.py

Checks, for each rotation representation, that:
  - the action space resizes to `3 + RotType.dim` (+1 gripper),
  - reset() returns a flat float32 state tensor,
  - a few random steps run and return (obs, reward, done, info) with success/terminated.
"""

from envs.robosuite_so3 import make_env
from envs.wrappers.tensor import TensorWrapper
from rotations.rotations import RotType


class Cfg(dict):
    """Minimal stand-in for the hydra/OmegaConf cfg (attribute access + .get)."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def get(self, k, default=None):
        return dict.get(self, k, default)


def check(task, rot_type):
    cfg = Cfg(
        task=task,
        obs="state",
        rot_type=rot_type,
        rot_obs_type="matrix",
        rot_control_mode="rel",
        rot_step_len=None,
        save_video=False,
    )
    env = TensorWrapper(make_env(cfg))

    # Lift uses a single arm with a gripper: pos(3) + rot(dim) + gripper(1).
    expected = 3 + RotType(rot_type).dim + 1
    got = env.action_space.shape[0]
    assert got == expected, f"{task}/{rot_type}: action_dim {got} != expected {expected}"

    obs = env.reset()
    assert obs.dtype.is_floating_point and obs.ndim == 1, f"{task}/{rot_type}: bad obs {obs.shape}/{obs.dtype}"

    for _ in range(5):
        obs, reward, done, info = env.step(env.rand_act())
        assert "success" in info and "terminated" in info

    print(f"  OK  {task:24s} rot={rot_type:8s} action_dim={got:2d} obs_dim={obs.shape[0]:3d}")


def main():
    print("robosuite-lift (single Panda + gripper):")
    for rt in ["tangent", "euler", "quat", "matrix", "r6"]:
        check("robosuite-lift", rt)
    print("\nAll robosuite SO(3) smoke checks passed.")


if __name__ == "__main__":
    main()
