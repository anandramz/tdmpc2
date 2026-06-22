"""Smoke test for the her_orient SO(3) integration into TD-MPC2.

Run from the inner package dir (where train.py lives), on a machine with mujoco:

    cd tdmpc2/tdmpc2
    python test_her_orient_so3.py

Checks, for each rotation representation, that:
  - the action space resizes to `3 + RotType.dim` (+1 gripper for pick-and-place),
  - reset() returns a flat float32 state tensor,
  - a few random steps run and return (obs, reward, done, info) with success/terminated.
"""

import numpy as np

from envs.her_orient import make_env
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


def check(task, rot_type, gripper):
    cfg = Cfg(task=task, obs="state", rot_type=rot_type, rot_relative=False, rot_scale=None)
    env = TensorWrapper(make_env(cfg))

    base = 4 if gripper else 3  # pos(3) + gripper(1) if present
    expected = base + RotType(rot_type).dim
    got = env.action_space.shape[0]
    assert got == expected, f"{task}/{rot_type}: action_dim {got} != expected {expected}"

    obs = env.reset()
    assert obs.dtype.is_floating_point and obs.ndim == 1, f"{task}/{rot_type}: bad obs {obs.shape}/{obs.dtype}"

    for _ in range(5):
        obs, reward, done, info = env.step(env.rand_act())
        assert "success" in info and "terminated" in info

    print(f"  OK  {task:20s} rot={rot_type:8s} action_dim={got:2d} obs_dim={obs.shape[0]:2d}")


def main():
    print("ReachOrient (no gripper):")
    for rt in ["tangent", "euler", "quat", "matrix"]:
        check("her-reach-orient", rt, gripper=False)
    print("PickAndPlaceOrient (gripper):")
    for rt in ["tangent", "euler", "quat", "matrix"]:
        check("her-pnp-orient", rt, gripper=True)
    print("\nAll her_orient SO(3) smoke checks passed.")


if __name__ == "__main__":
    main()
