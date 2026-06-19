import numpy as np
import gymnasium as gym
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from envs.wrappers.timeout import Timeout


# Task registry

# Maps cfg.task strings to RoboSuite environment names

ROBOSUITE_TASKS = {
    "rs-lift":           "Lift",
    "rs-stack":          "Stack",
    "rs-nut-assembly":   "NutAssemblyRound",
    "rs-pick-place":     "PickPlace",
    "rs-door":           "Door",
    "rs-wipe":           "Wipe",
    "rs-two-arm-lift":   "TwoArmLift",
    "rs-peg-in-hole":    "TwoArmPegInHole",
    "rs-handover":       "TwoArmHandover",
}

# Tasks that require two robots in opposed configuration
TWO_ARM_TASKS = {"rs-two-arm-lift", "rs-peg-in-hole", "rs-handover"}

# Max steps per episode
MAX_EPISODE_STEPS = {
    "rs-lift":          500,
    "rs-stack":         500,
    "rs-nut-assembly":  500,
    "rs-pick-place":    500,
    "rs-door":          500,
    "rs-wipe":          500,
    "rs-two-arm-lift":  500,
    "rs-peg-in-hole":   500,
    "rs-handover":      500,
}

# Geodesic reward

def geodesic_reward(R1: np.ndarray, R2: np.ndarray) -> float:
    """Compute negative geodesic distance between two rotation matrices.

    Dense reward signal from the proposal (Eq. 2):
        d(R1, R2) = arccos((tr(R1^T R2) - 1) / 2)

    Returns the negative distance so that closer = higher reward.

    Args:
        R1: (3, 3) current rotation matrix.
        R2: (3, 3) target rotation matrix.

    Returns:
        Scalar reward in range (-pi, 0].
    """
    trace = np.trace(R1.T @ R2)
    cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
    return -float(np.arccos(cos_angle))


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix.

    Args:
        q: (4,) quaternion [x, y, z, w].

    Returns:
        (3, 3) rotation matrix.
    """
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float32)

# Wrapper

class RoboSuiteWrapper(gym.Env):
    """Wraps RoboSuite to match TD-MPC2's expected interface.
    """

    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg
        self.is_two_arm = cfg.task in TWO_ARM_TASKS

        # Dummy reset to compute spaces
        obs_dict = self.env.reset()
        self._target_R = self._get_target_rotation(obs_dict)
        flat_obs = self._get_obs(obs_dict)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=flat_obs.shape,
            dtype=np.float32,
        )
        low  = self.env.action_spec[0]
        high = self.env.action_spec[1]
        self.action_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

    def _get_eef_rotation(self, obs_dict: dict) -> np.ndarray:
        """Extract current end-effector rotation matrix from observation.

        Always uses robot0's end-effector (primary arm for two-arm tasks).

        Args:
            obs_dict: Raw RoboSuite observation dict.

        Returns:
            (3, 3) rotation matrix.
        """
        q = obs_dict["robot0_eef_quat"].copy()  # [x, y, z, w]
        return quat_to_matrix(q)

    def _get_target_rotation(self, obs_dict: dict) -> np.ndarray:
        """Get the target rotation for this task.

        Args:
            obs_dict: Raw RoboSuite observation dict.

        Returns:
            (3, 3) target rotation matrix.
        """
        # Default: identity rotation (upright end-effector)
        return np.eye(3, dtype=np.float32)

    def _get_obs(self, obs_dict: dict) -> np.ndarray:
        """Extract observation as flattened rotation matrix (vec(Rt) in R^9).

        Args:
            obs_dict: Raw RoboSuite observation dict.

        Returns:
            (9,) flattened rotation matrix as float32.
        """
        R = self._get_eef_rotation(obs_dict)
        return R.flatten().astype(np.float32)

    def reset(self, **kwargs):
        obs_dict = self.env.reset()
        self._target_R = self._get_target_rotation(obs_dict)
        return self._get_obs(obs_dict)

    def step(self, action):
        obs_dict, _, done, info = self.env.step(action)
        obs = self._get_obs(obs_dict)

        # Geodesic reward (Eq. 2 from proposal)
        current_R = self._get_eef_rotation(obs_dict)
        reward = geodesic_reward(current_R, self._target_R)

        return obs, reward, done, info

    def render(self, *args, **kwargs):
        return self.env.render(
            mode="rgb_array",
            height=384,
            width=384,
            camera_name="agentview",
        )

    @property
    def unwrapped(self):
        return self.env

# make_env

def make_env(cfg):
    """Make a RoboSuite environment for TD-MPC2.

    cfg.task options:
        "rs-lift"         -> Block Lifting       (single Panda)
        "rs-stack"        -> Block Stacking      (single Panda)
        "rs-nut-assembly" -> Nut Assembly Round  (single Panda)
        "rs-pick-place"   -> Pick and Place      (single Panda)
        "rs-door"         -> Door Opening        (single Panda)
        "rs-wipe"         -> Table Wiping        (single Panda)
        "rs-two-arm-lift" -> Two-Arm Lifting     (two Pandas, opposed)
        "rs-peg-in-hole"  -> Two-Arm Peg-in-Hole (two Pandas, opposed)
        "rs-handover"     -> Two-Arm Handover    (two Pandas, opposed)
    """
   
    if cfg.task not in ROBOSUITE_TASKS:
        raise ValueError(
            f"Unknown task: {cfg.task}. "
            f"Choose from: {list(ROBOSUITE_TASKS.keys())}"
        )

    assert cfg.obs == "state", "RoboSuite wrapper only supports state observations."

    env_name = ROBOSUITE_TASKS[cfg.task]

    if cfg.task in TWO_ARM_TASKS:
        robots = ["Panda", "Panda"]
        env_configuration = "opposed"
        robot_name = "Panda"
    else:
        robots = "Panda"
        env_configuration = "default"
        robot_name = "Panda"

    # BASIC composite controller uses OSC_POSE for the arm
    controller_config = load_composite_controller_config(
        controller="BASIC",
        robot=robot_name,
    )

    raw_env = suite.make(
        env_name=env_name,
        robots=robots,
        env_configuration=env_configuration,
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=False, 
        control_freq=20,
        ignore_done=False,
        hard_reset=False,
        seed=cfg.seed,
    )

    env = RoboSuiteWrapper(raw_env, cfg)
    max_steps = MAX_EPISODE_STEPS[cfg.task]
    env = Timeout(env, max_episode_steps=max_steps)

    return env
