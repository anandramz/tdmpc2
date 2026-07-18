#!/usr/bin/env python
"""SageMaker bring-your-own-container entry point for TD-MPC2.

SageMaker starts a training job by running `docker run <image> train` with no
further arguments. The per-job settings are NOT passed on the command line;
they arrive as a JSON dict written to `/opt/ml/input/config/hyperparameters.json`.
TD-MPC2's `train.py` is a Hydra script that reads overrides from the command
line in `key=value` form. This shim bridges the two:

    1. read the hyperparameters JSON,
    2. reformat each pair into Hydra's `key=value` syntax,
    3. rewrite `sys.argv` so Hydra sees them as if typed on the command line,
    4. run the existing `train.py` unchanged.

It also wires in the SageMaker filesystem contract (see the table below) so
outputs survive the job and spot-interruptions self-heal via `resume`.

    /opt/ml/input/config/hyperparameters.json  <- job settings (input)
    /opt/ml/checkpoints/                        -> live-synced to S3 during the run
    /opt/ml/output/data/                        -> uploaded as output.tar.gz at exit

Install this file as the executable `/usr/local/bin/train` in the image so that
`docker run <image> train` finds it (see docker/Dockerfile.sagemaker).
"""
import json
import os
import sys
from pathlib import Path

# Location of the repo baked into the image (see Dockerfile.sagemaker COPY).
# The inner package (train.py, config.yaml, common/, envs/, trainer/) lives one
# level down; `rotations/` and `benchmarks/` sit at the repo root and are added
# to sys.path by the env modules themselves (they resolve via parents[2]).
REPO_ROOT = Path(os.environ.get("TDMPC2_REPO", "/workspace/tdmpc2"))
INNER_PKG = REPO_ROOT / "tdmpc2"

# SageMaker filesystem contract.
HYPERPARAMS_FILE = Path("/opt/ml/input/config/hyperparameters.json")
CHECKPOINT_DIR = Path("/opt/ml/checkpoints")   # live-synced to S3 during the run
OUTPUT_DATA_DIR = Path("/opt/ml/output/data")  # -> output.tar.gz at job exit


def load_hyperparameters() -> dict:
    """Read SageMaker's hyperparameters JSON, or {} when run locally for testing."""
    if HYPERPARAMS_FILE.exists():
        return json.loads(HYPERPARAMS_FILE.read_text())
    print(f"[entry] no {HYPERPARAMS_FILE} (running outside SageMaker); using defaults")
    return {}


def normalize(value) -> str:
    """SageMaker stringifies every hyperparameter; make bools Hydra-friendly."""
    s = str(value)
    if s in ("True", "False"):        # Python-bool -> Hydra lowercase bool
        return s.lower()
    return s


def build_overrides(hp: dict) -> list:
    """Merge SageMaker-specific defaults with the job's hyperparameters.

    Defaults route persistence at the SageMaker paths and make the run
    resumable so a spot-interruption restart continues instead of restarting
    from scratch. Anything in `hp` wins, so a job can still override them.
    """
    merged = {
        # Persist checkpoints where SageMaker live-syncs to S3.
        "checkpoint_dir": str(CHECKPOINT_DIR),
        # Off by default in config.yaml; enable so interruptions are recoverable.
        "checkpoint_freq": 25000,
        # Safe even on a first run: load_checkpoint() no-ops and continues when
        # no checkpoint exists yet.
        "resume": "true",
    }
    merged.update(hp)  # job-supplied values override the defaults above
    return [f"{k}={normalize(v)}" for k, v in merged.items()]


def main() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # MuJoCo needs a headless GL backend on the GPU box (nvidia_icd.json is baked
    # into the image). train.py already defaults to egl, but set it explicitly.
    os.environ.setdefault("MUJOCO_GL", "egl")

    # Make the inner package importable (common/, envs/, tdmpc2/, trainer/).
    sys.path.insert(0, str(INNER_PKG))

    # cfg.work_dir is derived from Hydra's original cwd; run from a persisted
    # directory so logs/, models/, eval.csv, and videos land in output.tar.gz.
    os.chdir(OUTPUT_DATA_DIR)

    overrides = build_overrides(load_hyperparameters())
    print("[entry] hydra overrides:", " ".join(overrides))

    # Rewrite argv so Hydra reads these as command-line overrides, then run the
    # unmodified train.py (config.yaml is found relative to that file, not cwd).
    sys.argv = [str(INNER_PKG / "train.py")] + overrides

    from train import train  # noqa: E402  (import after sys.path/argv are set)
    train()


if __name__ == "__main__":
    main()
