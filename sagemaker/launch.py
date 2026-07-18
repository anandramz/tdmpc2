#!/usr/bin/env python
"""Launch TD-MPC2 SO(3) sweeps as SageMaker training jobs (run on your laptop).

This submits one SageMaker training job per (task, rot_type, seed) combination
and returns immediately -- the GPU boxes run in the cloud, report to W&B, and
self-heal on spot interruption via the resume path baked into the entry point.

Prerequisites (one-time, see sagemaker/README.md):
  * AWS CLI configured with credentials; GPU (ml.g5) quota approved.
  * The image built and pushed to ECR (docker/Dockerfile.sagemaker).
  * An IAM execution role ARN with S3 + ECR access.
  * WANDB_API_KEY set in your local environment (passed through to the jobs).

Usage:
  export WANDB_API_KEY=...
  python sagemaker/launch.py \
    --image-uri  <acct>.dkr.ecr.<region>.amazonaws.com/tdmpc2:sm \
    --role-arn   arn:aws:iam::<acct>:role/<SageMakerExecutionRole> \
    --bucket     s3://<your-bucket>/tdmpc2 \
    --wandb-project tdmpc2-so3 --wandb-entity <you>
"""
import argparse
import os
import re
import sys
from datetime import datetime, timezone

# ---- Edit these to define your sweep -------------------------------------
TASKS = ["her-reach-orient"]                       # e.g. add "robosuite-lift", "rot-orient"
ROT_TYPES = ["euler", "tangent", "quat", "matrix"] # r6 is robosuite/rot-orient only
SEEDS = [1, 2, 3]

# Base hyperparameters applied to every job (Hydra key=value overrides).
# checkpoint_dir / checkpoint_freq / resume are injected by the entry point.
BASE_HYPERPARAMS = {
    "model_size": 5,
    "steps": 1_000_000,
    "compile": "true",
    "save_video": "false",   # video needs wandb on + extra rendering; off by default
    "enable_wandb": "true",
}
# --------------------------------------------------------------------------


def job_name(task: str, rot_type: str, seed: int) -> str:
    """SageMaker job names: <=63 chars, alphanumeric and hyphens only, unique."""
    stamp = datetime.now(timezone.utc).strftime("%m%d-%H%M%S")
    raw = f"tdmpc2-{task}-{rot_type}-s{seed}-{stamp}"
    return re.sub(r"[^a-zA-Z0-9-]", "-", raw)[:63].strip("-")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image-uri", required=True, help="ECR image URI (…/tdmpc2:sm)")
    p.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    p.add_argument("--bucket", required=True, help="S3 prefix, e.g. s3://my-bucket/tdmpc2")
    p.add_argument("--wandb-project", required=True)
    p.add_argument("--wandb-entity", required=True)
    p.add_argument("--instance-type", default="ml.g5.xlarge")
    p.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    p.add_argument("--no-spot", action="store_true", help="use on-demand instead of spot")
    p.add_argument("--max-run-hours", type=float, default=48.0)
    p.add_argument("--dry-run", action="store_true", help="print jobs without submitting")
    args = p.parse_args()

    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if not wandb_key and not args.dry_run:
        sys.exit("WANDB_API_KEY is not set in your environment; export it first.")

    # Import lazily so --dry-run works without the sagemaker SDK installed.
    if not args.dry_run:
        import sagemaker
        from sagemaker.estimator import Estimator
        session = sagemaker.Session()

    bucket = args.bucket.rstrip("/")
    use_spot = not args.no_spot
    max_run = int(args.max_run_hours * 3600)

    combos = [(t, r, s) for t in TASKS for r in ROT_TYPES for s in SEEDS]
    print(f"Preparing {len(combos)} job(s): "
          f"{len(TASKS)} task(s) x {len(ROT_TYPES)} rot_type(s) x {len(SEEDS)} seed(s)")

    for task, rot_type, seed in combos:
        name = job_name(task, rot_type, seed)
        hyperparameters = {
            **BASE_HYPERPARAMS,
            "task": task,
            "rot_type": rot_type,
            "seed": seed,
            "wandb_project": args.wandb_project,
            "wandb_entity": args.wandb_entity,
            "exp_name": f"{rot_type}",
        }
        # Per-job checkpoint prefix so jobs never share checkpoints; SageMaker
        # repopulates /opt/ml/checkpoints from here when it restarts this job
        # after a spot interruption.
        ckpt_s3 = f"{bucket}/checkpoints/{task}/{rot_type}/seed{seed}"

        print(f"\n=== {name} ===")
        print(f"    hyperparameters: {hyperparameters}")
        print(f"    checkpoint_s3_uri: {ckpt_s3}")
        if args.dry_run:
            continue

        estimator = Estimator(
            image_uri=args.image_uri,
            role=args.role_arn,
            instance_count=1,
            instance_type=args.instance_type,
            sagemaker_session=session,
            output_path=f"{bucket}/output",
            checkpoint_s3_uri=ckpt_s3,          # <-> /opt/ml/checkpoints (live sync)
            checkpoint_local_path="/opt/ml/checkpoints",
            hyperparameters=hyperparameters,
            environment={"WANDB_API_KEY": wandb_key},
            use_spot_instances=use_spot,
            max_run=max_run,
            max_wait=max_run if use_spot else None,  # required when spot is on
            enable_network_isolation=False,          # jobs need to reach W&B
        )
        estimator.fit(job_name=name, wait=False)     # fan out; don't block
        print(f"    submitted (spot={use_spot})")

    if not args.dry_run:
        print("\nAll jobs submitted. Watch them in the SageMaker console / CloudWatch, "
              "or your W&B project.")


if __name__ == "__main__":
    main()
