# Running TD-MPC2 SO(3) sweeps on AWS SageMaker

A SageMaker training job is a batch job, not a machine you log into: you hand AWS
a container image + a config, AWS rents a GPU box by the second, runs the image
to completion, copies a few blessed directories to S3, and destroys the box.
These files make this repo run that way.

## What's here

| file | role |
| --- | --- |
| `sagemaker/train_entry.py` | Installed as the `train` executable in the image. Reads `/opt/ml/input/config/hyperparameters.json`, turns it into Hydra `key=value` overrides, and runs the unmodified `tdmpc2/train.py`. Also routes checkpoints/outputs to the SageMaker paths and enables `resume`. |
| `docker/Dockerfile.sagemaker` | Bakes the repo into the image and installs `train`. Build context is the **repo root**. |
| `sagemaker/launch.py` | Runs on your laptop. Submits one training job per `(task, rot_type, seed)` and returns immediately. |
| `.dockerignore` | Keeps the build context small. |

The `np.concat` → `np.concatenate` fix in `benchmarks/her_orient/` is unrelated
to SageMaker but required for the `her_orient` tasks to run in the image (its
numpy pin is < 2.0).

## One-time AWS setup (this part is you, not the code)

1. **AWS CLI + credentials.** `aws configure` (or SSO). Verify: `aws sts get-caller-identity`.
2. **GPU quota.** New accounts often have **zero** `ml.g5` quota. In Service
   Quotas, request an increase for `ml.g5.xlarge for training job usage` (and the
   spot variant if using spot). This can take a day — do it first.
3. **IAM execution role.** A role SageMaker assumes, with `AmazonSageMakerFullAccess`
   plus read/write on your S3 bucket and pull on your ECR repo. Note its ARN.
4. **S3 bucket** for checkpoints + outputs, e.g. `s3://my-bucket/tdmpc2`.
5. **W&B key** available locally: `export WANDB_API_KEY=...`

## Build and push the image to ECR

SageMaker cannot pull from Docker Hub or your laptop — the image must live in ECR.

```bash
ACCT=<your-account-id>; REGION=us-east-1
REPO=$ACCT.dkr.ecr.$REGION.amazonaws.com/tdmpc2

# create the ECR repo once
aws ecr create-repository --repository-name tdmpc2 --region $REGION

# log docker in to ECR
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCT.dkr.ecr.$REGION.amazonaws.com

# build from the REPO ROOT (note the trailing dot and -f path)
docker build -f docker/Dockerfile.sagemaker -t $REPO:sm .
docker push $REPO:sm
```

## Test the container locally first (highly recommended)

Debugging on SageMaker is a slow submit → wait ~4 min → read CloudWatch loop.
Catch mistakes on your laptop instead, using the exact command SageMaker runs:

```bash
# fake the hyperparameters file, then run `train` like SageMaker will.
# (needs a GPU + nvidia container runtime for a real run; without one it will
#  still exercise the shim and fail only at the CUDA assertion in train.py.)
mkdir -p /tmp/sm/input/config
echo '{"task":"her-reach-orient","rot_type":"matrix","seed":"1","model_size":"5","steps":"2000","enable_wandb":"false","compile":"false"}' \
  > /tmp/sm/input/config/hyperparameters.json

docker run --rm --gpus all \
  -v /tmp/sm/input/config:/opt/ml/input/config \
  $REPO:sm train
```

You should see `[entry] hydra overrides: ...` followed by TD-MPC2 starting up.

## Launch the sweep

Edit the `TASKS` / `ROT_TYPES` / `SEEDS` and `BASE_HYPERPARAMS` at the top of
`launch.py`, then:

```bash
export WANDB_API_KEY=...
python sagemaker/launch.py \
  --image-uri  <acct>.dkr.ecr.<region>.amazonaws.com/tdmpc2:sm \
  --role-arn   arn:aws:iam::<acct>:role/<SageMakerExecutionRole> \
  --bucket     s3://<your-bucket>/tdmpc2 \
  --wandb-project tdmpc2-so3 --wandb-entity <you>
```

Add `--dry-run` first to print the jobs without submitting. Add `--no-spot` to
use on-demand instead of (cheaper, interruptible) spot instances.

## How persistence & spot interruptions work

- `checkpoint_s3_uri` is unique per job and maps to `/opt/ml/checkpoints`, which
  SageMaker **live-syncs to S3 during the run**. If a spot instance is reclaimed,
  SageMaker restarts the *same job*, repopulates `/opt/ml/checkpoints` from S3,
  and the entry point's `resume=true` picks up where it left off. (The replay
  buffer is not checkpointed; on resume the trainer re-collects `seed_steps` of
  data before updating — expected.)
- Everything else (`logs/`, `models/final.pt`, `eval.csv`, videos) is written
  under `/opt/ml/output/data` and uploaded as `output.tar.gz` at job exit.
- W&B logs stream live regardless, since `WANDB_API_KEY` is passed to each job
  and network isolation is off.

## Common gotchas

- **Build context must be the repo root.** `rotations/` and `benchmarks/` live at
  the root and are needed at runtime. Building from `docker/` will omit them.
- **GPU instance required.** `train.py` asserts CUDA; use `ml.g5.xlarge` or larger.
- **`model_size` is mandatory.** It's `???` in `config.yaml`; always pass it.
- **First run:** set a small `steps` and `enable_wandb=false compile=false` to get
  a green job quickly before launching the full sweep.
