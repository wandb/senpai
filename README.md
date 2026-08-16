<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# SENPAI

***[reseearch preview]*** -- Paper: [*SENPAI: Self-ExperimentatioN for Physical AI—An Observability-Based Research Harness*](https://openreview.net/forum?id=g0bJFA9gVT)

SENPAI is an semi-autonomous ML research loop - an `Advisor` proposes and reviews experiments; `Students` implement one assigned experiment each, train, and return evidence through GitHub and W&B. SENPAI works best with occasional steering from a domain-expert, steering happens by opening a Github Issue.

SENPAI is problem-agnostic. It runs against a separate target repository, and every experiment branch, commit, and PR lands there—not in this runner repository.


## News
- **[August 2026] MLX.fast challenge:** SENPAI places 7th in MLX.fast - an inference optimization speedup challenge on Mac for Poolside's Laguna 2.1 XS 30B LLM
- **[July 2026]   ICML 2026:** [*SENPAI: Self-ExperimentatioN for Physical AI—An Observability-Based Research Harness*](https://openreview.net/forum?id=g0bJFA9gVT) was presented at the AI for Science Workshop; see the [project site](https://wandb.github.io/senpai/).
- **[June 2026]   ICLR 2026:** Kagent, a SENPAI variant, placed fourth in the [GRaM competition](https://gram-competition.github.io/).

## Quick start

Kubernetes, local Docker, AWS GPU, and AWS Mac use the same launcher, role
configuration, training limits, GitHub workflow, and W&B record. This quick
start shows Kubernetes; see [Docker](#docker), [AWS](#aws), or
[AWS Mac](#aws-mac) for a host without a cluster.

### 1. Prerequisites

- Python 3.13, [uv](https://docs.astral.sh/uv/), and Git. Kubernetes launches
  also need `kubectl`.
- A Kubernetes context and existing namespace with outbound access to GitHub,
  Exa, W&B, and each configured model provider. Your identity must be able to
  get, list, create, update, patch, and delete Deployments, ConfigMaps, and
  Secrets there.
- An existing PVC with enough space for the dataset, plus concurrent mounts from every scheduled node—normally `ReadWriteMany`, unless your storage driver explicitly supports another multi-node topology. The launcher mounts this claim but does not create it; role state stays on each pod's node-local `emptyDir` volume.
- NVIDIA GPU nodes, the Kubernetes NVIDIA device plugin, and a host driver compatible with CUDA 13 and the shipped student image.
- A target GitHub repository that Senpai can clone and modify.
- Immutable advisor and student images reachable by every cluster node.

### 2. Install Senpai

```bash
git clone https://github.com/wandb/senpai.git
cd senpai
uv sync --locked
```

For development, include the test dependencies:

```bash
uv sync --locked --extra dev
```

### 3. Add credentials

```bash
cp example.env .env
```

Fill in the required values in the gitignored `.env` file. Model-provider keys
are required when any configured model uses that provider; `HF_TOKEN` is
optional:

```dotenv
GITHUB_TOKEN=
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
EXA_API_KEY=
WANDB_API_KEY=
HF_TOKEN=
```

| Credential | Required access |
|---|---|
| `GITHUB_TOKEN` | Target-repository Contents, Pull requests, and Issues read/write. A classic token with `repo` scope also works. GitHub CLI authentication is the fallback when this value is absent. |
| `ANTHROPIC_API_KEY` | Required when an `anthropic/...` model is configured. |
| `OPENAI_API_KEY` | Required when an `openai/...` model is configured. Every default profile uses GPT-5.6. |
| `EXA_API_KEY` | General-web and research-publication search. |
| `WANDB_API_KEY` | Read/write access to the configured W&B entity and project. |
| `HF_TOKEN` | Optional access to private or gated Hugging Face models and datasets. It is omitted from launch secrets when unset. |

`k8s/launch.py` reads shell environment variables first and then the
repository-root `.env`; only the GitHub token also falls back to `gh auth
token`.

Kubernetes stores runtime credentials in a per-launch Secret. Docker stores
them in private local run state; AWS sends them over SSH into the same private
run state on EC2. Termination removes that state. During role bootstrap, the
GitHub write token is removed from the process environment and handed to the
controller through a one-use channel; it is not exposed to the model or
subagents.

### 4. Prepare the target repository

The target repository needs one concise `program.md` describing the research
goal, metrics, data, constraints, allowed edits, and target-specific guidance.

A useful structure is:

- `## Mission` — Explain why the research is being run and name the primary target metric or metrics being optimized.
- `## Data` — Describe where the data lives, its type and structure, train/validation/test splits, and important nuances, exclusions, or caveats.
- `## Evaluation` — Define each evaluation metric concretely and, when using W&B, give its exact logged name, such as `val/loss` rather than "validation loss."
- `## Files` — List the key data-loading, preprocessing, training, scoring, and evaluation files, briefly describing each and whether agents may edit it.
- `## Research` — Add useful task or domain background and possible research directions without prescribing a narrow approach that limits the agents' creativity.

Put it at the repository root or one directory below it, such as
`senpai/program.md`. A blank `program_path` auto-discovers exactly one match in
those locations; otherwise set `program_path` in `senpai.yaml` or pass
`--program_path` at launch. Senpai appends the selected file to every agent's
system prompt. The built-in role charters need no target-specific prompt
overlay.

Use the [bootstrap-target guide](.agents/skills/bootstrap-target/SKILL.md) to
inspect a new target and create the file. Target repositories may also provide
project skills, but their `AGENTS.md`, `AGENT.md`, and `CLAUDE.md` files are not
loaded as Senpai project context.

The target repository must be different from the SENPAI runner repository.

### 5. Configure the launch

Copy the checked-in defaults and replace the W&B, branch, PVC, and resource values for your environment:

```bash
cp senpai.yaml senpai.local.yaml
```

The most important settings are:

```yaml
target_repo_branch: main
advisor_branch: senpai-research
program_path: ""  # auto-discover, or set e.g. senpai/program.md

wandb_entity: your-team
wandb_project: your-project

advisor_model: openai/gpt-5.6-sol
advisor_reasoning_effort: xhigh
student_model: openai/gpt-5.6-sol
student_reasoning_effort: xhigh

smart_model: openai/gpt-5.6-sol
smart_reasoning_effort: xhigh
fast_model: openai/gpt-5.6-luna
fast_reasoning_effort: high
frontier_model: openai/gpt-5.6-sol
frontier_reasoning_effort: max
local_condenser_max_events: 0
local_condenser_max_tokens: 0
local_condenser_target_events: 0

pvc_claim_name: your-existing-pvc
pvc_mount_path: /mnt/data

n_students: 4
gpus_per_student: 1
cpu_per_gpu: 8
memory_gi_per_gpu: 64

timeout_minutes: 30
max_epochs: 50
```

OpenHands uses LiteLLM, so LLM provider names are required as prefixes. For
example, configure Claude Fable 5 as `anthropic/claude-fable-5`. Anthropic
`reasoning_effort: max` on Claude Fable 5, Opus 5, and Sonnet 5 stays
provider-native and is sent as `output_config.effort: max`; it does not enable
OpenAI Pro mode.

W&B Inference uses LiteLLM's native `wandb/` provider. For example, set every
model profile to `wandb/zai-org/GLM-5.2` with reasoning effort `max`. Senpai
uses `WANDB_API_KEY`, routes requests through the W&B chat endpoint, explicitly
enables GLM thinking, and sends
`OpenAI-Project: <wandb_entity>/<wandb_project>` on every request.
Providers without native compaction use Senpai's local summarizing condenser.
Zero selects model-specific defaults for each of its three limits. Unknown
local models retain the existing 80-event fallback with token and target limits
disabled. W&B GLM-5.2 instead uses its exact `zai-org/GLM-5.2` chat-template
tokenizer, condenses at 180,000 input tokens, targets about 40 retained events,
and keeps 600 events as an emergency fuse. The 82,144-token margin below W&B's
262,144-token context window leaves room for a large tool result and the next
model response. OpenAI Responses and Anthropic keep their native compaction and
ignore all three local limits.

Every advisor root turn also receives a live controller invariant in its
non-condensed system context: the campaign is active, no campaign round limit
is configured, and round labels or compaction summaries cannot declare the
research finished. `max_turns` is identified explicitly as a per-turn safety
bound, not a research-completion counter.

Set positive values to override any dimension with
`local_condenser_max_events`, `local_condenser_max_tokens`, and
`local_condenser_target_events`; the target must leave room for the preserved
prefix and meaningful progress. The corresponding environment variables are
`SENPAI_OPENHANDS_LOCAL_CONDENSER_MAX_EVENTS`,
`SENPAI_OPENHANDS_LOCAL_CONDENSER_MAX_TOKENS`, and
`SENPAI_OPENHANDS_LOCAL_CONDENSER_TARGET_EVENTS`. AWS Mac preparation downloads
the GLM tokenizer once into `/Users/ec2-user/.senpai/huggingface`, verifies it
offline, and shares that cache with the roles' private home directories.

The defaults in `senpai.yaml` describe W&B's deployment and should not be copied unchanged into another environment. Every setting can also be overridden on the command line. `--tag` and `--target_repo_url` are required unless your chosen config file supplies them.

Deployments require one immutable image per active role, all built from the same
Senpai revision. Full `sha-<40-character-commit>` tags identify it directly;
digest-pinned role images require an explicit shared `senpai_repo_revision`.
Kubernetes and Docker fetch that commit from `senpai_repo_url`. The public
default is read-only and needs no PR permission; override it only for images
built from another Senpai repository. AWS transfers the exact clean local
commit instead. `target_repo_url` is the separate repository where agents
create commits and PRs. PR image checks build but do not publish images.

### 6. Run preflight

```bash
revision=$(git rev-parse HEAD)
launch_args=(
  --config_path senpai.local.yaml
  --tag first-run
  --target_repo_url https://github.com/OWNER/TARGET.git
  --advisor
  --names frieren
  --advisor_image "ghcr.io/wandb/senpai-advisor:sha-$revision"
  --student_image "ghcr.io/wandb/senpai-student:sha-$revision"
)

uv run python k8s/launch.py "${launch_args[@]}" --preflight_only
```

Every preflight authenticates GitHub, Exa, W&B, and every model provider used by
an active role or shared model profile. It verifies GitHub Contents write
access, resolves the target branch, rejects active student-label collisions,
and checks immutable image references against one expected runner revision.
Backend-specific checks are described below. Preflight never changes GitHub or
starts advisor or student roles.

### 7. Launch

For a Senpai commit whose images have been published, inspect the preflight
result and then launch with the same arguments:

```bash
uv run python k8s/launch.py "${launch_args[@]}"
```

The launcher creates routing labels, one launch Secret, role ConfigMaps, and Deployments. It does not create the namespace, PVC, Service, or general cluster RBAC.

Inspect and stop the launch:

```bash
kubectl get deployments,pods -l research-tag=first-run
kubectl logs -f deployment/senpai-first-run-frieren
kubectl delete deployments,configmaps,secrets -l research-tag=first-run
```

Use `--kube_context` and `--namespace` when the desired cluster is not your current default. Use `--dry_run` to render redacted manifests without checking credentials or writing to the cluster.

## Docker

Docker uses the same role specification and in-container `pvc_mount_path` as
Kubernetes. It requires a Linux host with Docker Engine, Git, the NVIDIA
Container Toolkit, a compatible NVIDIA driver, and enough GPUs for all
students. Docker Desktop on macOS cannot run the GPU student image.

```bash
revision=$(git rev-parse HEAD)
uv run python k8s/launch.py \
  --config_path senpai.local.yaml \
  --backend docker \
  --tag first-docker-run \
  --target_repo_url https://github.com/OWNER/TARGET.git \
  --advisor \
  --names fern \
  --data_dir /srv/ml-data \
  --advisor_image "ghcr.io/wandb/senpai-advisor:sha-$revision" \
  --student_image "ghcr.io/wandb/senpai-student:sha-$revision"
```

This first run uses one student; omit `--names fern` to use the shared default
of four. `data_dir` is mounted at `pvc_mount_path`, so target code and data paths
do not change between Kubernetes and Docker. Preflight checks the local images,
source revision, Docker, CUDA, GPU allocation, state directory, and
container-name collisions before changing GitHub. Runtime state lives under
`~/.senpai/runs/<tag>`.

```bash
uv run python k8s/docker.py status first-docker-run
uv run python k8s/docker.py logs first-docker-run --role student-fern --follow
uv run python k8s/docker.py terminate first-docker-run
```

## AWS

AWS creates one temporary, public, on-demand EC2 GPU host and runs the Docker
backend on it. All students share that host. The target repository, role
settings, data path, and agent workflow remain identical to Kubernetes.

### AWS setup

Install AWS CLI v2 and OpenSSH. The selected AWS identity needs
`sts:GetCallerIdentity`, `ssm:GetParameter`, EC2 `Describe*` access for instance
types, images, networking, instances, security groups, volumes, and volume
modifications, plus `CreateKeyPair`, `DeleteKeyPair`, `CreateSecurityGroup`,
`AuthorizeSecurityGroupIngress`, `RevokeSecurityGroupIngress`,
`RevokeSecurityGroupEgress`, `DeleteSecurityGroup`, `RunInstances`,
`TerminateInstances`, `CreateTags`, `ModifyVolume`, and
`ec2:GetConsoleOutput`. The region needs GPU quota and a public subnet with an
internet-gateway route.

For AWS SSO:

```bash
aws configure sso --profile senpai
aws sso login --profile senpai
aws sts get-caller-identity --profile senpai
```

If the session expires, run `aws sso login --profile senpai` again before a
status, logs, or termination command. AWS credentials remain on the operator
machine and are never copied to EC2. Runtime credentials still come from the
gitignored `.env` described above; Senpai sends them over SSH into private
remote Docker state. Local AWS state contains only lifecycle identifiers and
the ephemeral SSH key.

The launcher authenticates each new host before its first SSH connection. The
guest publishes its OpenSSH public host keys to the EC2 serial console during
boot; Senpai retrieves them through the authenticated AWS control plane,
validates their encoding and embedded key type, and pins them in the run's
private `known_hosts` file. SSH uses strict host-key checking. If AWS console
output does not provide a valid key before the readiness deadline, launch
fails before Senpai uploads runtime credentials or data.

AWS currently needs anonymously pullable advisor and student images. Use
matching `:sha-<40-character-commit>` tags, or digest-pinned images with an
explicit matching `senpai_repo_revision`. Commit local changes before
launching: AWS requires a clean checkout whose `HEAD` exactly matches that
revision.

### Preflight and launch

This first run uses one student to control cost. Omit `--names fern` to use the
shared default of four one-GPU students.

```bash
profile=senpai
region=us-east-2
tag=first-aws-run
revision=$(git rev-parse HEAD)

launch_args=(
  --config_path senpai.local.yaml
  --backend aws
  --tag "$tag"
  --target_repo_url https://github.com/OWNER/TARGET.git
  --advisor
  --names fern
  --aws_profile "$profile"
  --aws_region "$region"
  --aws_ttl_hours 8
  --data_dir /absolute/path/to/data-root
  --advisor_image "ghcr.io/wandb/senpai-advisor:sha-$revision"
  --student_image "ghcr.io/wandb/senpai-student:sha-$revision"
)

uv run python k8s/launch.py "${launch_args[@]}" --preflight_only
```

Verify the printed account, region, instance shape, AMI, subnet count, and
volume before creating anything. Then launch with the same arguments:

```bash
uv run python k8s/launch.py "${launch_args[@]}"
```

AWS preflight is read-only. It checks the account, local source revision,
credentials, AMI, actual GPU/vCPU/memory shape, SSH boundary, and eligible
public subnets. Image pulls and CUDA validation require the temporary host, so
EC2 billing has started, but they still happen before data upload or GitHub
mutation.

The contents of `data_dir` appear at `pvc_mount_path`. For code that reads
`/mnt/data/datasets/...` with the configuration above, pass the directory whose
child is `datasets/`. The directory must be non-empty and contain no symlinks.
Senpai streams it over SSH and verifies its file count and bytes. The initial
`aws_volume_gib` is a bootstrap floor for the AMI and image-pull peak; after
image validation, Senpai measures free space and grows the root EBS volume for
the actual dataset plus `aws_runtime_reserve_gib`.

Usually only `aws_profile`, `aws_region`, `data_dir`, and `aws_ttl_hours` need
attention. Leave the others empty unless your AWS environment has known
resources:

| Setting | When to set it |
|---|---|
| `aws_instance_type` | Pin an available x86_64 NVIDIA type. Senpai validates its real GPU, vCPU, and memory shape; otherwise it selects the first fitting type from its catalog. |
| `aws_subnet_id` | Pin a known public subnet. Otherwise Senpai searches eligible subnets across AZs and retries recognized capacity failures. |
| `aws_ami_id` | Pin the supported Ubuntu 22.04 NVIDIA Deep Learning Base AMI; otherwise Senpai resolves the current regional image. |
| `aws_ssh_cidr` | Pin the operator's IPv4 `/32`; otherwise Senpai discovers the current public IP. |
| `aws_volume_gib` | Increase the initial AMI/image-pull space when using unusually large images. |
| `aws_runtime_reserve_gib` | Change the free space retained across all roles for worktrees, checkpoints, caches, and logs. |
| `aws_ready_timeout_s`, `aws_data_timeout_s` | Increase host-start or dataset-upload deadlines on slower environments. |

### Operate and clean up

```bash
uv run python k8s/aws.py status first-aws-run
uv run python k8s/aws.py logs first-aws-run --role student-fern --tail 200
uv run python k8s/aws.py logs first-aws-run --role advisor --tail 200
uv run python k8s/aws.py terminate first-aws-run
```

Always run `terminate`. It attempts to gracefully stop the agents, terminates
EC2, and removes the ephemeral key pair, security group, local SSH key, and
lifecycle state. Launch failures attempt the same cleanup automatically.

`aws_ttl_hours` is an emergency cost backstop measured from instance startup,
not a replacement for cleanup. If it fires, run `terminate` afterward to remove
the remaining key, security group, and local state. The current AWS backend
intentionally owns one disposable public host; it does not yet support Spot,
multiple nodes, existing-host reuse, private-subnet/SSM transport, private image
registries, or IAM instance profiles.

## AWS Mac

AWS Mac reuses an already allocated fleet of `mac-m4pro.metal` Dedicated Hosts
and runs Senpai natively under per-user `launchd` services. It assigns exactly
one student to each host and co-locates the optional advisor on the first host.
Fresh mode creates and terminates EC2 instances. Reuse mode adopts an exact
fleet of already-running instances without stopping or terminating them. The
backend never releases Dedicated Hosts.

Use a clean checkout at the exact committed revision being launched. The base
macOS AMI does not contain full Xcode or the Metal toolchain. Provide either a
local Xcode app or a prepared `ditto` zip, plus a zip containing exactly one
Metal `*.exportedBundle`. Launch imports that local artifact instead of relying
on Apple's component catalog. Each selected host also needs a public subnet in
its own Availability Zone and an existing base security group in the same VPC.
The launcher attaches that group unchanged and creates a second,
campaign-owned security group for the operator's temporary SSH `/32`. This
keeps concurrent campaigns from revoking each other's access. Senpai removes
and verifies the new group's default IPv4/IPv6 egress, allowing bounded AWS
eventual-consistency retries, before launching an instance. It fails closed if
egress remains, so attaching the group cannot broaden the base group's
outbound policy.

Package the directory produced by Xcode's component export without changing
its bundle layout:

```bash
ditto -c -k --sequesterRsrc --keepParent \
  /path/to/MetalToolchain-17F109.exportedBundle \
  /path/to/MetalToolchain.zip
```

```bash
tag=first-aws-mac-run
revision=$(git rev-parse HEAD)

launch_args=(
  --config_path senpai.local.yaml
  --backend aws-mac
  --tag "$tag"
  --target_repo_url https://github.com/OWNER/TARGET.git
  --advisor
  --n_students 2
  --gpus_per_student 1
  --senpai_repo_revision "$revision"
  --aws_region us-east-1
  --aws_instance_type mac-m4pro.metal
  --aws_mac_host_ids h-HOST1,h-HOST2
  --aws_mac_subnet_ids us-east-1a=subnet-A,us-east-1b=subnet-B
  --aws_mac_security_group_id sg-EXAMPLE
  --aws_mac_xcode_archive /absolute/path/to/Xcode.zip
  --aws_mac_metal_toolchain_archive /absolute/path/to/MetalToolchain.zip
  --aws_mac_yukon_bundle "$HOME/.local/share/yukon/yukon.js"
  --aws_mac_official_submit
  --aws_ttl_hours 0
)

uv run python k8s/launch.py "${launch_args[@]}" --preflight_only
uv run python k8s/launch.py "${launch_args[@]}"
```

### Adopt running Mac instances

Reuse mode takes the complete fleet and its existing SSH access from a
schema-v1 manifest. Access files must be nonempty regular files, not symlinks,
and not writable by group or other users; the private key must additionally be
inaccessible to them. Every student, instance, host, placement, and
security-group set is exact; mixed created/adopted fleets are not supported.

```yaml
schema_version: 1
access:
  private_key_path: /absolute/path/to/id_ed25519
  known_hosts_path: /absolute/path/to/known_hosts
  ownership: external
nodes:
  - student: fern
    source:
      adopted_instance_id: i-0123456789abcdef0
    expect:
      host_id: h-0123456789abcdef0
      availability_zone: us-east-1a
      subnet_id: subnet-0123456789abcdef0
      security_group_ids:
        - sg-0123456789abcdef0
    prior_native_run: prior-fern-run
```

```bash
tag=adopted-aws-mac-run
revision=$(git rev-parse HEAD)

launch_args=(
  --config_path senpai.local.yaml
  --backend aws-mac
  --tag "$tag"
  --target_repo_url https://github.com/OWNER/TARGET.git
  --advisor
  --names fern,tanjiro
  --gpus_per_student 1
  --senpai_repo_revision "$revision"
  --aws_region us-east-1
  --aws_instance_type mac-m4pro.metal
  --aws_mac_bootstrap_mode reuse
  --aws_mac_nodes_path /absolute/path/to/aws-mac-nodes.yaml
  --aws_mac_yukon_bundle "$HOME/.local/share/yukon/yukon.js"
  --aws_mac_official_submit
  --aws_ttl_hours 0
)

uv run python k8s/launch.py "${launch_args[@]}" --preflight_only
uv run python k8s/launch.py "${launch_args[@]}"
```

Reuse preflight makes only read calls to AWS. It proves that every instance is
running on the manifest's exact Dedicated Host, AZ, subnet, VPC, and security
groups, then connects with the imported host keys under
`StrictHostKeyChecking=yes`. Guest IMDSv2 must report the expected instance ID,
the reusable Xcode/Metal/Homebrew/Chromium tools must be present, and every
LaunchDaemon recorded by `prior_native_run` must already be unloaded. A
changed public IP fails closed until access is authenticated out of band.

Reuse requires `--aws_ttl_hours 0` and rejects the fresh-placement flags
`--aws_mac_host_ids`, `--aws_mac_subnet_ids`, and
`--aws_mac_security_group_id`. It does not upload Xcode or Metal and does not
overwrite prior source, virtualenv, or native-run roots. Launch uploads and
installs the selected submission CLI: Yukon when `--aws_mac_yukon_bundle` is
set, otherwise the backward-compatible MLXFast bundle. It leaves the unselected
CLI untouched. Each campaign gets an isolated source and virtualenv below
`~/.senpai/aws-mac-runners/<tag>/`. The external key and `known_hosts` remain
untouched; only private copies are placed in the new lifecycle directory.

Fresh preflight is read-only: it validates the exact source revision, Apple
Silicon AMI, host availability and capacity, subnet placement, base security group,
Xcode source, and Metal toolchain archive. Launch creates an ephemeral SSH key
and a campaign-owned SSH security group, permits the operator's IPv4 `/32`
only on that group, validates one native canary,
prepares the remaining Macs in parallel, and holds every role at a fleet-wide
start gate before opening it. Each Mac uses the same authenticated EC2-console
host-key pinning and strict, fail-closed SSH policy described above, including
when `--aws_ttl_hours 0` disables scheduled shutdown.
Host preparation installs tmux and Chromium and smoke-tests both with a fresh
role-like `HOME`, matching the private-home environment used by native roles.
Each native role also has a private, shortened tmux socket root under
`~/.senpai/t`. This keeps macOS Unix-socket paths within the platform limit
while giving co-located roles and their recursive subagents separate tmux
namespaces. All native roles still run under the configured Unix user; this is
namespace isolation, not a hostile-process security boundary.
With `--aws_mac_official_submit`, the launcher also requires
`YUKON_API_TOKEN` when `--aws_mac_yukon_bundle` is set, or the legacy
`MLXFAST_API_TOKEN` otherwise, and gives every active role official dispatch
capability.
Coordinate submissions so students send distinct, validated candidates rather
than duplicate jobs.

AWS Mac alone accepts `--aws_ttl_hours 0` to omit the scheduled instance
shutdown and make guest-initiated shutdowns stop rather than terminate the
instance; a positive value retains the automatic termination backstop. Zero
does not change explicit termination or release the Dedicated Hosts, so monitor
the fleet and run `terminate` manually. The GPU AWS backend continues to
require a positive TTL.

```bash
uv run python k8s/aws_mac.py status first-aws-mac-run
uv run python k8s/aws_mac.py logs first-aws-mac-run --role advisor
uv run python k8s/aws_mac.py logs first-aws-mac-run --role student-fern
uv run python k8s/aws_mac.py terminate first-aws-mac-run
```

Termination unloads only the new native tag and removes its private state.
Fresh runs terminate only instances explicitly recorded as `created`, then
delete their ephemeral key and campaign-owned SSH security group. Reuse runs
remove only their tag-scoped runner and local access copies; adopted instances,
external keys, security groups, Dedicated Hosts, prior native roots, and other
networking resources are preserved. The selected global submission CLI remains
installed on adopted Macs for subsequent runs. Missing or invalid ownership
stops cleanup before any AWS or remote mutation and retains lifecycle state for
operator reconciliation. Cleanup also retains access while a created instance outcome
is unresolved and persists partial failures for a safe retry. Runs created by
the older shared-rule launcher never revoke that shared rule automatically;
cleanup preserves their state with an explicit manual-review message while the
exact rule remains. Remove it only when safe, then rerun `terminate` so cleanup
can verify its absence and finish.

## Experiment workflow

GitHub is both the coordination layer and the durable scientific notebook. W&B is the metric and artifact record.

```mermaid
flowchart LR
    H["Advisor records hypothesis, baseline, and acceptance rule"]
    P["Typed draft PR<br/>student:name + status:wip"]
    I["Student implements and commits"]
    T["Supervised jobs<br/>optional W&B metrics"]
    R["Structured result<br/>status:review"]
    D["Advisor merges, closes, requests a revision, or sends feedback"]

    H --> P --> I --> T --> R --> D
    D -->|revision| I
```

1. The advisor creates a falsifiable assignment with the exact required research-base SHA, baseline metrics, expected mechanism, implementation scope, and stopping rules.
2. `create_assignment` creates the student branch and draft PR, embeds a typed assignment record, and applies the routing labels.
3. The assigned student receives one OpenHands conversation for that assignment revision. New PR comments and reviews are queued durably even while a turn is active, then delivered in the next bounded turn.
4. The student commits the exact implementation, launches supervised training, and records every referenced run in W&B.
5. The student calls `submit_experiment_result`; the tool validates and publishes the branch before changing the PR to `status:review`.
6. The advisor compares the evidence, then uses the corresponding operation-specific tool to merge a reproducible winner, close a useful negative result, request a new revision, or send non-revision feedback.

The structured result records its terminal status, exact result commit, W&B run IDs and URLs, bounded conclusion, and baseline/candidate metric comparison when available. Once published for an assignment revision and head, that evidence is immutable: exact duplicate publication is an idempotent replay, while changed evidence requires a new commit or revision. Non-revision feedback continues the same student conversation; a revision request intentionally creates a fresh revision identity and conversation.

`status:wip` owns a student compute slot; `status:review` does not. The advisor can therefore review one result while that student starts another experiment. Sibling assignment mutations within one worker are serialized end to end, including advisor-base publication and student preflight, push, and result publication. Across advisor and student workers, exact assignment, revision, head, and branch-lease preconditions detect stale work; if a revision wins during result publication, SENPAI restores the current revision's WIP routing before returning the stale-result error.

Trusted collaborator comments, submitted reviews, and inline review comments are delivered automatically to the relevant student; feedback from untrusted authors and unrecognized bots is ignored. `get_prs` can still retrieve the complete discussion explicitly. If the configured research base changes while an experiment is running, SENPAI emits `research_base_changed` with the assignment's `required_base_sha` and the live `current_base_sha` without cancelling the assignment. When reviewing its terminal result, the advisor either requests a revision on the current base or records why that exact result remains valid with `accept_result_on_current_base`; `merge_experiment` still verifies the live SHA immediately before merging.

Before each assignment or PR-feedback turn, the student controller authenticates
and hydrates the exact assignment head and recorded baseline into
`refs/senpai/assignment/` without resetting the checkout. A revision can
atomically record an explicitly accepted live baseline SHA. Divergent local
history and dirty work are preserved and reported once; an unchanged assignment
reminder does not repeatedly wake the model, while changed refs, local work, or
new feedback still do.

`get_prs` returns complete PR bodies and discussions. Up to five PRs are returned in context by default; larger selections become a Markdown artifact outside the target checkout so long histories do not pollute the main conversation.

## Long-running jobs and monitoring

Advisor and student root conversations receive four typed tools for durable,
non-blocking process supervision. Students use them for GPU work; advisors can
use them for repository-side watchers or other bounded long-running commands.

| Tool | Contract |
|---|---|
| `run_job` | Accepts structured `argv`, `cwd`, a hard timeout, a `read_only` or `mutable` workspace-access declaration, and an optional W&B credential grant. It starts a supervised process group without blocking, persists its identity, full log, and bounded error tail, discovers W&B run IDs, and automatically registers terminal-state monitoring for the current conversation. Suitable work includes training, inference, evaluation, builds, and receipt watchers. |
| `get_job_status` | Performs one bounded read of the latest persisted state, exit code, elapsed time, W&B run IDs, and error tail. |
| `monitor_job` | Sets or replaces optional W&B policy for an already-running job: metric, minimize/maximize direction, `lte`, `gte`, `improved_by`, or `regressed_by` gates, poll interval, and stale-update detection. It cannot disable terminal wakes. |
| `cancel_job` | Stops the complete process group through the supervised TERM/KILL path, waits for a durable terminal state, and retires its monitor. |

After launch, the role can finish its turn. The deterministic controller polls
process state and at most one selected W&B metric without consuming model
tokens. A threshold crossing, regression, stale metric, terminal state, or
monitor error creates one compact durable event and resumes the same
conversation. Due monitors are processed in bounded batches with a time budget;
a slow or broken monitor can delay its batch only within those bounds and cannot
prevent later jobs, GitHub feedback, or child-agent results from being
processed. Monitor policy and ownership have one durable SQLite source of truth.
While any monitor is active, the controller sleeps only until its earliest due
poll rather than the ordinary advisor/student heartbeat.

`workspace_access="mutable"` is the default for builds, training, evaluation,
or any command that can write in the checkout. A student must have a clean
worktree before launching such a job, and controller-driven branch changes wait
until it finishes. `read_only` is reserved for passive watchers and does not take
that lease. A job receives no ambient credentials; request `WANDB_API_KEY` in
`secret_env` only when it actually communicates with W&B.

`improved_by` and `regressed_by` compare with the monitor policy's first observed sample; they do not silently reuse the assignment's documented baseline.

Worker and container restarts preserve completed OpenHands events. A recovered
live job is terminated safely rather than adopted under an unverifiable process
identity; its original conversation receives the persisted terminal outcome.

Interactive browser operations are progressively disclosed. A fresh root
conversation initially sees only `load_browser`; invoking it adds the fourteen
OpenHands browser operations and records the choice in conversation state so a
resumed conversation restores them. `--no-browser` exposes neither the loader
nor the browser family.

`task_tracker` is optional persisted working memory for multi-step work, parallel
workstreams, delegated agents, and long-running jobs; several items may be
`in_progress` when the work is genuinely concurrent. The legacy `think`
scratchpad tool is not exposed to any Senpai root or child—the selected models'
native reasoning remains enabled.

## Subagents

`spawn_agents` launches a batch and immediately returns stable task IDs;
`await_agents` collects them with an `all`, `first`, `quorum`, or any-state
`change` join; `change` also surfaces an uncollected terminal result
immediately. A timed-out wait returns current state and suggests non-blocking
next steps; it does not cancel the children. Every child runs in a fresh
OpenHands conversation and separate process group.

| Agent | Best for | Recommended tier |
|---|---|---|
| [General Purpose](.agents/agents/general-purpose.md) | Bounded work combining terminal investigation, code editing, task tracking, tests, and one controlled level of leaf delegation. | `smart` for ordinary implementation or review; `frontier` for the hardest generalist work. |
| [Explore](.agents/agents/explore.md) | Read-only search across code, data, experiment artifacts, papers, or durable conversation history. It returns conclusions with paths and line numbers rather than dumping source. | `fast` for mechanical exploration; `smart` when relationships are subtle. |
| [Search](.agents/agents/search.md) | External research through Exa via the explicit `search_general_web` or `search_research_publications` task form, with primary-source links. | `smart`. |
| [Bash Runner](.agents/agents/bash-runner.md) | Tests, builds, linters, dependency commands, Git inspection, and noisy CLI work. It returns counts and actionable failures rather than raw logs. | `fast`. |

The model tier is independent of the agent specialization. With the default
`agent=general-purpose`, `model=frontier` launches GPT-5.6 Sol at `max`, sent
to the Responses API with `reasoning.mode: pro`
with the general-purpose terminal and code-editing toolset. Pair `frontier`
with `search_general_web` or `search_research_publications` when the hard task
is external research.

A root spawn batch and its descendants form one delegation tree, which may
create at most eight children total. A role runs at most eight active tasks
concurrently across all trees. Root tasks count toward the tree total, so leave
slots when a General Purpose child needs helpers. Recursion is limited to two
child edges: the root may spawn any agent, and a depth-one General Purpose
child may spawn leaf helpers; Explore, Search, Bash Runner, and all depth-two
children cannot delegate. The tree shares one absolute root-turn deadline, and
a nested child must await or cancel all of its helpers before returning.
Individual tasks are capped at ten minutes for `fast`, thirty for `smart`, and
one hour for `frontier`, shortened when the root deadline is nearer.

An await call is capped at five minutes and does not cancel unfinished work.
`agent_status` provides a non-blocking snapshot; with no task IDs, it returns
up to eight direct tasks that are active or have an uncollected terminal result.
`cancel_agents` records terminal cancellation. Atomic records keyed by the
required batch key and each optional task key (or stable list index) make replay
return the original task IDs instead of spawning duplicates.
`include_context=false` sends only the system prompt and task; the child can
still search the supplied parent-history directory. `include_context=true`
also copies the model-visible parent history. The root advisor or student may
leave useful tasks running and receives their terminal results as durable
events; nested children may not detach descendants.

Children share the parent workspace, so their process and conversation are isolated but their filesystem is not. They receive only their declared tools and never receive GitHub credentials, GitHub workflow tools, or job tools.

## Task guides

Live advisors and students load Senpai-owned skills only from
`plugins/senpai/skills`; target repositories may also supply project skills.
Guides under `.agents/skills` are for human users and Senpai developers and
are not installed into autoresearch pods.

### Live autoresearch

| Guide | Purpose |
|---|---|
| [Assign an experiment](plugins/senpai/skills/assign-experiment/SKILL.md) | Turn a hypothesis into a typed student branch and draft PR. |
| [Delegate subagents](plugins/senpai/skills/delegate-subagents/SKILL.md) | Launch and coordinate bounded parallel research, review, and implementation help. |
| [Submit experiment results](plugins/senpai/skills/submit-experiment-results/SKILL.md) | Commit the tested implementation and publish a structured, evidence-backed result. |
| [Review an experiment](plugins/senpai/skills/review-experiment/SKILL.md) | Merge a reproducible winner, close a useful negative, or request the missing evidence. |
| [Handle human Issues](plugins/senpai/skills/check-human-issues/SKILL.md) | Respond to authenticated human-to-agent messages delivered through GitHub Issues. |
| [Senpai status check](plugins/senpai/skills/senpai-status-check/SKILL.md) | Produce a bounded, read-only GitHub, W&B, and local-controller status report. |
| [Exa search](plugins/senpai/skills/exa-search/SKILL.md) | Search the current web or scholarly publications with mode-specific defaults. |
| [AlphaXiv paper lookup](plugins/senpai/skills/alphaxiv-paper-lookup/SKILL.md) | Get a structured overview before reading a primary paper deeply. |
| [W&B and Weave](plugins/senpai/skills/wandb-primary/SKILL.md) | Inspect runs, metrics, artifacts, evaluations, and agent traces. |

### Human and developer guides

| Guide | Purpose |
|---|---|
| [Bootstrap a target](.agents/skills/bootstrap-target/SKILL.md) | Build `program.md` from a new ML repository. |
| [Experiment report](.agents/skills/experiment-report/SKILL.md) | Create the project-standard `nn_cfd` W&B comparison report; this guide is target-specific rather than part of the generic runtime. |
| [Training code style](literature_and_guidance/TRAINING-CODE-STYLE.md) | Structure expensive ML entrypoints so configuration, artifacts, validation, and failure boundaries stay explicit. |

The repository also contains two reusable optimization case studies:

- [LLM inference optimization](literature_and_guidance/LLM-INFERENCE-OPTIMIZATION-SENPAI-GUIDE.md)
- [LLM training optimization](LLM-TRAINING-OPTIMIZATION-GUIDE.md)

## Architecture and durability

```mermaid
flowchart LR
    GH["GitHub<br/>PR and Issue state"]
    WB["W&B<br/>runs and metrics"]
    A["Advisor<br/>controller + OpenHands"]
    S["Students<br/>controller + OpenHands + GPU"]

    A <--> GH
    S <--> GH
    A --> WB
    S --> WB
```

GitHub PR labels, typed comments, reviews, and human-tagged Issues are the only advisor/student communication protocol; W&B is the shared experiment store. Role-local SQLite stores the ordered delivery inbox and its receipts plus training-monitor policies; it is never shared across nodes.

Each role runs a small Python supervisor around the deterministic controller:

```text
entrypoint
  clone and configure
  exec supervisor

supervisor
  restart crashed workers with bounded backoff
  terminate and restart an overdue phase

controller
  poll -> reconcile -> bounded OpenHands turn -> verify -> acknowledge -> sleep
```

The controller owns cadence, durable events, conversation selection, verified GitHub operations, process supervision, and monitoring. OpenHands owns research judgment, code changes, and evidence interpretation.

- The advisor keeps one conversation UUID under the pod-local `/var/lib/senpai/<tag>/advisor/openhands_state`; it survives controller and container restarts within that pod.
- A student uses one UUID per assignment revision; feedback, monitor events, and child-task results resume that exact conversation.
- Still-actionable GitHub state is re-delivered on the configured reminder cadence, which defaults to at least ten minutes even when GitHub is polled more frequently. Immediate post-turn polls deliver changed state but not timed reminders, so a successful research-only turn cannot enter a no-sleep reminder loop. `research_base_changed` is keyed by assignment, revision, PR head, and the exact required/current base pair; each identity or base movement requires a new decision. Merge repeats the live-base check immediately before its mutation, while external base writers still require strict up-to-date branch protection or a merge queue for an atomic guarantee.
- Each model request gets one bounded 15-minute attempt. Foreground terminal calls return control within ten minutes for explicit continuation, the whole turn retains its one-hour hard lease, and two consecutive failed turns exit to the supervisor for a clean worker restart. Restart backoff grows across failed workers to a five-minute ceiling; only a successfully acknowledged turn resets that streak, not process uptime or idle sleep.
- Every controller prompt, GitHub event, monitor signal, and child result follows one durable `pending -> delivered -> processed` inbox. A provider failure resumes the already-delivered turn without resending it, and a crash after inference performs mailbox acknowledgement without another model call. New events wait behind an unresolved turn in the same conversation; normal drains are FIFO and bounded to 16 events or 64 KiB, while ready conversations take fair turns.
- A newly completed tool observation renews the consecutive retry budget; timeout, error, interruption, state, and delivery events do not. A persisted final response is reconciled even if cancellation left the SDK status paused. After three no-progress attempts or three total hours, Senpai preserves the raw trace and retries one canonical copy on a fresh branch with the complete initial controller context. If that recovery exhausts the same budget, the turn is durably quarantined, reported as `SENPAI_TURN_QUARANTINED` on every controller start, and excluded from scheduling rather than entering a restart loop. `SENPAI_INBOX_MAX_STALLED_ATTEMPTS`, `SENPAI_INBOX_MAX_TURN_AGE_SECONDS`, and `SENPAI_INBOX_MAX_RECOVERY_GENERATIONS` configure these positive attempt/age limits and the non-negative number of fresh branches.
- A typed context-window or malformed-history failure uses the same bounded fresh-branch recovery. The reset and its canonical recovery copy are durable across crashes; transient failures remain unacknowledged and retry after at least ten minutes.
- On restart, an incomplete persisted tool action is rejected rather than replayed implicitly. A checked-out assignment branch that was deliberately rebased or extended locally is preserved and surfaced to its existing student conversation for explicit reconciliation.
- The complete OpenHands event log remains locally searchable. Senpai does not prune conversation directories; operators own retention.
- Student state may be ephemeral because the branch, PR, typed result, W&B runs, and Weave trace are the durable handoff.
- Explicit project skills remain available through OpenHands skill context. Repository `AGENTS.md`, `AGENT.md`, and `CLAUDE.md` instruction files are reserved for human-facing development tools and are not loaded as Senpai project context.

The command policy blocks raw GitHub mutations, direct training, `git push`, polling loops, and log streams. Operation-specific typed tools enforce repository, branch, assignment, revision, head-SHA, label, and replay preconditions. This policy keeps routine operations deterministic while leaving high-entropy research work to the agent.

When `WANDB_ENTITY` and `WANDB_PROJECT` are configured, [`weave-openhands`](https://github.com/morganmcg1/weave-openhands) traces advisor, student, and child conversations. Each `OPENHANDS_RUN` record includes a direct Weave Agent Observability URL.

## Operations

Useful launch controls:

- `--names frieren,fern` selects stable students; otherwise use `--n_students` and `--student_prefix`.
- `--gpus_per_student`, `--cpu_per_gpu`, and `--memory_gi_per_gpu` size each student.
- `--timeout_minutes` and `--max_epochs` are hard per-training limits.
- `--poll_interval_s` and `--poll_jitter_s` control idle GitHub cadence without teaching the model to poll.
- `--gh_history_scope branch` keeps normal advisor-branch memory, `fresh` creates a shallow ablation checkout, and `repo` exposes full repository history.
- `--extra_instructions` accepts optional human operator guidance as a Markdown file or literal user context.
- `human_issues: false` disables GitHub Issue polling for isolated launches.

Advisor and student images are built from the same source revision. The advisor image excludes CUDA and PyTorch; the student image contains the CUDA/PyTorch runtime; the cutoff image contains only the minimal job runtime and pinned `kubectl`. Advisor and student builds install Chromium and execute an OpenHands browser smoke test.

For multi-day fleets, [`arm_senpai_cluster_cutoff.sh`](scripts/arm_senpai_cluster_cutoff.sh) creates a cluster-side hard cutoff that does not depend on an operator laptop remaining online. It can also hold a shared start gate until the expected fleet is ready or its readiness deadline expires.

On Kubernetes, pod startup and liveness probes read the supervisor lease.
Restarting a Deployment resumes the durable advisor or student conversation
when its state directory survives. For any backend, stop a live role before
copying or snapshotting its advisor state directory.

## Development and reference

```bash
uv sync --locked --extra dev
uv run pytest -q
bash -n k8s/*.sh scripts/*.sh plugins/senpai/scripts/*.sh
```

Deep references:

- [SPEC.md](SPEC.md): canonical runtime, persistence, safety, and acceptance contract.
- [OpenHands plugin](plugins/senpai/README.md): skills and lifecycle hooks.
- [Harness instructions](system_instructions/SENPAI-HARNESS.md): shared agent/tool contract.
- [Advisor instructions](system_instructions/ADVISOR.md) and [student instructions](system_instructions/STUDENT.md): role workflows.
- [OpenHands fork modifications](https://github.com/morganmcg1/software-agent-sdk/blob/main/FORK_MODS.md): provider continuation, compaction, reasoning, and cache changes.
- [Contributing](CONTRIBUTING.md): development and CLA requirements.
- [W&B dashboard](https://wandb.ai/wandb-applied-ai-team/senpai-v1): the default project's experiment record.
