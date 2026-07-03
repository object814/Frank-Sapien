# Frank Sapien Simulator

**GPU-accelerated, Gym-style reinforcement learning environments for the custom Frank robot, built on [SAPIEN](https://sapien.ucsd.edu/).**

Frank Sapien Simulator provides a standard table-top manipulation environment for our custom **Frank** robot, exposed through a clean [Gymnasium](https://gymnasium.farama.org/) interface. It is designed from the ground up to exploit SAPIEN's **GPU-parallel physics simulation and rendering**, so you can run thousands of environments in parallel on a single GPU and dramatically accelerate reinforcement learning research.

> ⚠️ **Status: early development.** This README documents the intended API and usage. Some components are still being implemented — see the [Roadmap](#roadmap).

## Features

- **Gymnasium-compatible** single-environment API for debugging, visualization, and library compatibility.
- **Vectorized GPU environment** for massively parallel, on-GPU rollouts (observations and actions stay on the GPU as torch tensors).
- **Custom Frank robot** with its own robot description, collision meshes, and configurable controllers.
- **Multiple control modes** — joint position, joint velocity, end-effector pose, delta end-effector pose.
- **Rich observations** — proprioception, object states, and GPU-rendered RGB / depth / segmentation.
- **Domain randomization** hooks for sim-to-real research.
- **Deterministic seeding** across parallel environments for reproducibility.

---

## Installation

Frank Sapien targets **Linux + NVIDIA GPU** and **Python 3.10**. The GPU-parallel
physics and rendering are provided by **SAPIEN 3.x**, so a CUDA-capable GPU is
required for the full feature set.

The recommended way to get a reproducible environment is the provided
**dev container**, which pins CUDA 12.1, PyTorch (cu121), and all system
libraries needed for both on-screen (GUI viewer) and headless (offscreen) GPU
rendering.

### Prerequisites

- **NVIDIA GPU** with a recent driver (CUDA 12.1-compatible).
- **[Docker](https://docs.docker.com/engine/install/)** with **Docker Compose v2**.
- **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)** (so containers can access the GPU).
- *(Optional)* **[VS Code](https://code.visualstudio.com/)** with the **Dev Containers** extension, if you want the one-click "Reopen in Container" workflow.

### Option A — Dev container (recommended)

```bash
git clone https://github.com/oxfordrobotics/frank-sapien.git
cd frank-sapien

# 1. Generate the .devcontainer/.env file (sets USER_NAME / USER_UID, and lets
#    you fill in your WANDB_API_KEY).
bash .devcontainer/setupEnv.sh
#    Then edit .devcontainer/.env to add your WANDB_API_KEY if you use W&B.
```

Then start the container using **either** VS Code **or** the CLI:

- **VS Code:** open the folder and run
  `Ctrl/Cmd + Shift + P → "Dev Containers: Rebuild and Reopen in Container"`.
  Connect to a specific GPU container from the **Remote Explorer** tab.

- **CLI:**

  ```bash
  cd .devcontainer
  docker compose build
  docker compose up -d frank_sapien_gpu0      # or frank_sapien_gpu1
  docker compose exec frank_sapien_gpu0 bash  # drop into a shell
  ```

The workspace is mounted at `/Frank-Sapien` inside the container, and
`PYTHONPATH` already includes it, so `import frank_sapien` works without an
explicit install once the package exists.

> The compose file defines one service per GPU (`frank_sapien_gpu0`,
> `frank_sapien_gpu1`, …). Duplicate a service block and change
> `NVIDIA_VISIBLE_DEVICES` / `device_ids` to add more GPUs.

### Rendering (GUI vs. headless)

SAPIEN 3 renders with **Vulkan**. The container is configured so the NVIDIA
driver is usable for graphics (`NVIDIA_DRIVER_CAPABILITIES=all`), with the
Vulkan/EGL ICD loaders installed. Both modes work:

- **Headless (host has no display):** GPU offscreen rendering works out of the
  box — RGB / depth / segmentation observations and video capture require no
  X server. Just don't open a viewer.

- **On-screen GUI viewer (host has a display):** X11 forwarding is already wired
  up in `docker-compose.yml` (`DISPLAY` + the `/tmp/.X11-unix` mount). On the
  **host**, allow the container to talk to your X server before launching the
  GUI:

  ```bash
  xhost +local:root   # run on the host, once per session
  ```

  Then, inside the container, a SAPIEN viewer window (`render_mode="human"`)
  will appear on your host display.

Verify the GPU and renderer are visible from inside the container:

```bash
nvidia-smi          # GPU is visible
vulkaninfo | head   # Vulkan sees the NVIDIA driver
python -c "import sapien; print(sapien.__version__)"
```

### Option B — Local install (no Docker)

If you prefer a bare environment, replicate what the Dockerfile does. Using conda:

```bash
conda create -n frank-sapien python=3.10 -y
conda activate frank-sapien

# PyTorch built against CUDA 12.1 (kept out of requirements.txt on purpose)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Project dependencies
pip install -r .devcontainer/requirements.txt

# Install the package itself (editable), once packaging is in place
pip install -e .
```

> For headless GPU rendering outside Docker you may still need the Vulkan
> runtime (`libvulkan1`) and the NVIDIA Vulkan ICD available on your system —
> see the `.devcontainer/Dockerfile` for the exact system packages.

## Quickstart

### Single environment (Gymnasium API)

```python
import gymnasium as gym
import frank_sapien  # registers the environments

env = gym.make("FrankTableTop-v0", render_mode="human")

obs, info = env.reset(seed=0)
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Vectorized GPU environment (parallel RL)

The vectorized environment keeps observations, rewards, and actions as GPU tensors, so there is no host–device copy in the training loop.

```python
import torch
from frank_sapien import make_vec_env

# 4096 environments simulated in parallel on the GPU
env = make_vec_env("FrankTableTop-v0", num_envs=4096, device="cuda")

obs, info = env.reset(seed=0)            # obs: dict of (num_envs, ...) cuda tensors
for _ in range(1000):
    action = torch.rand(env.num_envs, *env.single_action_space.shape, device="cuda")
    obs, reward, terminated, truncated, info = env.step(action)  # all on GPU

env.close()
```

---

## Environments

| ID | Description | Success criterion |
|----|-------------|-------------------|
| `FrankTableTop-v0` | Base table-top scene with Frank and a configurable object set. | Task-dependent |
| `FrankReach-v0` | Move the end-effector to a target position. | EE within tolerance of goal |
| `FrankPush-v0` | Push an object to a goal region on the table. | Object within goal region |
| `FrankPickPlace-v0` | Grasp an object and place it at a target. | Object placed at target |

> The environment suite is actively growing — see the [Roadmap](#roadmap).

### Configuration

Environments accept keyword overrides at creation time:

```python
env = gym.make(
    "FrankTableTop-v0",
    control_mode="delta_ee_pose",   # "joint_pos" | "joint_vel" | "delta_ee_pose"
    obs_mode="state",               # "state" | "rgb" | "rgbd" | "state+rgbd"
    reward_mode="dense",            # "dense" | "sparse"
    domain_randomization=True,
    max_episode_steps=200,
)
```

---

## Observations, actions, and rewards

- **Observation modes**
  - `state` — proprioception (joint positions/velocities, EE pose) and privileged object state.
  - `rgb` / `rgbd` — GPU-rendered images from configured cameras.
  - `state+rgbd` — combined dictionary observation.
- **Action space** — depends on `control_mode`; continuous and normalized to `[-1, 1]`.
- **Reward modes** — `dense` shaped rewards for fast learning, or `sparse` for benchmarking.

---

## The Frank robot

Frank is a **fully custom robot** with its own description and assets, located under [`frank_sapien/assets/frank/`](frank_sapien/assets/frank/):

```
frank_sapien/assets/frank/
├── frank.urdf            # kinematics + dynamics
├── meshes/               # visual and collision meshes
└── config.yaml           # joint limits, default pose, controller gains
```

To use an updated robot description, drop in new files and point the config at them — no code changes are required. See [`docs/robot.md`](docs/robot.md) for the full asset specification and controller options.

---

## Training example

A minimal PPO training loop using the vectorized GPU environment:

```python
from frank_sapien import make_vec_env
from frank_sapien.rl import PPO  # thin wrapper; bring your own algorithm if preferred

env = make_vec_env("FrankReach-v0", num_envs=2048, device="cuda")
agent = PPO(env)
agent.train(total_steps=50_000_000)
agent.save("checkpoints/frank_reach.pt")
```

The vectorized environment also conforms to common vector-env conventions, so it can be adapted to libraries such as [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), [RSL-RL](https://github.com/leggedrobotics/rsl_rl), or [CleanRL](https://github.com/vwxyzjn/cleanrl). See [`examples/`](examples/) for end-to-end scripts.

---

## Project structure

```
frank-sapien/
├── frank_sapien/
│   ├── envs/            # environment definitions and registration
│   ├── agents/          # Frank robot loader and controllers
│   ├── tasks/           # task logic, rewards, success criteria
│   ├── scene/           # table-top scene construction and randomization
│   ├── sensors/         # cameras and rendering
│   ├── assets/          # robot and object assets
│   └── rl/              # optional training utilities
├── examples/            # runnable training and demo scripts
├── tests/
└── docs/
```

---

## Roadmap

- [ ] Core table-top scene and Frank robot loader
- [ ] Single-environment Gymnasium API
- [ ] Vectorized GPU environment
- [ ] Reach / Push / Pick-and-Place tasks
- [ ] GPU-rendered image observations
- [ ] Domain randomization utilities
- [ ] Baseline PPO results and benchmarks
- [ ] Sim-to-real tooling and documentation

---

## Contributing

Contributions are welcome. Please open an issue to discuss substantial changes before submitting a pull request, and make sure `pytest` passes.

---

## Acknowledgements

Built on [SAPIEN](https://sapien.ucsd.edu/) from UC San Diego. The environment design draws inspiration from [ManiSkill](https://github.com/haosulab/ManiSkill) and the broader [Farama Gymnasium](https://gymnasium.farama.org/) ecosystem.

---

## Citation

If you use Frank Sapien Simulator in your research, please cite:

```bibtex
@software{frank_sapien_simulator,
  title  = {Frank Sapien Simulator: GPU-Accelerated RL Environments for the Frank Robot},
  author = {Oxford Robotics Institute},
  year   = {2026},
  url    = {https://github.com/oxfordrobotics/frank-sapien}
}
```

## License

TBD.
