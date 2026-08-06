# libero_smolvla container

Reproduces the `libero_smolvla` conda environment (Python 3.10, lerobot fork +
LIBERO + this project's scripts) as a Docker image, for GPU training/rollout collection.

## Build

```bash
cd re_vla
docker/build.sh
```

This first stages the lerobot fork source (`/home/eunju/research/lerobot` by default —
override with `LEROBOT_FORK_SRC=/path/to/lerobot docker/build.sh`) into
`docker/_lerobot_fork_src/`, then runs `docker build`. The staged copy is regenerated
(and any stale one removed) on every run, so re-run `docker/build.sh` after pulling fork
updates.

Build is heavy (torch + CUDA wheels, ~168 pinned packages from `docker/requirements.txt`)
— expect it to take a while and use several GB of disk/bandwidth the first time.

## Run

```bash
docker run --rm -it --runtime=nvidia --gpus all \
    -e MUJOCO_GL=egl \
    -v $(pwd)/scripts/fine_tuning/outputs:/workspace/scripts/fine_tuning/outputs \
    -v $(pwd)/scripts/data_generation/outputs:/workspace/scripts/data_generation/outputs \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/.netrc:/root/.netrc:ro \
    libero_smolvla:latest
```

- `--runtime=nvidia --gpus all`: GPU passthrough; nvidia-container-toolkit injects the
  host's NVIDIA driver libs (including EGL) into the container at run time — the image
  itself only ships the loader (`libegl1`/`libglvnd`), not the proprietary driver.
- Mount `outputs/` directories so checkpoints/rollouts land on the host, not just inside
  the (ephemeral) container.
- Mount `~/.cache/huggingface` to reuse already-downloaded models/datasets instead of
  re-fetching them inside the container.
- Mount `~/.netrc` (read-only) to reuse existing W&B/HF Hub login instead of logging in
  again inside the container.

Inside the container, the scripts work exactly as documented in `scripts/README.md`,
e.g.:

```bash
cd /workspace/scripts/fine_tuning
./finetune_smolvla_libero.sh --output-dir ./outputs/libero_smolvla --steps 30000
```

## Notes

- The image installs `lerobot` and `libero` with `pip install -e . --no-deps` from the
  staged/copied source — this bakes in every fix made in this project session (the
  `lerobot_train.py` tqdm/validation/wandb fixes, the `libero/__init__.py` packaging fix)
  without needing to reapply them by hand.
- `docker/requirements.txt` is a frozen snapshot (`pip freeze`, minus the two editable
  installs) of the exact package versions in the working `libero_smolvla` conda env at
  the time this image was set up. Regenerate it (`pip freeze | grep -v '^-e '`) if you
  intentionally upgrade a dependency in the conda env and want the image to match.
