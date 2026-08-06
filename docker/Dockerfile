# Reproduces the `libero_smolvla` conda environment (Python 3.10) as a container:
# lerobot fork (CILAB-ROBOTICS/lerobot @ dev/visualization, with this project's
# lerobot_train.py/configs/train.py fixes), LIBERO (with the libero/__init__.py fix that
# makes `pip install -e .` actually register the package), and the fine-tuning /
# rollout-collection / labeling scripts under scripts/.
#
# Build (from re_vla/, after running docker/build.sh once to stage the lerobot fork):
#   docker/build.sh
#
# Run (GPU, headless MuJoCo rendering):
#   docker run --rm -it --runtime=nvidia --gpus all \
#       -e MUJOCO_GL=egl -e NVIDIA_DRIVER_CAPABILITIES=all \
#       -v $(pwd)/outputs:/workspace/outputs \
#       libero_smolvla:latest

FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/envs/libero_smolvla/bin:/opt/conda/bin:$PATH

# System deps: build toolchain (some pip packages compile native extensions) + the
# headless-rendering stack MuJoCo/robosuite/EGL need. The actual NVIDIA EGL/GLX driver
# libraries are injected at `docker run` time by nvidia-container-toolkit (hence no
# nvidia driver package here) - libegl1/libglvnd are just the loader/ICD layer.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    cmake \
    pkg-config \
    libegl1 \
    libgl1 \
    libglvnd0 \
    libglx0 \
    libosmesa6 \
    libglew2.2 \
    libglfw3 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    patchelf \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Miniconda, just to host a Python 3.10 env matching the fork's requirement
# (`requires-python = ">=3.10"`; the rest of this project targets >=3.12).
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh && \
    ${CONDA_DIR}/bin/conda create -y -n libero_smolvla python=3.10 && \
    ${CONDA_DIR}/bin/conda clean -afy

WORKDIR /workspace

# Pinned dependency set (from `pip freeze` on the working libero_smolvla env, minus the
# two editable installs handled below). Installing this before COPYing source code keeps
# the layer cacheable across source-only changes.
COPY docker/requirements.txt /workspace/docker/requirements.txt
RUN pip install --no-cache-dir -r /workspace/docker/requirements.txt

# lerobot fork source, staged into docker/_lerobot_fork_src by build.sh (it lives outside
# this repo on the host, at /home/eunju/research/lerobot - not something a plain `docker
# build` context alone can reach without staging).
COPY docker/_lerobot_fork_src/ /workspace/lerobot/
RUN pip install --no-cache-dir --no-deps -e /workspace/lerobot

# LIBERO source (already includes this project's libero/__init__.py fix).
COPY LIBERO/ /workspace/LIBERO/
RUN pip install --no-cache-dir --no-deps -e /workspace/LIBERO

# Fine-tuning / rollout-collection / labeling scripts.
COPY scripts/ /workspace/scripts/

CMD ["/bin/bash"]
