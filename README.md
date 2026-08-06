# Re-VLA

Getting Started
git clone https://github.com/CILAB-ROBOTICS/re-vla.git

Install LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git

Install lerobot
git clone https://github.com/CILAB-ROBOTICS/lerobot.git

Docker
cd docker && ./build.sh
docker run --rm -it --runtime=nvidia --gpus all -e MUJOCO_GL=egl libero_smolvla:latest
See docker/README.md for volume mounts (outputs, HF cache, W&B login) and details.
