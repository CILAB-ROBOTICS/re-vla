# Re-VLA

## Getting Started

```bash
git clone https://github.com/CILAB-ROBOTICS/re-vla.git
```


### Install LIBERO
```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
```

### Install lerobot
```bash
git clone https://github.com/CILAB-ROBOTICS/lerobot.git
```

## Docker
```bash
cd docker && ./build.sh
docker run --rm -it --runtime=nvidia --gpus all -e MUJOCO_GL=egl libero_smolvla:latest
```