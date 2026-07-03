# For more information, please refer to https://aka.ms/vscode-docker-python
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Args for user creation - will be overridden by values from .env
ARG USER_NAME=developer
ARG USER_UID=1000
ARG WANDB_API_KEY=""

RUN apt-get update --allow-releaseinfo-change && \
    apt-get install -y gnupg && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys $(apt-key list | grep expired | awk '{print $2}') || true && \
    apt-get update && \
    apt-get install -y \
        build-essential \
        libssl-dev \
        libffi-dev \
        libgl1 \
        imagemagick \
        python3-dev \
        wget \
        curl \
        ca-certificates \
        python3-pip \
        sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install EGL / GL / OSMesa dependencies
RUN apt-get update && \
    apt-get install -y \
        libegl1 \
        libegl1-mesa \
        libegl-mesa0 \
        libgl1-mesa-glx \
        libgl1-mesa-dri \
        libgles2 \
        libosmesa6 \
        libosmesa6-dev \
        libx11-6 \
        libxext6 \
        libglew-dev \
        libxrender1 \
        libxcb1 \
        libxkbcommon0 \
        libxi6 \
        libxrandr2 \
        libxcursor1 \
        libxinerama1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Vulkan runtime (required by SAPIEN 3's renderer for both GUI viewer
# and headless offscreen rendering).
RUN apt-get update && \
    apt-get install -y \
        libvulkan1 \
        vulkan-tools && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Register the NVIDIA driver as a Vulkan ICD and EGL vendor. The NVIDIA
# container runtime mounts the driver libraries into the container, but the
# loader configs below are what let Vulkan/EGL actually discover them. Without
# these, SAPIEN rendering fails inside the container.
RUN mkdir -p /usr/share/vulkan/icd.d && \
    printf '%s\n' \
    '{' \
    '    "file_format_version" : "1.0.0",' \
    '    "ICD": {' \
    '        "library_path": "libGLX_nvidia.so.0",' \
    '        "api_version" : "1.3.223"' \
    '    }' \
    '}' > /usr/share/vulkan/icd.d/nvidia_icd.json && \
    mkdir -p /usr/share/glvnd/egl_vendor.d && \
    printf '%s\n' \
    '{' \
    '    "file_format_version" : "1.0.0",' \
    '    "ICD": {' \
    '        "library_path": "libEGL_nvidia.so.0"' \
    '    }' \
    '}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Install git and tmux
RUN apt-get update && \
    apt-get install -y git tmux
# CHANGE to your desired workspace name
RUN git config --global --add safe.directory /Frank-Sapien

# Make 'python' point to python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Upgrade pip, setuptools, wheel
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Attempt to install CUDA-enabled PyTorch wheels for cu121
RUN python -m pip install --no-cache-dir --upgrade \
    "torch" --index-url https://download.pytorch.org/whl/cu121 \
    torchvision --index-url https://download.pytorch.org/whl/cu121 \
    torchaudio --index-url https://download.pytorch.org/whl/cu121 || \
    (echo "WARNING: CUDA-enabled torch install may have failed or fallen back to CPU-only." && exit 0)

# Install additional requirements
COPY .devcontainer/requirements.txt .
RUN python -m pip install -r requirements.txt

# Install numpy 1.23.5 version
RUN python -m pip install numpy==1.23.5

# Creates a non-root user with an explicit UID and adds permission to access the folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
# CHANGES HERE
RUN mkdir -p /Frank-Sapien
WORKDIR /Frank-Sapien

# Setup user CHANGES HERE
RUN addgroup --gid ${USER_UID} ${USER_NAME} && \
    adduser --disabled-password --gecos "" --uid ${USER_UID} --gid ${USER_UID} ${USER_NAME} && \
    chown -R ${USER_UID}:${USER_UID} /home/${USER_NAME} && \
    chown -R ${USER_NAME}:${USER_NAME} /Frank-Sapien && \
    chmod -R 777 /Frank-Sapien

# Change the root access so we always have access as needed.
RUN echo 'root:root' | chpasswd
USER ${USER_NAME}

# Finally to fix proper permissions and imports, we modify the python sys path.
# CHANGES HERE
ENV PYTHONPATH="${PYTHONPATH}:/Frank-Sapien"
# Expose the NVIDIA driver's graphics + display capabilities (not just compute)
# so the mounted driver can be used for Vulkan/EGL rendering.
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV HYDRA_FULL_ERROR=1
ENV TORCH_LOGS="-all"
ENV TORCHDYNAMO_VERBOSE=1
ENV WANDB_API_KEY=${WANDB_API_KEY}

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python3"]