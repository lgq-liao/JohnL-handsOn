To enable GPU support (CUDA) in your Docker container, you'll need to do the following:

# 1. **Ensure CUDA and NVIDIA drivers are installed on your host machine**: For your application to leverage the GPU, your system must have the appropriate NVIDIA drivers and CUDA installed. You can check if the drivers are installed with the following command:

   ```bash
   nvidia-smi
   ```

   This will display information about your GPU if the NVIDIA drivers are correctly installed.

# 2. **Install CUDA within the Docker container** (optional, if your application requires GPU processing): 
   If you're using PyTorch and want to take advantage of CUDA for deep learning tasks, you should use a base image that has CUDA support. For instance, NVIDIA provides Docker images with CUDA preinstalled.

   You can modify your `Dockerfile` to use a CUDA-enabled base image, such as `nvidia/cuda`, and ensure the necessary GPU drivers and libraries are installed. Here's how you can update your `Dockerfile` to enable CUDA support:


# 3. **NVIDIA Docker Support**: To run Docker containers that utilize the GPU, you'll need the NVIDIA Container Toolkit (`nvidia-docker`) installed. This allows Docker to access the GPU resources of your host machine. You can install the `nvidia-docker` toolkit by following the instructions from the [NVIDIA Docker installation page](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
    ```bash
    # Update your package repository
    sudo apt-get update

    sudo apt-get install -y nvidia-container-toolkit

    # Install the NVIDIA container toolkit
    sudo apt-get install -y nvidia-docker2

    # Restart Docker service
    sudo systemctl restart docker
    ```

# 4. **Build the Docker image**

    ```bash
    docker build -t asr-api .
    ```

# 5. **Running the Docker container with GPU support**: Once the `nvidia-docker` toolkit is installed, you can run your Docker container with GPU support by using the `--gpus` flag:

   ```bash

   # NOTE: need root privileges if not config docker in rootless mode
   sudo docker run --gpus all --rm -p 8001:8001 asr-api # run with GPU
   docker run --rm -p 8001:8001 asr-api            # run with CPU

   # to check GPU available 
   sudo docker run --rm --gpus all asr-api nvidia-smi

   # run in debug mode
   sudo docker run --rm --gpus all -p 8001:8001 -it asr-api bash
   # (Optional) Verify PyTorch installation
   python3 -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

   ```

   This command will give your container access to all available GPUs. If you have multiple GPUs and want to specify one, you can use `--gpus '"device=0"'` for the first GPU, etc.

