# Use the official CUDA image as the base for GPU support (e.g., CUDA 11.8 with Ubuntu 22.04)
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set environment variables to prevent prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libsndfile1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


# Install Python dependencies, including PyTorch with CUDA support
RUN pip3 install --upgrade pip
RUN pip3 install flask librosa transformers torch --extra-index-url https://download.pytorch.org/whl/cu118

# Continue with other commands, like installing Python packages
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# Create and set the working directory
WORKDIR /app

# Copy the application files into the container
COPY asr_api.py /app/
COPY wave_2_vector.py /app/

# Expose the port that Flask will run on
EXPOSE 8001

# Command to run the Flask app
CMD ["python3", "asr_api.py"]
