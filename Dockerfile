# Use an official Python runtime as a parent image
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    bash \
    wget \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Create the checkpoints directory
RUN mkdir -p checkpoints

# Make the download script executable
RUN chmod +x download_ckpts.sh

# Download only the large checkpoint (use bash to avoid shebang issues)
RUN bash ./download_ckpts.sh

# Move the downloaded checkpoint to the checkpoints directory
RUN mv *.pt checkpoints/

# Entry point runs the RunPod worker
CMD ["python", "runpod_handler.py"]
