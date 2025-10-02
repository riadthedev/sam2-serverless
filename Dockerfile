# Use an official Python runtime as a parent image
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Set the working directory in the container
WORKDIR /app

# Install minimal system dependencies (rarely changes)
RUN apt-get update && apt-get install -y \
    bash \
    wget \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements file first (rarely changes)
COPY requirements.txt /app/

# Install Python dependencies (rarely changes)
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Create the checkpoints directory
RUN mkdir -p checkpoints

# Copy only the download script (rarely changes)
COPY download_ckpts.sh /app/

# Make the download script executable
RUN chmod +x download_ckpts.sh

# Download only the large checkpoint (rarely changes - this is the slowest step!)
RUN bash ./download_ckpts.sh

# Move the downloaded checkpoint to the checkpoints directory
RUN mv *.pt checkpoints/

# Copy the rest of your code (changes often - put this LAST!)
COPY . /app

# Entry point runs the RunPod worker
CMD ["python", "runpod_handler.py"]
