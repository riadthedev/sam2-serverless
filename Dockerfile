FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

# Install system dependencies and clean up aggressively
RUN apt-get update && apt-get install -y \
    bash \
    wget \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY requirements.txt /app/

# Install Python deps and clean up immediately
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip \
    && find /opt/conda -name "*.pyc" -delete \
    && find /opt/conda -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Clone SAM2 repo to get config files
RUN git clone https://github.com/facebookresearch/segment-anything-2.git /tmp/sam2 \
    && cp -r /tmp/sam2/sam2_configs /app/sam2_configs \
    && rm -rf /tmp/sam2

ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

RUN mkdir -p checkpoints

COPY download_ckpts.sh /app/
RUN chmod +x download_ckpts.sh

# Download checkpoint and clean up
RUN bash ./download_ckpts.sh \
    && mv *.pt checkpoints/ \
    && rm -f download_ckpts.sh

COPY . /app

CMD ["python", "runpod_handler.py"]