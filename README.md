# RunPod SAM 2 Image Embedding Serverless

Minimal RunPod Serverless worker that computes and returns SAM 2 image embeddings for client-side decoding.

## Features
- Returns full image embedding map (C×H×W) from SAM 2 image encoder.
- Serialization as compressed `npz_base64` for compact transport.
- Includes metadata for front-end coordinate mapping and decoding.

## Requirements
- Python 3.10+
- GPU on RunPod (recommended). CPU works but is slow.

## Install
1. Install deps:
   pip install -r requirements.txt
2. Download SAM 2 checkpoints:
   ./download_ckpts.sh
   mv *.pt checkpoints/

## Usage
The worker exposes a single action `embed_image` via RunPod Serverless.

Input event (either image_b64 or input_image_url):
{
  "input": {
    "action": "embed_image",
    "image_b64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ...",  
    "dtype": "float16",
    "resize_longer_side": 1024
  }
}

or

{
  "input": {
    "action": "embed_image",
    "input_image_url": "https://example.com/image.jpg",
    "dtype": "float16",
    "resize_longer_side": 1024
  }
}

Response (truncated):
{
  "type": "map",
  "encoding": "npz_base64",
  "data": "<base64 string>",
  "shape": [256, 64, 64],
  "dtype": "float16",
  "meta": {
    "model": "sam2_hiera_l.yaml",
    "orig_size": [H_orig, W_orig],
    "input_size": [1024, 1024],
    "resized_size": [H_resized, W_resized],
    "pad": [top, left, bottom, right],
    "scale": {"h": s, "w": s},
    "stride": 16
  },
  "device": "cuda",
  "model": "sam2_hiera_large.pt"
}

Client: base64-decode and load the NPZ (key `embedding`) with NumPy to reconstruct `C×H×W`.

## Config
- `SAM2_MODEL_CFG` (default `sam2_configs/sam2_hiera_l.yaml`)
- `SAM2_CHECKPOINT` (default `./checkpoints/sam2_hiera_large.pt`)
 - `PYTORCH_CUDA_ALLOC_CONF` (default set to `max_split_size_mb:128` in Dockerfile)

## Files
- `runpod_handler.py` — RunPod handler wiring (action: `embed_image`).
- `sam2_processor.py` — Model init and `embed_image` implementation.
- `sam2_configs/` — Model YAML configs.
- `download_ckpts.sh` — Helper to fetch checkpoints.
- `test_input_embed.json` — Sample RunPod event.

## License
MIT License
