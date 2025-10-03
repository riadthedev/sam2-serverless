import os
import io
import base64
import logging
import numpy as np
import requests
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize the SAM 2 image model once
MODEL_CFG = os.getenv("SAM2_MODEL_CFG", "sam2_hiera_l")
SAM2_CHECKPOINT = os.getenv("SAM2_CHECKPOINT", "/app/checkpoints/sam2_hiera_large.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    try:
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

logger.info(f"Loading SAM2 model on {DEVICE} with cfg {MODEL_CFG}")
sam2_model = build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device=DEVICE)
image_predictor = SAM2ImagePredictor(sam2_model)


def _serialize_npz_base64(arr: np.ndarray) -> str:
    buffer = io.BytesIO()
    # Use compressed NPZ to reduce payload size
    np.savez_compressed(buffer, embedding=arr)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def embed_image(job):
    """
    Compute and return the SAM 2 image embedding for a single image.

    Input (job["input"]):
      - input_image_url: URL to the image (preferred)
      - or input_video_url + ann_frame_idx: extract a frame first
      - dtype: "float16" (default) or "float32" for the returned embedding
      - resize_longer_side: default 1024 (metadata; predictor handles its own transforms)

    Output:
      JSON with keys: type, encoding, data (npz base64), shape, dtype, meta, timing_ms, device, model
    """
    job_input = job.get("input", {})
    image_url = job_input.get("input_image_url")
    image_b64 = job_input.get("image_b64")
    video_url = job_input.get("input_video_url")
    frame_index = job_input.get("ann_frame_idx")
    out_dtype = str(job_input.get("dtype", "float16")).lower()
    resize_longer_side = int(job_input.get("resize_longer_side", 1024))

    if not (image_b64 or image_url or (video_url and frame_index is not None)):
        return {"error": "Missing image_b64 or input_image_url (or input_video_url with ann_frame_idx)"}
    if out_dtype not in ("float16", "float32"):
        return {"error": f"Unsupported dtype: {out_dtype}"}

    # Load image (RGB np array); prefer base64 if provided
    try:
        if image_b64:
            b64_str = image_b64
            if "," in b64_str and b64_str.strip().startswith("data:"):
                # data URL pattern: data:image/png;base64,<payload>
                b64_str = b64_str.split(",", 1)[1]
            raw = base64.b64decode(b64_str, validate=True)
            from PIL import Image
            from io import BytesIO
            pil_img = Image.open(BytesIO(raw)).convert("RGB")
            image = np.array(pil_img)
        elif video_url and frame_index is not None:
            # Not supported in embeddings-only build
            return {"error": "input_video_url/frame extraction not supported in embeddings-only build"}
        else:
            resp = requests.get(image_url, timeout=20)
            resp.raise_for_status()
            from PIL import Image
            from io import BytesIO
            pil_img = Image.open(BytesIO(resp.content)).convert("RGB")
            image = np.array(pil_img)
    except Exception as e:
        return {"error": f"Failed to load image: {str(e)}"}

    orig_h, orig_w = int(image.shape[0]), int(image.shape[1])

    # Run predictor to compute (and cache) image embedding
    try:
        image_predictor.set_image(image)
    except Exception as e:
        return {"error": f"Failed to set image for predictor: {str(e)}"}

    # Try multiple ways to access the image embedding robustly
    emb_tensor = None
    try:
        if hasattr(image_predictor, "get_image_embedding"):
            emb_tensor = image_predictor.get_image_embedding()
        elif hasattr(image_predictor, "get_image_embeddings"):
            emb_tensor = image_predictor.get_image_embeddings()
        elif hasattr(image_predictor, "features"):
            feats = getattr(image_predictor, "features")
            # Common keys seen in predictors
            for key in ("image_embed", "image_embeddings", "image_embedding", "features"):
                if isinstance(feats, dict) and key in feats:
                    emb_tensor = feats[key]
                    break
    except Exception:
        # fallthrough to error below if nothing is set
        emb_tensor = None

    if emb_tensor is None:
        return {"error": "Could not retrieve image embedding from predictor. Ensure SAM2 version supports get_image_embedding()."}

    # Move to CPU and convert dtype
    try:
        emb = emb_tensor.detach().to("cpu")
        # Remove batch dim if present: (1, C, H, W) -> (C, H, W)
        if emb.ndim == 4 and emb.shape[0] == 1:
            emb = emb[0]
        elif emb.ndim == 3:
            pass
        else:
            # Unexpected shape, try to squeeze
            emb = emb.squeeze()

        if out_dtype == "float16":
            emb = emb.to(torch.float16)
        else:
            emb = emb.to(torch.float32)

        emb_np = emb.numpy()
    except Exception as e:
        return {"error": f"Failed to finalize embedding tensor: {str(e)}"}

    # Compute deterministic resize/pad metadata assuming 1024 pipeline
    longer = max(orig_h, orig_w)
    scale = resize_longer_side / float(longer)
    resized_h = int(round(orig_h * scale))
    resized_w = int(round(orig_w * scale))
    pad_top = 0
    pad_left = 0
    pad_bottom = max(0, resize_longer_side - resized_h)
    pad_right = max(0, resize_longer_side - resized_w)

    # Heuristic stride; most SAM2 image encoders downsample by 16
    stride = 16

    try:
        data_b64 = _serialize_npz_base64(emb_np)
    except Exception as e:
        return {"error": f"Failed to serialize embedding: {str(e)}"}

    meta = {
        "model": os.path.basename(MODEL_CFG),
        "orig_size": [orig_h, orig_w],
        "input_size": [resize_longer_side, resize_longer_side],
        "resized_size": [resized_h, resized_w],
        "pad": [pad_top, pad_left, pad_bottom, pad_right],
        "scale": {"h": scale, "w": scale},
        "stride": stride,
    }

    return {
        "type": "map",
        "encoding": "npz_base64",
        "data": data_b64,
        "shape": list(emb_np.shape),
        "dtype": out_dtype,
        "meta": meta,
        "device": DEVICE,
        "model": os.path.basename(SAM2_CHECKPOINT),
    }
    
def process_single_image(job):
    job_input = job["input"]
    image_url = job_input.get("input_image_url")
    video_url = job_input.get("input_video_url")
    frame_index = job_input.get("ann_frame_idx")
    points = job_input.get("points")
    labels = job_input.get("labels")

    if not (image_url or (video_url and frame_index is not None)):
        return {"error": "Missing image_url or video_url with frame_index"}
    if points is None or labels is None:
        return {"error": "Missing points or labels parameter"}

    try:
        points = np.array(points, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
    except ValueError:
        return {"error": "Invalid format for points or labels"}
    
    if video_url and frame_index is not None:
        try:
            image = extract_frame_from_video(video_url, frame_index)
        except Exception as e:
            return {"error": f"Failed to extract frame from video: {str(e)}"}
    else:
        try:
            image = load_image_from_url(image_url)
        except requests.RequestException as e:
            return {"error": f"Failed to download image: {str(e)}"}
        except IOError:
            return {"error": "Failed to open image"}

    if image is None:
        return {"error": "Failed to obtain image"}

    logger.debug("image predictor initialized successfully.")

    image_np = np.array(image)
    image_predictor.set_image(image_np)

    try:
        masks, scores, _ = image_predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]

    annotated_image = image_np.copy()
    for mask in masks:
        annotated_image = apply_mask(mask, annotated_image.copy(), random_color=True)

    # Add points to the final annotated image
    show_points(points, labels, annotated_image)

    try:
        annotated_buffer = encode_image(annotated_image)
        combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
        mask_buffer = encode_image(combined_mask)
    except Exception as e:
        return {"error": f"Failed to encode output images: {str(e)}"}

    try:
        bytescale_image_url = upload_to_bytescale(annotated_buffer)
        bytescale_mask_url = upload_to_bytescale(mask_buffer)
    except requests.RequestException as e:
        return {"error": f"Failed to upload images to Bytescale: {str(e)}"}

    return {
        "bytescale_image_url": bytescale_image_url,
        "bytescale_mask_url": bytescale_mask_url,
        "scores": scores.tolist()
    }
