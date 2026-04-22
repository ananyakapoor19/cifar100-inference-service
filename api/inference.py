"""
inference.py – Model loading, preprocessing, and forward-pass logic.

Models are loaded once at startup (from GCS or local path) and kept in memory.
Thread-safety: each request uses torch.no_grad() and a shared read-only model.
"""

import io
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger(__name__)

# ── CIFAR-100 class names ──────────────────────────────────────────────────────
CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
    "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
    "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman",
    "worm",
]

# ── Image preprocessing ────────────────────────────────────────────────────────
IMG_SIZE = 224
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

_PREPROCESS = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])


def _build_fp32_model(num_classes: int = 100) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def _download_from_gcs(gcs_uri: str, local_path: str) -> None:
    """Download a GCS object to a local path."""
    try:
        from google.cloud import storage
    except ImportError as e:
        raise RuntimeError("google-cloud-storage not installed") from e

    # Parse gs://bucket/blob
    assert gcs_uri.startswith("gs://"), f"Invalid GCS URI: {gcs_uri}"
    parts = gcs_uri[5:].split("/", 1)
    bucket_name, blob_name = parts[0], parts[1]

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(blob_name)
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    blob.download_to_filename(local_path)
    logger.info("Downloaded %s → %s", gcs_uri, local_path)


class ModelRegistry:
    """
    Holds loaded model variants keyed by precision string ('fp32', 'int8').
    Loaded lazily on first request or eagerly at startup via .load_all().
    """

    def __init__(self, device: Optional[torch.device] = None):
        self._device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._models: Dict[str, nn.Module] = {}
        logger.info("ModelRegistry initialised on device=%s", self._device)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def loaded_keys(self) -> List[str]:
        return list(self._models.keys())

    def _resolve_path(self, uri_or_path: str, cache_name: str) -> str:
        """Return a local filesystem path, downloading from GCS if necessary."""
        if uri_or_path.startswith("gs://"):
            local_path = os.path.join("/tmp/model_cache", cache_name)
            if not os.path.exists(local_path):
                logger.info("Downloading model from GCS: %s", uri_or_path)
                _download_from_gcs(uri_or_path, local_path)
            return local_path
        return uri_or_path

    def load_fp32(self, path: str) -> None:
        local = self._resolve_path(path, "efficientnet_b0_cifar100_fp32.pth")
        logger.info("Loading FP32 model from %s", local)
        model = _build_fp32_model()
        ckpt  = torch.load(local, map_location=self._device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval().to(self._device)
        self._models["fp32"] = model
        logger.info("FP32 model loaded (device=%s)", self._device)

    def load_int8(self, path: str) -> None:
        local = self._resolve_path(path, "efficientnet_b0_cifar100_int8.pth")
        logger.info("Loading INT8 TorchScript model from %s", local)
        model = torch.jit.load(local, map_location=torch.device("cpu"))
        model.eval()
        self._models["int8"] = model
        logger.info("INT8 model loaded (device=cpu, quantized)")

    def load_all(self) -> None:
        fp32_path = os.environ.get("MODEL_FP32_PATH", "")
        int8_path = os.environ.get("MODEL_INT8_PATH", "")

        if fp32_path:
            try:
                self.load_fp32(fp32_path)
            except Exception as exc:
                logger.error("Failed to load FP32 model: %s", exc)

        if int8_path:
            try:
                self.load_int8(int8_path)
            except Exception as exc:
                logger.error("Failed to load INT8 model: %s", exc)

        if not self._models:
            raise RuntimeError(
                "No models loaded. Set MODEL_FP32_PATH and/or MODEL_INT8_PATH env vars."
            )

    def get(self, precision: str) -> nn.Module:
        if precision not in self._models:
            raise KeyError(f"Model '{precision}' not loaded. Available: {self.loaded_keys}")
        return self._models[precision]


# ── Preprocessing helpers ──────────────────────────────────────────────────────

def decode_image(image_bytes: bytes) -> Image.Image:
    """Decode raw image bytes to a PIL RGB image."""
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def preprocess(pil_image: Image.Image) -> torch.Tensor:
    """Apply standard CIFAR-100 preprocessing → (1, 3, 224, 224) tensor."""
    return _PREPROCESS(pil_image).unsqueeze(0)


def preprocess_batch(pil_images: List[Image.Image]) -> torch.Tensor:
    """Preprocess a list of PIL images → (N, 3, 224, 224) tensor."""
    return torch.stack([_PREPROCESS(img) for img in pil_images])


# ── Inference ──────────────────────────────────────────────────────────────────

def run_inference(
    model: nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
    top_k: int = 5,
) -> Tuple[List[dict], float]:
    """
    Run a forward pass and return top-k predictions + forward latency in ms.

    Returns:
        predictions: list of dicts with class_id, class_name, confidence
        forward_ms:  wall-clock time for the forward pass
    """
    # Move tensor to the right device (INT8 model always on CPU)
    is_int8 = next(model.parameters(), None) is None or str(device) == "cpu"
    input_device = torch.device("cpu") if is_int8 else device
    x = tensor.to(input_device)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(x)
    forward_ms = (time.perf_counter() - t0) * 1000

    probs    = F.softmax(logits, dim=1)
    topk_vals, topk_idxs = probs.topk(top_k, dim=1)

    predictions = []
    for i in range(x.size(0)):
        preds_for_image = []
        for rank in range(top_k):
            class_id = topk_idxs[i, rank].item()
            preds_for_image.append({
                "class_id":   class_id,
                "class_name": CIFAR100_CLASSES[class_id],
                "confidence": round(float(topk_vals[i, rank].item()), 6),
            })
        predictions.append(preds_for_image)

    return predictions, forward_ms
