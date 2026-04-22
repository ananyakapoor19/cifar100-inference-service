#!/usr/bin/env bash
# startup.sh – Bootstrap script for GCE VM with NVIDIA T4.
#
# Run this as the VM startup script or manually after SSH:
#   gcloud compute instances create cifar100-gpu-vm \
#       --zone=us-central1-a \
#       --machine-type=n1-standard-4 \
#       --accelerator=type=nvidia-tesla-t4,count=1 \
#       --image-family=common-cu118 \
#       --image-project=deeplearning-platform-release \
#       --maintenance-policy=TERMINATE \
#       --metadata-from-file=startup-script=infra/gce/startup.sh \
#       --scopes=https://www.googleapis.com/auth/cloud-platform

set -euo pipefail

echo "=== CIFAR-100 GPU VM Startup ==="

# ── Install NVIDIA drivers if not present ─────────────────────────────────────
if ! command -v nvidia-smi &>/dev/null; then
  echo "Installing NVIDIA drivers…"
  apt-get update -q
  apt-get install -y -q linux-headers-$(uname -r) build-essential
  apt-get install -y -q nvidia-driver-525 nvidia-cuda-toolkit
fi

nvidia-smi

# ── Install Docker + NVIDIA Container Toolkit ─────────────────────────────────
if ! command -v docker &>/dev/null; then
  echo "Installing Docker…"
  curl -fsSL https://get.docker.com | bash
  systemctl enable --now docker
fi

if ! dpkg -l | grep -q nvidia-container-toolkit; then
  echo "Installing NVIDIA Container Toolkit…"
  distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -q
  apt-get install -y -q nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

# ── Authenticate with GCR ─────────────────────────────────────────────────────
gcloud auth configure-docker --quiet

# ── Pull and run the GPU inference container ──────────────────────────────────
PROJECT_ID=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/project/project-id" \
             -H "Metadata-Flavor: Google")
BUCKET=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/attributes/gcs-bucket" \
         -H "Metadata-Flavor: Google" 2>/dev/null || echo "YOUR_BUCKET")

IMAGE="gcr.io/${PROJECT_ID}/cifar100-inference:gpu-latest"

echo "Pulling image: $IMAGE"
docker pull "$IMAGE"

echo "Starting GPU inference container…"
docker run -d \
  --name cifar100-gpu \
  --gpus all \
  --restart unless-stopped \
  -p 8080:8080 \
  -e MODEL_FP32_PATH="gs://${BUCKET}/models/efficientnet_b0_cifar100/efficientnet_b0_cifar100_fp32.pth" \
  -e MODEL_INT8_PATH="gs://${BUCKET}/models/efficientnet_b0_cifar100/efficientnet_b0_cifar100_int8.pth" \
  -e PORT=8080 \
  -e LOG_LEVEL=info \
  "$IMAGE"

# Wait for health
echo "Waiting for /health…"
for i in $(seq 1 30); do
  if curl -sf http://localhost:8080/health &>/dev/null; then
    echo "Service healthy!"
    break
  fi
  sleep 2
done

docker logs cifar100-gpu --tail 20
echo "=== Startup complete ==="
