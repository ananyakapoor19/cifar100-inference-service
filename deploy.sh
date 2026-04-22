#!/usr/bin/env bash
# deploy.sh – Build, push, and deploy the inference service to GCP.
#
# Usage:
#   PROJECT_ID=my-project BUCKET=my-bucket ./deploy.sh [cpu|gpu|both]
#
# Requires: gcloud CLI authenticated, Docker, Cloud Run API enabled.

set -euo pipefail

TARGET="${1:-cpu}"   # cpu | gpu | both
PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID env var}"
BUCKET="${BUCKET:?Set BUCKET env var}"
REGION="${REGION:-us-central1}"
REGISTRY="gcr.io/${PROJECT_ID}"

echo "=== Deploy CIFAR-100 Inference Service ==="
echo "Target:  $TARGET"
echo "Project: $PROJECT_ID"
echo "Region:  $REGION"

build_and_push() {
  local variant="$1"          # cpu or gpu
  local dockerfile="$2"
  local tag="${REGISTRY}/cifar100-inference:${variant}-latest"

  echo ""
  echo "── Building $variant image ──"
  docker build \
    -f "docker/${dockerfile}" \
    -t "$tag" \
    .

  echo "── Pushing $tag ──"
  docker push "$tag"
  echo "  ✓ Pushed $tag"
}

deploy_cloud_run() {
  echo ""
  echo "── Deploying to Cloud Run ──"
  # Replace placeholders in service.yaml
  sed \
    -e "s/PROJECT_ID/${PROJECT_ID}/g" \
    -e "s/YOUR_BUCKET/${BUCKET}/g" \
    infra/cloud_run/service.yaml > /tmp/service_rendered.yaml

  gcloud run services replace /tmp/service_rendered.yaml \
    --region="$REGION" \
    --project="$PROJECT_ID"

  # Allow unauthenticated (remove for production)
  gcloud run services add-iam-policy-binding cifar100-inference \
    --region="$REGION" \
    --project="$PROJECT_ID" \
    --member="allUsers" \
    --role="roles/run.invoker" \
    --quiet

  URL=$(gcloud run services describe cifar100-inference \
    --region="$REGION" --project="$PROJECT_ID" \
    --format="value(status.url)")
  echo "  ✓ Cloud Run URL: $URL"
}

# Configure Docker for GCR
gcloud auth configure-docker --quiet

case "$TARGET" in
  cpu)
    build_and_push "cpu" "Dockerfile"
    deploy_cloud_run
    ;;
  gpu)
    build_and_push "gpu" "Dockerfile.gpu"
    echo "GPU image pushed. Run infra/gce/startup.sh on your T4 instance to deploy."
    ;;
  both)
    build_and_push "cpu" "Dockerfile"
    build_and_push "gpu" "Dockerfile.gpu"
    deploy_cloud_run
    echo "GPU image pushed. Run infra/gce/startup.sh on your T4 instance to deploy."
    ;;
  *)
    echo "Unknown target: $TARGET. Use cpu | gpu | both"
    exit 1
    ;;
esac

echo ""
echo "=== Deployment complete ==="
