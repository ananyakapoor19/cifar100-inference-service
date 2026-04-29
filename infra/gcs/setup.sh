#!/usr/bin/env bash
# setup.sh – Create GCS bucket and IAM bindings for the inference service.
#
# Usage:
#   PROJECT_ID=my-project BUCKET=my-bucket ./infra/gcs/setup.sh

set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID}"
BUCKET="${BUCKET:?Set BUCKET}"
REGION="${REGION:-us-central1}"
SA_NAME="cifar100-inference-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "=== GCS + IAM Setup for CIFAR-100 Inference Service ==="
echo "Project: $PROJECT_ID"
echo "Bucket:  gs://$BUCKET"
echo "Region:  $REGION"

# ── Create bucket ──────────────────────────────────────────────────────────────
if ! gsutil ls -b "gs://${BUCKET}" &>/dev/null; then
  gsutil mb -p "$PROJECT_ID" -l "$REGION" -b on "gs://${BUCKET}"
  echo "Created bucket gs://$BUCKET"
else
  echo "Bucket gs://$BUCKET already exists."
fi

# Set lifecycle: delete objects older than 90 days (to keep costs low)
cat > /tmp/lifecycle.json <<EOF
{
  "lifecycle": {
    "rule": [{"action": {"type": "Delete"}, "condition": {"age": 90}}]
  }
}
EOF
gsutil lifecycle set /tmp/lifecycle.json "gs://${BUCKET}"

# ── Create service account ────────────────────────────────────────────────────
if ! gcloud iam service-accounts describe "$SA_EMAIL" \
      --project="$PROJECT_ID" &>/dev/null; then
  gcloud iam service-accounts create "$SA_NAME" \
    --project="$PROJECT_ID" \
    --display-name="CIFAR-100 Inference Service Account"
  echo "Created service account: $SA_EMAIL"
fi

# Grant object viewer on the model bucket
gsutil iam ch "serviceAccount:${SA_EMAIL}:roles/storage.objectViewer" \
  "gs://${BUCKET}"
echo "Granted storage.objectViewer to $SA_EMAIL"

# Grant Cloud Run invoker (for unauthenticated public access) – adjust as needed
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/run.invoker" \
  --quiet

# Grant Cloud Monitoring metric writer so the service can push custom metrics
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/monitoring.metricWriter" \
  --quiet
echo "Granted monitoring.metricWriter to $SA_EMAIL"

echo "=== Setup complete. ==="
echo ""
echo "Next steps:"
echo "  1. Upload checkpoints:"
echo "     python model/upload_to_gcs.py --bucket $BUCKET --files checkpoints/*.pth"
echo ""
echo "  2. Deploy to Cloud Run:"
echo "     ./deploy.sh"
