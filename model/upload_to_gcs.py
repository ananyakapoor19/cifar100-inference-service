"""
upload_to_gcs.py – Upload model checkpoints to Google Cloud Storage.

Usage:
    python model/upload_to_gcs.py \
        --bucket my-bucket-name \
        --prefix models/efficientnet_b0_cifar100 \
        --files checkpoints/efficientnet_b0_cifar100_fp32.pth \
                checkpoints/efficientnet_b0_cifar100_int8.pth

Requires:
    pip install google-cloud-storage
    GOOGLE_APPLICATION_CREDENTIALS env var set (or gcloud auth application-default login)
"""

import argparse
import os
import sys
from pathlib import Path


def upload_blob(bucket_name: str, source_path: str, dest_blob: str) -> str:
    try:
        from google.cloud import storage
    except ImportError:
        print("ERROR: google-cloud-storage not installed. Run: pip install google-cloud-storage")
        sys.exit(1)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(dest_blob)

    print(f"Uploading {source_path} → gs://{bucket_name}/{dest_blob} …")
    blob.upload_from_filename(source_path)
    gcs_uri = f"gs://{bucket_name}/{dest_blob}"
    print(f"  ✓ {gcs_uri}")
    return gcs_uri


def main():
    parser = argparse.ArgumentParser(description="Upload checkpoints to GCS")
    parser.add_argument("--bucket",  required=True, help="GCS bucket name (no gs:// prefix)")
    parser.add_argument("--prefix",  default="models/efficientnet_b0_cifar100",
                        help="Destination prefix inside the bucket")
    parser.add_argument("--files",   nargs="+", required=True,
                        help="Local file paths to upload")
    args = parser.parse_args()

    uploaded = []
    for local_path in args.files:
        if not os.path.exists(local_path):
            print(f"WARNING: {local_path} not found, skipping.")
            continue
        filename  = Path(local_path).name
        dest_blob = f"{args.prefix.rstrip('/')}/{filename}"
        uri = upload_blob(args.bucket, local_path, dest_blob)
        uploaded.append(uri)

    print(f"\nUploaded {len(uploaded)} file(s):")
    for u in uploaded:
        print(f"  {u}")

    # Write a manifest so the inference server knows which URIs to fetch
    manifest_path = "checkpoints/gcs_manifest.txt"
    os.makedirs("checkpoints", exist_ok=True)
    with open(manifest_path, "w") as f:
        f.write("\n".join(uploaded) + "\n")
    print(f"\nManifest saved → {manifest_path}")


if __name__ == "__main__":
    main()
