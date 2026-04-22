"""
evaluate.py – Evaluate a checkpoint on the CIFAR-100 test set.
Prints top-1 / top-5 accuracy and per-class breakdown.

Usage:
    python model/evaluate.py \
        --ckpt ./checkpoints/efficientnet_b0_cifar100_fp32.pth \
        --data-dir ./data \
        [--scripted]   # if the checkpoint is a TorchScript model
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import torch.nn as nn

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)
IMG_SIZE = 224


def build_model(num_classes: int = 100) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def load_model(ckpt_path: str, scripted: bool, device: torch.device):
    if scripted:
        model = torch.jit.load(ckpt_path, map_location=device)
    else:
        model = build_model()
        ckpt  = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    top1_correct, top5_correct, total = 0, 0, 0
    class_correct = [0] * 100
    class_total   = [0] * 100

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        probs  = F.softmax(logits, dim=1)

        # Top-1
        _, pred1 = probs.max(1)
        top1_correct += pred1.eq(labels).sum().item()

        # Top-5
        _, pred5 = probs.topk(5, dim=1)
        top5_correct += pred5.eq(labels.unsqueeze(1).expand_as(pred5)).any(1).sum().item()

        total += labels.size(0)

        for i in range(labels.size(0)):
            c = labels[i].item()
            class_total[c]   += 1
            class_correct[c] += int(pred1[i].item() == c)

    top1 = top1_correct / total
    top5 = top5_correct / total
    per_class = {i: class_correct[i] / max(class_total[i], 1) for i in range(100)}
    return top1, top5, per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",      type=str, required=True)
    parser.add_argument("--data-dir",  type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers",   type=int, default=4)
    parser.add_argument("--scripted",  action="store_true")
    parser.add_argument("--device",    type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output",    type=str, default=None,
                        help="Optional JSON path to save per-class results")
    args = parser.parse_args()

    device = torch.device(args.device)

    val_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    val_ds     = datasets.CIFAR100(args.data_dir, train=False, download=True, transform=val_tf)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    print(f"Loading model from {args.ckpt}…")
    model = load_model(args.ckpt, args.scripted, device)

    print("Evaluating…")
    top1, top5, per_class = evaluate(model, val_loader, device)

    print(f"\nTop-1 accuracy: {top1:.4f} ({top1*100:.2f}%)")
    print(f"Top-5 accuracy: {top5:.4f} ({top5*100:.2f}%)")

    worst5 = sorted(per_class.items(), key=lambda x: x[1])[:5]
    best5  = sorted(per_class.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\nWorst 5 classes: {worst5}")
    print(f"Best  5 classes: {best5}")

    if args.output:
        result = {"top1": top1, "top5": top5, "per_class": per_class}
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved results → {args.output}")


if __name__ == "__main__":
    main()
