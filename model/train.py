"""
train.py – Fine-tune EfficientNet-B0 on CIFAR-100 via transfer learning.

Usage:
    python model/train.py --epochs 20 --batch-size 128 --lr 1e-3 \
        --output-dir ./checkpoints

The script:
  1. Downloads CIFAR-100 and applies standard augmentation.
  2. Loads a torchvision EfficientNet-B0 pretrained on ImageNet.
  3. Replaces the classifier head for 100 classes.
  4. Trains with cosine-annealing LR schedule.
  5. Saves the best checkpoint (by val accuracy) to --output-dir.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

# EfficientNet-B0 expects 224×224 input
IMG_SIZE = 224


def get_dataloaders(data_dir: str, batch_size: int, num_workers: int = 4):
    train_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomCrop(IMG_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    train_ds = datasets.CIFAR100(data_dir, train=True,  download=True, transform=train_tf)
    val_ds   = datasets.CIFAR100(data_dir, train=False, download=True, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def build_model(num_classes: int = 100, pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # Freeze all backbone layers for the first phase
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze the entire network for fine-tuning phase."""
    for param in model.parameters():
        param.requires_grad = True


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total   += images.size(0)

        if i % 50 == 0:
            print(f"  [Epoch {epoch}] step {i}/{len(loader)}  "
                  f"loss={loss.item():.4f}  acc={correct/total:.3f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, preds    = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total   += images.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Fine-tune EfficientNet-B0 on CIFAR-100")
    parser.add_argument("--data-dir",    type=str,   default="./data")
    parser.add_argument("--output-dir",  type=str,   default="./checkpoints")
    parser.add_argument("--epochs",      type=int,   default=25)
    parser.add_argument("--batch-size",  type=int,   default=128)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-4,
                        help="LR applied to backbone after unfreeze (epoch > warmup-epochs)")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Epochs to train only the head before unfreezing backbone")
    parser.add_argument("--workers",     type=int,   default=4)
    parser.add_argument("--device",      type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size, args.workers)

    model = build_model(num_classes=100, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Phase 1: head-only
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.warmup_epochs)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Switch to full fine-tune after warmup
        if epoch == args.warmup_epochs + 1:
            print(">>> Unfreezing backbone for full fine-tuning <<<")
            unfreeze_backbone(model)
            optimizer = optim.AdamW([
                {"params": model.features.parameters(), "lr": args.lr_backbone},
                {"params": model.classifier.parameters(), "lr": args.lr},
            ], weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
              f"time={elapsed:.1f}s")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.output_dir, "efficientnet_b0_cifar100_fp32.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint (val_acc={val_acc:.4f}) → {ckpt_path}")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
