"""
train.py — Rice Leaf Disease Classifier
Fine-tunes EfficientNet-B0 (timm) on merged Rice Leaf Dataset1 + Dataset2.
Outputs:  rice_disease_model.pth, class_names.json
"""

import os, json, copy, random, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 32
NUM_EPOCHS = 3
LR         = 1e-3
NUM_WORKERS = 0          # safe on macOS
DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ──────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

class RiceLeafDataset(Dataset):
    """Loads images from multiple root directories that share the same
    sub-folder class structure."""

    def __init__(self, roots, transform=None):
        self.samples = []       # list of (path, label_idx)
        self.transform = transform

        # Collect class names from the first root
        class_dirs = sorted(
            [d for d in os.listdir(roots[0])
             if os.path.isdir(os.path.join(roots[0], d)) and not d.startswith(".")]
        )
        self.class_names = class_dirs
        self.class_to_idx = {c: i for i, c in enumerate(class_dirs)}

        for root in roots:
            for cls_name, idx in self.class_to_idx.items():
                cls_dir = os.path.join(root, cls_name)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in os.listdir(cls_dir):
                    if Path(fname).suffix.lower() in VALID_EXTS:
                        self.samples.append((os.path.join(cls_dir, fname), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def build_model(num_classes: int):
    """Create an EfficientNet-B0 with a fresh classifier head."""
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
    return model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train rice leaf disease classifier")
    parser.add_argument("--data1", default="Rice Leaf Dataset1",
                        help="Path to first dataset root")
    parser.add_argument("--data2", default="Rice Leaf Dataset2",
                        help="Path to second dataset root")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    # ---- dataset ----
    full_dataset = RiceLeafDataset(
        roots=[args.data1, args.data2],
        transform=None,  # applied later via wrapper
    )
    class_names = full_dataset.class_names
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Total images: {len(full_dataset)}")

    # ---- stratified split 80/10/10 ----
    labels = [s[1] for s in full_dataset.samples]
    indices = list(range(len(full_dataset)))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=SEED
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=SEED
    )

    # Wrap subsets with appropriate transforms
    class TransformSubset(Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            path, label = self.dataset.samples[self.indices[idx]]
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label

    train_ds = TransformSubset(full_dataset, train_idx, train_transform)
    val_ds   = TransformSubset(full_dataset, val_idx,   val_transform)
    test_ds  = TransformSubset(full_dataset, test_idx,  val_transform)

    print(f"Split → train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # ---- model ----
    model = build_model(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- training loop ----
    best_val_acc = 0.0
    best_state = None
    print("\n" + "=" * 60)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        print(
            f"Epoch {epoch}/{args.epochs}  │  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  │  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
    print("=" * 60)

    # ---- test evaluation ----
    model.load_state_dict(best_state)
    test_loss, test_acc, preds, gts = evaluate(model, test_loader, criterion, DEVICE)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(gts, preds, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(gts, preds))

    # ---- save ----
    model_path = os.path.join(args.output_dir, "rice_disease_model.pth")
    torch.save(best_state, model_path)
    print(f"\nModel saved → {model_path}")

    class_path = os.path.join(args.output_dir, "class_names.json")
    with open(class_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Class names saved → {class_path}")

if __name__ == "__main__":
    main()
