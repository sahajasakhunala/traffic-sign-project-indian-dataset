import csv
import os
import time
import collections
import math
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

from model import TrafficSignCNN

# ── Paths ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=os.path.join("data", "Indian_Dataset"), help='Path to dataset images folder')
parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained GTSRB model checkpoint (.pth)')
parser.add_argument('--model_type', type=str, default='custom_cnn', choices=['custom_cnn', 'resnet50'], help='Model architecture to use')
parser.add_argument('--image_size', type=int, default=64, help='Image height and width for training')
parser.add_argument('--checkpoint_dir', type=str, default="models", help='Directory to save checkpoints')
parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
parser.add_argument('--mixup_alpha', type=float, default=0.3, help='Mixup alpha (0.0 to disable)')
parser.add_argument('--cutmix_alpha', type=float, default=1.0, help='Cutmix alpha (0.0 to disable)')
args = parser.parse_args()

DATA_DIR   = args.data_dir
MODEL_DIR  = args.checkpoint_dir

# Suffix paths if we are fine-tuning, so we don't overwrite the scratch run!
suffix = f"_{args.model_type}_finetuned" if args.pretrained_path else f"_{args.model_type}"
MODEL_PATH = os.path.join(MODEL_DIR, f"traffic_sign{suffix}.pth")
LAST_PATH  = os.path.join(MODEL_DIR, f"traffic_sign{suffix}_last.pth")
LOG_PATH   = os.path.join(MODEL_DIR, f"training_log{suffix}.csv")

# ── Hyperparameters ────────────────────────────────────────────────────────────
IMAGE_SIZE        = args.image_size
BATCH_SIZE        = 64
LR                = 3e-4          # Lower base LR; warmup handles ramp-up
EPOCHS            = 30            # More epochs — Indian data benefits from longer training
WARMUP_EPOCHS     = 3             # Linear warmup to stabilise early gradient norms
VAL_SPLIT         = 0.15
EARLY_STOP_PAT    = 8             # More patience — Indian dataset is noisier
GRAD_CLIP         = 1.0           # Tighter clip for stability on noisy labels
NUM_WORKERS       = min(4, os.cpu_count() or 1)
SEED              = 42
LABEL_SMOOTHING   = 0.10          # Stronger smoothing for noisy/ambiguous signs
MIXUP_ALPHA       = args.mixup_alpha
CUTMIX_ALPHA      = args.cutmix_alpha
USE_WEIGHTED_SAMPLER = True       # UPDATED: Fix class-imbalance with per-sample weights

# ── Device ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Reproducibility ────────────────────────────────────────────────────────────
torch.manual_seed(SEED)
if device.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True


# ── Transforms ─────────────────────────────────────────────────────────────────
# UPDATED: Heavier augmentation targeting real Indian road conditions:
#   • Stronger colour jitter  → handles dust haze, glare, and mixed lighting
#   • GaussianBlur            → motion blur from fast-moving vehicles
#   • RandomPerspective       → signs viewed at sharp angles (elevated, tilted posts)
#   • RandomGrayscale         → teaches colour-invariant shape features
#   • RandomErasing           → occlusion by vehicles, foliage, stickers
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 8, IMAGE_SIZE + 8)),   # slight oversize for random crop
    transforms.RandomCrop(IMAGE_SIZE),                      # random-crop replaces simple resize
    # Removed RandomHorizontalFlip(p=0.1) to avoid confusing left/right directional signs (e.g. 23 <-> 24)
    transforms.RandomRotation(degrees=20),                  # UPDATED: wider rotation
    transforms.RandomAffine(
        degrees=0,
        translate=(0.12, 0.12),                             # UPDATED: slightly more shift
        scale=(0.85, 1.15),                                 # UPDATED: wider scale range
        shear=8,                                            # UPDATED: more shear
    ),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.4),  # UPDATED: perspective warp
    transforms.ColorJitter(
        brightness=0.5,                                     # UPDATED: stronger — sunlight/shadow
        contrast=0.4,                                       # UPDATED
        saturation=0.4,                                     # UPDATED
        hue=0.08,                                           # UPDATED
    ),
    transforms.RandomGrayscale(p=0.08),                    # UPDATED: colour-invariant features
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),   # UPDATED: motion/rain blur
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    # UPDATED: Random erasing simulates stickers, damage, partial occlusion
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.0), value=0),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ── Dataset helpers ────────────────────────────────────────────────────────────
class TransformSubset(torch.utils.data.Dataset):
    """Wraps a Subset and applies an independent transform."""

    def __init__(self, subset: torch.utils.data.Subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        from PIL import Image
        image = Image.open(path).convert("RGB")
        return self.transform(image), label

    def get_labels(self) -> list[int]:
        """Return all labels — needed for WeightedRandomSampler."""
        return [self.subset.dataset.samples[i][1] for i in self.subset.indices]


def stratified_split(
    dataset:   datasets.ImageFolder,
    val_split: float,
    seed:      int,
) -> tuple[list[int], list[int]]:
    rng = torch.Generator().manual_seed(seed)
    class_indices: dict[int, list[int]] = collections.defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)

    train_indices, val_indices = [], []
    for label in sorted(class_indices):
        idxs    = class_indices[label]
        perm    = torch.randperm(len(idxs), generator=rng).tolist()
        idxs    = [idxs[i] for i in perm]
        n_val_c = max(1, int(len(idxs) * val_split))
        val_indices.extend(idxs[:n_val_c])
        train_indices.extend(idxs[n_val_c:])

    return train_indices, val_indices


# UPDATED: Builds a WeightedRandomSampler to over-sample minority classes.
def make_weighted_sampler(dataset: TransformSubset) -> WeightedRandomSampler:
    labels          = dataset.get_labels()
    class_counts    = collections.Counter(labels)
    num_classes     = len(class_counts)
    class_weights   = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights  = [class_weights[lbl] for lbl in labels]
    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True,
    )
    return sampler


def build_loaders(
    data_dir: str,
    val_split: float,
) -> tuple[DataLoader, DataLoader, int, list[str]]:
    base_dataset = datasets.ImageFolder(root=data_dir)
    class_names  = base_dataset.classes
    num_classes  = len(class_names)

    train_indices, val_indices = stratified_split(base_dataset, val_split, SEED)

    train_set = TransformSubset(Subset(base_dataset, train_indices), train_transform)
    val_set   = TransformSubset(Subset(base_dataset, val_indices),   val_transform)

    _loader_kwargs = dict(
        batch_size  = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        pin_memory  = (device.type == "cuda"),
        persistent_workers = (NUM_WORKERS > 0),
    )

    # UPDATED: Replace shuffle=True with a class-balancing sampler
    if USE_WEIGHTED_SAMPLER:
        sampler      = make_weighted_sampler(train_set)
        train_loader = DataLoader(train_set, sampler=sampler, **_loader_kwargs)
    else:
        train_loader = DataLoader(train_set, shuffle=True, **_loader_kwargs)

    val_loader = DataLoader(val_set, shuffle=False, **_loader_kwargs)

    print(f"  Dataset   : {data_dir}")
    print(f"  Classes   : {num_classes}")
    print(f"  Train     : {len(train_indices):,}  ({100*(1-val_split):.0f}%)")
    print(f"  Val       : {len(val_indices):,}  ({100*val_split:.0f}%)")
    return train_loader, val_loader, num_classes, class_names


# ── Mixup / CutMix helpers ─────────────────────────────────────────────────────
# UPDATED: Mixup and CutMix are strong regularisers for small/noisy datasets.
def mixup_data(x, y, alpha: float, device: torch.device):
    """Returns mixed inputs, pairs of targets, and lambda."""
    lam = torch.distributions.Beta(alpha, alpha).sample().item() if alpha > 0 else 1.0
    batch_size = x.size(0)
    index      = torch.randperm(batch_size, device=device)
    mixed_x    = lam * x + (1 - lam) * x[index]
    y_a, y_b   = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha: float, device: torch.device):
    """Returns CutMix inputs, pairs of targets, and lambda."""
    lam   = torch.distributions.Beta(alpha, alpha).sample().item() if alpha > 0 else 1.0
    batch = x.size(0)
    index = torch.randperm(batch, device=device)

    _, _, H, W = x.shape
    cut_rat    = math.sqrt(1.0 - lam)
    cut_h      = int(H * cut_rat)
    cut_w      = int(W * cut_rat)

    cx = torch.randint(W, (1,)).item()
    cy = torch.randint(H, (1,)).item()
    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, H)

    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return x, y, y[index], lam


def mixed_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── LR schedule helpers ────────────────────────────────────────────────────────
# UPDATED: Cosine annealing with linear warmup prevents the large early gradient
# norms that destabilise training on the noisier Indian dataset.
def get_lr(epoch: int, warmup_epochs: int, total_epochs: int, base_lr: float) -> float:
    if epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return base_lr * (1 + math.cos(math.pi * progress)) / 2


# ── Train / evaluate loops ─────────────────────────────────────────────────────
def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device:    torch.device,
    epoch:     int = 0,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train() if training else model.eval()

    running_loss = 0.0
    correct      = 0
    total        = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

                # UPDATED: Randomly pick Mixup or CutMix per batch for variety
                r = torch.rand(1).item()
                if MIXUP_ALPHA > 0 and r < 0.4:
                    images, y_a, y_b, lam = mixup_data(images, labels, MIXUP_ALPHA, device)
                    outputs = model(images)
                    loss    = mixed_criterion(criterion, outputs, y_a, y_b, lam)
                elif CUTMIX_ALPHA > 0 and r < 0.7:
                    images, y_a, y_b, lam = cutmix_data(images, labels, CUTMIX_ALPHA, device)
                    outputs = model(images)
                    loss    = mixed_criterion(criterion, outputs, y_a, y_b, lam)
                else:
                    outputs = model(images)
                    loss    = criterion(outputs, labels)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            else:
                outputs = model(images)
                loss    = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted  = outputs.max(1)
            total        += labels.size(0)
            correct      += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 72)
    print("  Traffic Sign Recognition — Training (Indian Dataset)")
    print("=" * 72)
    print(f"  Device      : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"  Batch size  : {BATCH_SIZE}")
    print(f"  Epochs      : {EPOCHS}  (warmup: {WARMUP_EPOCHS})")
    print(f"  LR          : {LR}  |  weight_decay : 2e-4")
    print(f"  Grad clip   : {GRAD_CLIP}  |  early-stop patience : {EARLY_STOP_PAT}")
    print(f"  Label smooth: {LABEL_SMOOTHING}  |  Mixup α: {MIXUP_ALPHA}  |  CutMix α: {CUTMIX_ALPHA}")
    print(f"  Val split   : {VAL_SPLIT:.0%}  |  seed : {SEED}")
    print("-" * 72)

    train_loader, val_loader, num_classes, class_names = build_loaders(DATA_DIR, VAL_SPLIT)
    print("-" * 72)

    if args.model_type == "resnet50":
        import torchvision.models as tv_models
        print("  Initializing ResNet50 architecture...")
        model = tv_models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        model = TrafficSignCNN(num_classes=num_classes)
        
    model = model.to(device)
    
    resumed = False
    start_epoch = 1
    best_val_acc   = 0.0
    patience_count = 0
    history: list[dict] = []

    # Check if we should resume
    if args.resume and os.path.exists(LAST_PATH):
        print(f"  Resumed checkpoint found at: {LAST_PATH}")
        resumed = True
    
    # Load pretrained weights ONLY if not resuming
    if args.pretrained_path and not resumed:
        if os.path.exists(args.pretrained_path):
            print(f"  Loading pretrained weights from: {args.pretrained_path}")
            checkpoint = torch.load(args.pretrained_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
            else:
                state_dict = checkpoint
            
            # Since number of classes is different (e.g. 43 in GTSRB vs 58 in Indian), 
            # we skip loading the final linear layer weight/bias.
            filtered_state_dict = {}
            for k, v in state_dict.items():
                # Skip the classification head depending on the architecture
                if args.model_type == "resnet50":
                    if k.startswith("fc."):
                        continue
                else:
                    if "classifier.12" in k or "classifier.13" in k:  # Last layer in custom sequential head
                        continue
                        
                # Compatibility check for keys
                if k in model.state_dict() and model.state_dict()[k].shape == v.shape:
                    filtered_state_dict[k] = v
                else:
                    print(f"    Skipping key: {k} (shape mismatch or not in model)")
            
            # Load the matching weights
            msg = model.load_state_dict(filtered_state_dict, strict=False)
            print(f"  Pretrained load status: {msg}")
        else:
            print(f"  [Warning] Pretrained file not found at: {args.pretrained_path}. Training from scratch.")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=2e-4)

    # Now load resume state if applicable
    if resumed:
        print(f"  Loading last checkpoint state...")
        checkpoint = torch.load(LAST_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("best_val_acc", checkpoint.get("val_acc", 0.0))
        patience_count = checkpoint.get("patience_count", 0)
        history = checkpoint.get("history", [])
        print(f"  Resumed successfully! Starting from Epoch {start_epoch} (Best Val Acc so far: {best_val_acc:.2f}%)")

    header = (
        f"{'Epoch':>7}  {'T-Loss':>8}  {'T-Acc':>7}  "
        f"{'V-Loss':>8}  {'V-Acc':>7}  {'LR':>10}  {'Time':>7}"
    )
    print(header)
    print("-" * len(header))

    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.perf_counter()

        # UPDATED: Manual LR with warmup; set before the forward pass
        current_lr = get_lr(epoch, WARMUP_EPOCHS, EPOCHS, LR)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, None,      device, epoch)

        elapsed = time.perf_counter() - t0

        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  4),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,    4),
            "lr":         round(current_lr, 8),
            "time_s":     round(elapsed,    2),
        })

        # ── Checkpoint ──────────────────────────────────────────────────────────
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc   = val_acc
            patience_count = 0
            torch.save(
                {
                    "epoch":          epoch,
                    "state_dict":     model.state_dict(),
                    "optimizer":      optimizer.state_dict(),
                    "val_acc":        val_acc,
                    "val_loss":       val_loss,
                    "best_val_acc":   best_val_acc,
                    "patience_count": patience_count,
                    "history":        history,
                    "class_names":    class_names,
                    "num_classes":    num_classes,
                    "image_size":     IMAGE_SIZE,
                },
                MODEL_PATH,
            )
        else:
            patience_count += 1

        # Save last checkpoint for resuming
        torch.save(
            {
                "epoch":          epoch,
                "state_dict":     model.state_dict(),
                "optimizer":      optimizer.state_dict(),
                "val_acc":        val_acc,
                "val_loss":       val_loss,
                "best_val_acc":   best_val_acc,
                "patience_count": patience_count,
                "history":        history,
                "class_names":    class_names,
                "num_classes":    num_classes,
                "image_size":     IMAGE_SIZE,
            },
            LAST_PATH,
        )

        marker = " ✓" if improved else ""
        print(
            f"{epoch:>6}/{EPOCHS}"
            f"  {train_loss:>8.4f}"
            f"  {train_acc:>6.2f}%"
            f"  {val_loss:>8.4f}"
            f"  {val_acc:>6.2f}%"
            f"  {current_lr:>10.6f}"
            f"  {elapsed:>6.1f}s"
            f"{marker}"
        )

        if patience_count >= EARLY_STOP_PAT:
            print(f"\n  Early stopping triggered — no improvement for {EARLY_STOP_PAT} epochs.")
            break

    # ── Save log ──────────────────────────────────────────────────────────────
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print("=" * 72)
    print(f"  Best val accuracy : {best_val_acc:.2f}%")
    print(f"  Model saved       : {MODEL_PATH}")
    print(f"  Training log      : {LOG_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()