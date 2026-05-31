"""
Fine-tune SegFormer-B0 on UAVid dataset.
Optimized for CPU training with mixed precision emulation, gradient accumulation,
learning rate warmup + cosine decay, class-weighted loss, and early stopping.

Run: python src/train_segformer.py
"""

import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))
from config import (
    TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS,
    CLASS_MAP, CLASS_NAMES, CLASS_COLORS, OUTPUT_DIR, TOLERANCE, NUM_CLASSES
)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

TRAIN_CONFIG = {
    "image_size":          512,
    "batch_size":          2,         # keep low for CPU RAM
    "gradient_accum":      4,         # effective batch = 2*4 = 8
    "num_epochs":          20,
    "lr":                  6e-5,
    "warmup_epochs":       3,
    "weight_decay":        0.01,
    "early_stop_patience": 8,
    "num_workers":         0,         # 0 = main process only (Windows safe)
    "seed":                42,
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def color_to_mask(label_rgb: np.ndarray) -> np.ndarray:
    h, w   = label_rgb.shape[:2]
    mask   = np.zeros((h, w), dtype=np.int64)
    tol    = TOLERANCE
    for cls_id, (_, color) in enumerate(CLASS_MAP.items()):
        m = (
            (np.abs(label_rgb[:, :, 0].astype(int) - color[0]) <= tol) &
            (np.abs(label_rgb[:, :, 1].astype(int) - color[1]) <= tol) &
            (np.abs(label_rgb[:, :, 2].astype(int) - color[2]) <= tol)
        )
        mask[m] = cls_id
    return mask


class UAVidDataset(Dataset):
    def __init__(self, img_dir: Path, lbl_dir: Path, size: int = 512, augment: bool = True):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.size    = size
        self.augment = augment
        self.samples = sorted([p for p in img_dir.glob("*.png")
                                if (lbl_dir / p.name).exists()])
        print(f"  Dataset ({img_dir.parent.name}): {len(self.samples)} samples")

        self.img_tf = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])
        self.lbl_tf = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        lbl_path = self.lbl_dir / img_path.name

        image  = Image.open(img_path).convert("RGB")
        label  = Image.open(lbl_path).convert("RGB")

        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
            # Random vertical flip
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                label = label.transpose(Image.FLIP_TOP_BOTTOM)
            # Random crop + resize
            if random.random() > 0.4:
                w, h   = image.size
                scale  = random.uniform(0.75, 1.0)
                cw, ch = int(w * scale), int(h * scale)
                x0     = random.randint(0, w - cw)
                y0     = random.randint(0, h - ch)
                image  = image.crop((x0, y0, x0 + cw, y0 + ch))
                label  = label.crop((x0, y0, x0 + cw, y0 + ch))
            # Color jitter on image only
            if random.random() > 0.5:
                image = transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
                )(image)

        label_resized = self.lbl_tf(label)
        label_arr     = color_to_mask(np.array(label_resized))
        image_tensor  = self.img_tf(image)
        label_tensor  = torch.from_numpy(label_arr).long()

        return image_tensor, label_tensor


def compute_class_weights(dataset: UAVidDataset, num_classes: int = 8) -> torch.Tensor:
    print("  Computing class weights from training labels...")
    freq = np.zeros(num_classes, dtype=np.float64)
    for _, lbl in tqdm(dataset, desc="  Counting pixels"):
        for c in range(num_classes):
            freq[c] += (lbl == c).sum().item()
    total   = freq.sum()
    weights = total / (num_classes * freq + 1e-6)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.5, 10.0)
    print(f"  Class weights:")
    for i, (cls, w) in enumerate(zip(CLASS_NAMES, weights)):
        print(f"    {cls:<22}: {w:.4f}")
    return torch.tensor(weights, dtype=torch.float32)


def dice_loss(pred_softmax: torch.Tensor, target: torch.Tensor, num_classes: int = 8, smooth: float = 1.0):
    total = 0.0
    for c in range(num_classes):
        pred_c   = pred_softmax[:, c]
        target_c = (target == c).float()
        inter    = (pred_c * target_c).sum()
        union    = pred_c.sum() + target_c.sum()
        total   += 1.0 - (2.0 * inter + smooth) / (union + smooth)
    return total / num_classes


def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 8):
    ious = []
    pred_np   = pred.cpu().numpy().flatten()
    target_np = target.cpu().numpy().flatten()
    for c in range(num_classes):
        p = (pred_np == c)
        g = (target_np == c)
        inter = (p & g).sum()
        union = (p | g).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def get_cosine_lr(epoch, total_epochs, warmup_epochs, base_lr):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))


def build_model(num_classes: int = 8):
    from transformers import SegformerForSemanticSegmentation, SegformerConfig
    print("  Loading SegFormer-B0 from HuggingFace (nvidia/mit-b0)...")
    config = SegformerConfig.from_pretrained(
        "nvidia/mit-b0",
        num_labels=num_classes,
        id2label={str(i): n for i, n in enumerate(CLASS_NAMES)},
        label2id={n: i for i, n in enumerate(CLASS_NAMES)},
        ignore_mismatched_sizes=True,
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        config=config,
        ignore_mismatched_sizes=True,
    )
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params   : {total_params/1e6:.2f}M")
    print(f"  Trainable      : {train_params/1e6:.2f}M")
    return model


def train():
    set_seed(TRAIN_CONFIG["seed"])
    device = torch.device("cpu")
    print(f"\nSegFormer-B0 Fine-tuning on UAVid")
    print("=" * 60)
    print(f"  Device         : {device}")
    print(f"  Epochs         : {TRAIN_CONFIG['num_epochs']}")
    print(f"  Batch size     : {TRAIN_CONFIG['batch_size']} (effective: {TRAIN_CONFIG['batch_size'] * TRAIN_CONFIG['gradient_accum']})")
    print(f"  Image size     : {TRAIN_CONFIG['image_size']}px")
    print(f"  Learning rate  : {TRAIN_CONFIG['lr']}")
    print(f"  Warmup epochs  : {TRAIN_CONFIG['warmup_epochs']}")

    print("\nBuilding datasets...")
    train_ds = UAVidDataset(TRAIN_IMAGES, TRAIN_LABELS, TRAIN_CONFIG["image_size"], augment=True)
    val_ds   = UAVidDataset(VAL_IMAGES,   VAL_LABELS,   TRAIN_CONFIG["image_size"], augment=False)

    train_loader = DataLoader(train_ds, batch_size=TRAIN_CONFIG["batch_size"],
                               shuffle=True,  num_workers=TRAIN_CONFIG["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=1,
                               shuffle=False, num_workers=TRAIN_CONFIG["num_workers"])

    print("\nComputing class weights...")
    class_weights = compute_class_weights(train_ds)

    print("\nBuilding model...")
    model = build_model()
    model.to(device)

    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device), ignore_index=255)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )

    history     = {"train_loss": [], "val_miou": [], "val_pix_acc": [], "lr": []}
    best_miou   = 0.0
    patience_ct = 0
    best_path   = CHECKPOINT_DIR / "segformer_b0_uavid_best.pth"
    grad_accum  = TRAIN_CONFIG["gradient_accum"]

    print(f"\nStarting training...")
    print(f"  Best checkpoint: {best_path}\n")

    for epoch in range(TRAIN_CONFIG["num_epochs"]):
        t_epoch = time.time()
        lr_now  = get_cosine_lr(epoch, TRAIN_CONFIG["num_epochs"],
                                 TRAIN_CONFIG["warmup_epochs"], TRAIN_CONFIG["lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        model.train()
        train_losses = []
        optimizer.zero_grad()

        for step, (images, labels) in enumerate(tqdm(train_loader, desc=f"  Epoch {epoch+1:02d} train", leave=False)):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(pixel_values=images)
            logits  = outputs.logits

            logits_up = F.interpolate(logits, size=labels.shape[-2:],
                                       mode="bilinear", align_corners=False)
            softmax_up = F.softmax(logits_up, dim=1)

            loss_ce   = ce_loss_fn(logits_up, labels)
            loss_dice = dice_loss(softmax_up, labels)
            loss      = 0.6 * loss_ce + 0.4 * loss_dice
            loss      = loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_losses.append(loss.item() * grad_accum)

        mean_train_loss = float(np.mean(train_losses))

        model.eval()
        val_ious, val_accs = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"  Epoch {epoch+1:02d} val  ", leave=False):
                images = images.to(device)
                labels = labels.to(device)
                outputs   = model(pixel_values=images)
                logits_up = F.interpolate(outputs.logits, size=labels.shape[-2:],
                                           mode="bilinear", align_corners=False)
                preds = logits_up.argmax(dim=1)
                val_ious.append(compute_iou(preds.squeeze(), labels.squeeze()))
                correct = (preds == labels).float().mean().item()
                val_accs.append(correct)

        val_miou   = float(np.mean(val_ious))
        val_pix    = float(np.mean(val_accs)) * 100
        epoch_time = time.time() - t_epoch

        history["train_loss"].append(mean_train_loss)
        history["val_miou"].append(val_miou * 100)
        history["val_pix_acc"].append(val_pix)
        history["lr"].append(lr_now)

        improved = val_miou > best_miou
        if improved:
            best_miou   = val_miou
            patience_ct = 0
            torch.save({
                "epoch":      epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_miou":   best_miou,
                "config":     TRAIN_CONFIG,
            }, best_path)
            flag = " <-- BEST"
        else:
            patience_ct += 1
            flag = ""

        print(f"  Epoch {epoch+1:02d}/{TRAIN_CONFIG['num_epochs']} | "
              f"loss={mean_train_loss:.4f} | "
              f"mIoU={val_miou*100:.2f}% | "
              f"pix_acc={val_pix:.2f}% | "
              f"lr={lr_now:.2e} | "
              f"{epoch_time:.0f}s{flag}")

        if patience_ct >= TRAIN_CONFIG["early_stop_patience"]:
            print(f"\n  Early stopping at epoch {epoch+1} (no improvement for {patience_ct} epochs)")
            break

    print(f"\nTraining complete.")
    print(f"  Best val mIoU  : {best_miou*100:.4f}%")
    print(f"  Checkpoint     : {best_path}")

    plot_training_history(history)
    save_training_summary(history, best_miou)
    return best_path


def plot_training_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("SegFormer-B0 Training History", fontsize=12, fontweight="bold")

    axes[0].plot(history["train_loss"], color="#e63946", linewidth=1.5, label="Train Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss (CE + Dice)")
    axes[0].set_title("Training Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history["val_miou"], color="#2196F3", linewidth=1.5, label="Val mIoU")
    axes[1].plot(history["val_pix_acc"], color="#4CAF50", linewidth=1.5, linestyle="--", label="Val Pix Acc")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score (%)")
    axes[1].set_title("Validation Metrics"); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(history["lr"], color="#FF9800", linewidth=1.5)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("LR Schedule (Warmup + Cosine)"); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "training_history.png"
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Training history chart: {save_path}")


def save_training_summary(history, best_miou):
    summary = {
        "model":         "SegFormer-B0 (nvidia/mit-b0)",
        "dataset":       "Modified UAVid",
        "num_classes":   8,
        "best_val_miou": round(best_miou * 100, 4),
        "final_train_loss": round(history["train_loss"][-1], 6),
        "epochs_trained":   len(history["train_loss"]),
        "config":           TRAIN_CONFIG,
        "class_names":      CLASS_NAMES,
    }
    path = OUTPUT_DIR / "training_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Training summary: {path}")


if __name__ == "__main__":
    train()