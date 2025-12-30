import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    CKPT_DIR, CLASSES, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY,
    NUM_WORKERS, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, VIDEOS_DIR, LABELS_CSV,
    SEED, PATIENCE, MIN_DELTA, LR_PATIENCE, LR_FACTOR
)
from src.utils import set_seed, ensure_dir
from src.model_twostream import TwoStreamFusionNet
from src.dataset_video_pose_multilabel import (
    read_labels_csv, split_by_videos, VideoPoseMultiLabelDataset
)

def batch_f1_micro(y_true, y_prob, thr=0.5):
    """
    y_true: (B,C) 0/1
    y_prob: (B,C) probabilities [0,1]
    """
    y_pred = (y_prob >= thr).float()
    tp = (y_pred * y_true).sum()
    fp = (y_pred * (1 - y_true)).sum()
    fn = ((1 - y_pred) * y_true).sum()
    denom = (2 * tp + fp + fn).clamp(min=1e-8)
    f1 = (2 * tp) / denom
    return f1.item()

@torch.no_grad()
def validate(model, loader, device, thr=0.5):
    model.eval()
    bce = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_f1 = 0.0
    n_batches = 0

    for frames, pose, y in loader:
        frames, pose, y = frames.to(device), pose.to(device), y.to(device)
        logits = model(frames, pose)
        loss = bce(logits, y)

        probs = torch.sigmoid(logits)
        f1 = batch_f1_micro(y, probs, thr=thr)

        total_loss += loss.item()
        total_f1 += f1
        n_batches += 1

    return total_loss / max(n_batches, 1), total_f1 / max(n_batches, 1)

def train_one_epoch(model, loader, optimizer, device, thr=0.5):
    model.train()
    bce = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_f1 = 0.0
    n_batches = 0

    for frames, pose, y in tqdm(loader, desc="Train", leave=False):
        frames, pose, y = frames.to(device), pose.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(frames, pose)
        loss = bce(logits, y)
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits)
        f1 = batch_f1_micro(y, probs, thr=thr)

        total_loss += loss.item()
        total_f1 += f1
        n_batches += 1

    return total_loss / max(n_batches, 1), total_f1 / max(n_batches, 1)

def main():
    set_seed(SEED)
    ensure_dir(CKPT_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    items = read_labels_csv(LABELS_CSV)
    train_items, val_items, test_items = split_by_videos(
        items, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, seed=SEED
    )

    train_ds = VideoPoseMultiLabelDataset(VIDEOS_DIR, train_items, augment=True)
    val_ds = VideoPoseMultiLabelDataset(VIDEOS_DIR, val_items, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device == "cuda")
    )

    model = TwoStreamFusionNet(num_classes=len(CLASSES), pretrained_backbone=False).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Optional: ReduceLROnPlateau على val_loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, verbose=True
    )

    best_val = -1.0
    best_path = CKPT_DIR / "squats_twostream_gru.pt"

    # EarlyStopping
    no_improve = 0

    print(f"Train videos: {len(train_ds)} | Val videos: {len(val_ds)}")
    print(f"Batch={BATCH_SIZE} | Epochs={EPOCHS} | LR={LR} | Patience={PATIENCE}")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_f1 = train_one_epoch(model, train_loader, optimizer, device, thr=0.5)
        va_loss, va_f1 = validate(model, val_loader, device, thr=0.5)

        # Scheduler خطوة على val_loss
        scheduler.step(va_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train_loss={tr_loss:.4f} train_f1={tr_f1:.4f} | "
            f"val_loss={va_loss:.4f} val_f1={va_f1:.4f} | lr={lr_now:.2e}"
        )

        # Save best + EarlyStopping على val_f1
        if va_f1 > best_val + MIN_DELTA:
            best_val = va_f1
            no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "classes": CLASSES,
                "threshold": 0.5
            }, best_path)
            print(f"✅ Saved best -> {best_path} (val F1={best_val:.4f})")
        else:
            no_improve += 1
            print(f"⏳ No improvement: {no_improve}/{PATIENCE}")

            if no_improve >= PATIENCE:
                print(f"🛑 EarlyStopping: val F1 لم يتحسن لمدة {PATIENCE} epochs. Stopping.")
                break

    print("Done.")

if __name__ == "__main__":
    main()
