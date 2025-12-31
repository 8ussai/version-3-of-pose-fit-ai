import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    CKPT_DIR, CLASSES, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY,
    NUM_WORKERS, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, VIDEOS_DIR, LABELS_CSV,
    SEED, LR_PATIENCE, LR_FACTOR
)
from src.utils import set_seed, ensure_dir
from src.model_twostream import TwoStreamFusionNet
from src.dataset_video_pose_multilabel import (
    read_labels_csv, split_by_videos, VideoPoseMultiLabelDataset
)


def batch_f1_micro(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).float()
    tp = (y_pred * y_true).sum()
    fp = (y_pred * (1 - y_true)).sum()
    fn = ((1 - y_pred) * y_true).sum()
    denom = (2 * tp + fp + fn).clamp(min=1e-8)
    return ((2 * tp) / denom).item()


def batch_accuracy(y_true, y_prob, thr=0.5):
    """
    element-wise accuracy: متوسط (pred==true) لكل العناصر B*C
    subset accuracy: العينة تعتبر صح فقط إذا كل الكلاسات تطابقت (صارمة)
    """
    y_pred = (y_prob >= thr).float()
    elem_acc = (y_pred == y_true).float().mean().item()
    subset_acc = (y_pred == y_true).all(dim=1).float().mean().item()
    return elem_acc, subset_acc


def compute_pos_weight(items):
    """
    items: list[(fname, y)] where y shape = (C,)
    pos_weight = (N_neg / N_pos) لكل كلاس (مفيد للـimbalance)
    """
    ys = torch.tensor([y for _, y in items], dtype=torch.float32)  # (N,C)
    n = ys.shape[0]
    pos = ys.sum(dim=0)                        # (C,)
    neg = n - pos
    pos_weight = (neg / pos.clamp(min=1.0))    # avoid div0
    return pos_weight, pos, neg, n


@torch.no_grad()
def validate(model, loader, device, loss_fn, thr=0.5):
    model.eval()
    total_loss, total_f1, total_eacc, total_sacc = 0.0, 0.0, 0.0, 0.0
    n_batches = 0

    for frames, pose, y in loader:
        frames, pose, y = frames.to(device), pose.to(device), y.to(device)
        logits = model(frames, pose)
        loss = loss_fn(logits, y)

        probs = torch.sigmoid(logits)
        f1 = batch_f1_micro(y, probs, thr=thr)
        eacc, sacc = batch_accuracy(y, probs, thr=thr)

        total_loss += loss.item()
        total_f1 += f1
        total_eacc += eacc
        total_sacc += sacc
        n_batches += 1

    n_batches = max(n_batches, 1)
    return (
        total_loss / n_batches,
        total_f1 / n_batches,
        total_eacc / n_batches,
        total_sacc / n_batches,
    )


def train_one_epoch(model, loader, optimizer, device, loss_fn, thr=0.5):
    model.train()
    total_loss, total_f1, total_eacc, total_sacc = 0.0, 0.0, 0.0, 0.0
    n_batches = 0

    for frames, pose, y in tqdm(loader, desc="Train", leave=False):
        frames, pose, y = frames.to(device), pose.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(frames, pose)
        loss = loss_fn(logits, y)
        loss.backward()

        # (اختياري لكنه مفيد مع GRU)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        probs = torch.sigmoid(logits)
        f1 = batch_f1_micro(y, probs, thr=thr)
        eacc, sacc = batch_accuracy(y, probs, thr=thr)

        total_loss += loss.item()
        total_f1 += f1
        total_eacc += eacc
        total_sacc += sacc
        n_batches += 1

    n_batches = max(n_batches, 1)
    return (
        total_loss / n_batches,
        total_f1 / n_batches,
        total_eacc / n_batches,
        total_sacc / n_batches,
    )


def main():
    set_seed(SEED)
    ensure_dir(CKPT_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    items = read_labels_csv(LABELS_CSV)
    train_items, val_items, test_items = split_by_videos(
        items, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, seed=SEED
    )

    # ======= توزيع الكلاسات من CSV (train فقط) =======
    pos_weight, pos, neg, n_train = compute_pos_weight(train_items)
    print("\n=== Train class distribution (from CSV) ===")
    for i, cls in enumerate(CLASSES):
        p = int(pos[i].item())
        ng = int(neg[i].item())
        rate = (p / max(n_train, 1)) * 100.0
        print(f"- {cls:14s}: pos={p:4d} neg={ng:4d} pos_rate={rate:5.1f}%  pos_weight={pos_weight[i].item():.4f}")

    # loss with pos_weight
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, verbose=True
    )

    best_val_f1 = -1.0
    best_path = CKPT_DIR / "squats_twostream_gru.pt"

    print(f"\nTrain videos: {len(train_ds)} | Val videos: {len(val_ds)}")
    print(f"Batch={BATCH_SIZE} | Epochs={EPOCHS} | LR={LR}")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_f1, tr_eacc, tr_sacc = train_one_epoch(model, train_loader, optimizer, device, loss_fn, thr=0.5)
        va_loss, va_f1, va_eacc, va_sacc = validate(model, val_loader, device, loss_fn, thr=0.5)

        scheduler.step(va_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        # ======= طباعة Accuracy بعد كل epoch =======
        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train_loss={tr_loss:.4f} train_f1={tr_f1:.4f} "
            f"train_acc(elem)={tr_eacc:.4f} train_acc(subset)={tr_sacc:.4f} | "
            f"val_loss={va_loss:.4f} val_f1={va_f1:.4f} "
            f"val_acc(elem)={va_eacc:.4f} val_acc(subset)={va_sacc:.4f} | "
            f"lr={lr_now:.2e}"
        )

        # حفظ أفضل موديل (بدون EarlyStopping)
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            torch.save({
                "model_state": model.state_dict(),
                "classes": CLASSES,
                "threshold": 0.5
            }, best_path)
            print(f"✅ Saved best -> {best_path} (val F1={best_val_f1:.4f})")

    print("Done.")


if __name__ == "__main__":
    main()
