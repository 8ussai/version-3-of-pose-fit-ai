# src/eval.py

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# اختياري: لو مش مركب seaborn، رح نرسم بدون heatmap
try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

from sklearn.metrics import (
    classification_report,
    multilabel_confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)

from src.config import (
    BATCH_SIZE, NUM_WORKERS, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    CKPT_DIR, CLASSES, VIDEOS_DIR, LABELS_CSV, SEED
)
from src.utils import set_seed
from src.dataset_video_pose_multilabel import read_labels_csv, split_by_videos, VideoPoseMultiLabelDataset
from src.model_twostream import TwoStreamFusionNet


def _safe_name(s: str) -> str:
    return s.strip().replace(" ", "_").replace("/", "_")


@torch.no_grad()
def evaluate(model, loader, device="cuda", threshold=0.5):
    model.eval()
    bce = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    ys, ps = [], []

    for frames, pose, y in loader:
        frames, pose, y = frames.to(device), pose.to(device), y.to(device)
        logits = model(frames, pose)
        loss = bce(logits, y)

        probs = torch.sigmoid(logits)
        ys.append(y.detach().cpu().numpy())
        ps.append(probs.detach().cpu().numpy())
        total_loss += loss.item()

    y_true = np.vstack(ys)  # (N,C)
    y_prob = np.vstack(ps)  # (N,C)
    y_pred = (y_prob >= threshold).astype(int)

    return total_loss / max(len(loader), 1), y_true, y_pred, y_prob


def save_confusion_matrices(y_true, y_pred, classes, out_dir: Path):
    mcm = multilabel_confusion_matrix(y_true, y_pred)  # (C,2,2)

    for i, cls in enumerate(classes):
        mat = mcm[i]  # [[TN,FP],[FN,TP]]

        plt.figure(figsize=(4.5, 3.6))
        if _HAS_SNS:
            sns.heatmap(
                mat,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"],
            )
        else:
            # رسم بسيط بدون seaborn
            plt.imshow(mat)
            for (r, c), v in np.ndenumerate(mat):
                plt.text(c, r, str(v), ha="center", va="center")
            plt.xticks([0, 1], ["Pred 0", "Pred 1"])
            plt.yticks([0, 1], ["True 0", "True 1"])

        plt.title(f"Confusion Matrix - {cls}")
        plt.tight_layout()
        plt.savefig(out_dir / f"cm_{_safe_name(cls)}.png", dpi=200)
        plt.close()


def save_roc_curves(y_true, y_prob, classes, out_dir: Path):
    for i, cls in enumerate(classes):
        # لو الكلاس كله 0 أو كله 1 بالتيست، roc_curve رح يفشل
        if len(np.unique(y_true[:, i])) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {cls}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(out_dir / f"roc_{_safe_name(cls)}.png", dpi=200)
        plt.close()


def save_pr_curves(y_true, y_prob, classes, out_dir: Path):
    for i, cls in enumerate(classes):
        if len(np.unique(y_true[:, i])) < 2:
            continue

        precision, recall, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])

        plt.figure(figsize=(5, 4))
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {cls}")
        plt.tight_layout()
        plt.savefig(out_dir / f"pr_{_safe_name(cls)}.png", dpi=200)
        plt.close()


def save_metric_bars(y_true, y_pred, classes, out_dir: Path):
    report = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0,
        digits=4
    )

    for metric in ["precision", "recall", "f1-score"]:
        vals = [report[c][metric] for c in classes]

        plt.figure(figsize=(7, 4))
        x = np.arange(len(classes))
        plt.bar(x, vals)
        plt.xticks(x, classes, rotation=20, ha="right")
        plt.ylim(0, 1)
        plt.title(metric.upper())
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}.png", dpi=200)
        plt.close()


def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ===== data split =====
    items = read_labels_csv(LABELS_CSV)
    train_items, val_items, test_items = split_by_videos(
        items,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
    )

    test_ds = VideoPoseMultiLabelDataset(VIDEOS_DIR, test_items, augment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )

    # ===== load model =====
    ckpt_path = CKPT_DIR / "squats_twostream_gru.pt"
    ckpt = torch.load(ckpt_path, map_location=device)

    model = TwoStreamFusionNet(num_classes=len(CLASSES), pretrained_backbone=False).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    threshold = float(ckpt.get("threshold", 0.5))
    loss, y_true, y_pred, y_prob = evaluate(model, test_loader, device=device, threshold=threshold)

    print(f"\nTest loss: {loss:.4f} | threshold={threshold}")

    print("\n=== Classification Report ===")
    print(classification_report(
        y_true, y_pred,
        target_names=CLASSES,
        digits=4,
        zero_division=0
    ))

    # ===== save plots =====
    plots_dir = CKPT_DIR / "eval"
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_confusion_matrices(y_true, y_pred, CLASSES, plots_dir)
    save_roc_curves(y_true, y_prob, CLASSES, plots_dir)
    save_pr_curves(y_true, y_prob, CLASSES, plots_dir)
    save_metric_bars(y_true, y_pred, CLASSES, plots_dir)

    print(f"\n✅ Saved evaluation plots -> {plots_dir}")


if __name__ == "__main__":
    main()
