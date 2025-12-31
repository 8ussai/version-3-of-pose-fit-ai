from pathlib import Path
import csv
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import CLASSES, IMG_SIZE, SEQ_LEN, FPS_SAMPLE, SEED
from src.mediapipe_pose import MediaPipePoseExtractor


def _csv_key_from_class(cls_name: str) -> str:
    # wrong position -> wrong_position
    return cls_name.strip().replace(" ", "_")


def read_labels_csv(csv_path: Path):
    """
    CSV format المتوقع:
      filename, correct, fast, uncomplete, wrong_position
    القيم تكون 0/1.
    """
    items = []
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("filename") or row.get("video") or row.get("path")
            if fname is None:
                raise ValueError("CSV لازم يحتوي عمود filename (أو video/path).")

            y = np.zeros((len(CLASSES),), dtype=np.float32)
            for i, cls in enumerate(CLASSES):
                # جرّب key بالunderscore أولاً، وبعدها الاسم الأصلي
                key = _csv_key_from_class(cls)
                v = row.get(key, row.get(cls, "0"))
                try:
                    y[i] = float(v)
                except Exception:
                    y[i] = 0.0

            items.append((fname, y))
    return items


def split_by_videos(items, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(items))
    rng.shuffle(idx)

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    train_items = [items[i] for i in train_idx]
    val_items = [items[i] for i in val_idx]
    test_items = [items[i] for i in test_idx]
    return train_items, val_items, test_items


class VideoPoseMultiLabelDataset(Dataset):
    def __init__(self, videos_dir: Path, items, augment=False):
        self.videos_dir = Path(videos_dir)
        self.items = items
        self.augment = augment
        self.pose_extractor = MediaPipePoseExtractor()

        if not self.videos_dir.exists():
            raise FileNotFoundError(f"Videos dir not found: {self.videos_dir}")

    def __len__(self):
        return len(self.items)

    def _read_video_frames(self, video_path: Path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 1 else 30.0

        # sampling step: ناخذ تقريباً FPS_SAMPLE في الثانية
        step = max(int(round(fps / float(FPS_SAMPLE))), 1)

        frames_rgb = []
        poses = []

        frame_i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_i % step == 0:
                frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                pose = self.pose_extractor.extract(frame_resized)

                frames_rgb.append(frame_resized)  # BGR (بنحوّله لاحقاً)
                poses.append(pose)

            frame_i += 1

        cap.release()

        # لو الفيديو فاضي
        if len(frames_rgb) == 0:
            frames_rgb = [np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)]
            poses = [np.zeros((132,), dtype=np.float32)]

        return frames_rgb, poses

    def _sample_or_pad(self, frames_rgb, poses):
        # إذا أكثر من SEQ_LEN: نأخذ مقطع
        if len(frames_rgb) >= SEQ_LEN:
            start = 0
            # augment: خذ بداية عشوائية
            if self.augment and (len(frames_rgb) > SEQ_LEN):
                start = np.random.randint(0, len(frames_rgb) - SEQ_LEN + 1)
            frames_rgb = frames_rgb[start:start + SEQ_LEN]
            poses = poses[start:start + SEQ_LEN]
        else:
            # pad بالتكرار آخر فريم
            pad_n = SEQ_LEN - len(frames_rgb)
            last_f = frames_rgb[-1]
            last_p = poses[-1]
            frames_rgb = frames_rgb + [last_f] * pad_n
            poses = poses + [last_p] * pad_n

        return frames_rgb, poses

    def __getitem__(self, idx):
        fname, y = self.items[idx]
        path = self.videos_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing video: {path}")

        frames_bgr, poses = self._read_video_frames(path)
        frames_bgr, poses = self._sample_or_pad(frames_bgr, poses)

        # BGR -> RGB
        frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]

        frames = np.stack(frames_rgb).astype(np.float32) / 255.0  # (T,H,W,3)
        frames = np.transpose(frames, (0, 3, 1, 2))               # (T,3,H,W)

        pose = np.stack(poses).astype(np.float32)                 # (T,132)
        y = y.astype(np.float32)                                  # (C,)

        return torch.from_numpy(frames), torch.from_numpy(pose), torch.from_numpy(y)
