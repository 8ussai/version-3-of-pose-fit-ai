from pathlib import Path
import csv
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import CLASSES, IMG_SIZE, SEQ_LEN, FPS_SAMPLE, SEED
from .mediapipe_pose import MediaPipePoseExtractor

def read_labels_csv(csv_path: Path):
    """
    Returns:
      items: list of (video_path, multi_hot_vector)
    """
    items = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"].strip()
            y = []
            for c in CLASSES:
                key = c.replace(" ", "_")  # wrong position -> wrong_position
                y.append(float(row.get(key, row.get(c, 0))))
            items.append((fname, np.array(y, dtype=np.float32)))
    return items

def split_by_videos(items, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    items: list of (fname, y)
    split by unique fname (no leakage)
    """
    rng = np.random.RandomState(SEED)
    idx = np.arange(len(items))
    rng.shuffle(idx)

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]

    def pick(ii):
        return [items[i] for i in ii]

    return pick(train_idx), pick(val_idx), pick(test_idx)

class VideoPoseMultiLabelDataset(Dataset):
    """
    returns:
      frames: (T,3,H,W)
      pose:   (T,132)
      y:      (num_classes,) float {0,1}
    """
    def __init__(self, videos_dir: Path, items, training=True):
        self.videos_dir = videos_dir
        self.items = items
        self.training = training
        self.pose_extractor = MediaPipePoseExtractor(static_image_mode=False)

    def __len__(self):
        return len(self.items)

    def _read_video_frames(self, path: Path):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1:
            fps = 30.0

        step = max(1, int(round(fps / FPS_SAMPLE)))

        frames_rgb = []
        poses = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                lm = self.pose_extractor.extract(rgb)
                pvec = self.pose_extractor.flatten(lm)

                frames_rgb.append(rgb)
                poses.append(pvec)
            idx += 1

        cap.release()

        if len(frames_rgb) == 0:
            frames_rgb = [np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)]
            poses = [np.zeros((132,), dtype=np.float32)]

        return frames_rgb, poses

    def _sample_or_pad(self, frames_rgb, poses):
        T = len(frames_rgb)
        if T >= SEQ_LEN:
            if self.training:
                start = np.random.randint(0, T - SEQ_LEN + 1)
            else:
                start = (T - SEQ_LEN) // 2
            frames_rgb = frames_rgb[start:start+SEQ_LEN]
            poses = poses[start:start+SEQ_LEN]
        else:
            last_f, last_p = frames_rgb[-1], poses[-1]
            pad_n = SEQ_LEN - T
            frames_rgb = frames_rgb + [last_f]*pad_n
            poses = poses + [last_p]*pad_n
        return frames_rgb, poses

    def __getitem__(self, i):
        fname, y = self.items[i]
        path = self.videos_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing video: {path}")

        frames_rgb, poses = self._read_video_frames(path)
        frames_rgb, poses = self._sample_or_pad(frames_rgb, poses)

        frames = np.stack(frames_rgb).astype(np.float32) / 255.0
        frames = np.transpose(frames, (0, 3, 1, 2))  # (T,3,H,W)

        pose = np.stack(poses).astype(np.float32)     # (T,132)
        y = y.astype(np.float32)                      # (C,)

        return torch.from_numpy(frames), torch.from_numpy(pose), torch.from_numpy(y)
