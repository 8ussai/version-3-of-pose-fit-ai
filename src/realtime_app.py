import cv2
import numpy as np
import torch
from collections import deque

from .config import IMG_SIZE, CLASSES, RT_WINDOW, RT_STRIDE, CKPT_DIR
from .mediapipe_pose import MediaPipePoseExtractor
from .model_twostream import TwoStreamFusionNet

def preprocess_frame(frame_bgr):
    frame = cv2.resize(frame_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0)
    x = np.transpose(x, (2, 0, 1))
    return rgb, x

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = CKPT_DIR / "best_multilabel.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    thr = float(ckpt.get("threshold", 0.5))

    model = TwoStreamFusionNet(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    pose_ex = MediaPipePoseExtractor(static_image_mode=False)

    frames_q = deque(maxlen=RT_WINDOW)
    pose_q = deque(maxlen=RT_WINDOW)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    tick = 0
    last_text = "-"

    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb, x = preprocess_frame(frame)
            lm = pose_ex.extract(rgb)
            pvec = pose_ex.flatten(lm)

            frames_q.append(x)
            pose_q.append(pvec)

            if len(frames_q) == RT_WINDOW and (tick % RT_STRIDE == 0):
                frames = np.stack(frames_q).astype(np.float32)
                pose = np.stack(pose_q).astype(np.float32)

                frames_t = torch.from_numpy(frames)[None, ...].to(device)
                pose_t = torch.from_numpy(pose)[None, ...].to(device)

                logits = model(frames_t, pose_t)[0]
                probs = torch.sigmoid(logits).detach().cpu().numpy()

                active = [f"{CLASSES[i]}:{probs[i]:.2f}" for i in range(len(CLASSES)) if probs[i] >= thr]
                last_text = " | ".join(active) if len(active) else "-"

            cv2.putText(frame, f"Labels: {last_text}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Real-Time Multi-Label Squat", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

            tick += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
