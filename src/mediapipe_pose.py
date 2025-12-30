import numpy as np

class MediaPipePoseExtractor:
    """
    يرجّع pose features طولها 132:
    33 landmark * 4 (x,y,z,visibility)
    إذا mediapipe غير متوفر أو فشل، يرجّع zeros.
    """
    def __init__(self):
        self.ok = False
        try:
            import mediapipe as mp
            self.mp = mp
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.ok = True
        except Exception:
            self.mp = None
            self.pose = None
            self.ok = False

    def extract(self, frame_bgr) -> np.ndarray:
        if not self.ok:
            return np.zeros((132,), dtype=np.float32)

        try:
            # mediapipe بده RGB
            frame_rgb = frame_bgr[..., ::-1]
            res = self.pose.process(frame_rgb)
            if not res.pose_landmarks:
                return np.zeros((132,), dtype=np.float32)

            feats = []
            for lm in res.pose_landmarks.landmark:
                feats.extend([lm.x, lm.y, lm.z, lm.visibility])
            return np.array(feats, dtype=np.float32)
        except Exception:
            return np.zeros((132,), dtype=np.float32)
