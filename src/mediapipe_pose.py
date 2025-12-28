import numpy as np
import mediapipe as mp

class MediaPipePoseExtractor:
    """
    يطلع Pose landmarks كـ features رقمية:
    لكل فريم: (33 نقطة * 4 قيم x,y,z,visibility) = 132
    """
    def __init__(self, static_image_mode=False, model_complexity=1, min_det=0.5, min_track=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_det,
            min_tracking_confidence=min_track
        )

    def extract(self, rgb_frame):
        # rgb_frame: HxWx3 uint8
        res = self.pose.process(rgb_frame)
        if not res.pose_landmarks:
            return np.zeros((33, 4), dtype=np.float32)
        lm = res.pose_landmarks.landmark
        out = np.array([[p.x, p.y, p.z, p.visibility] for p in lm], dtype=np.float32)
        return out  # (33,4)

    @staticmethod
    def flatten(lm_33x4: np.ndarray) -> np.ndarray:
        return lm_33x4.reshape(-1).astype(np.float32)  # (132,)
