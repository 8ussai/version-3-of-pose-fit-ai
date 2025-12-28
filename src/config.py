from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "videos" / "squats"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

CLASSES = ["correct", "fast", "uncomplete", "wrong position"]

# Training
IMG_SIZE = 224
SEQ_LEN = 40          # نافذة زمنية (0.5-1 ثانية حسب fps)
FPS_SAMPLE = 15       # نأخذ فريمات أقل لتسريع التدريب والريل تايم
BATCH_SIZE = 8
EPOCHS = 25
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

# Split
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42

# Real-time
RT_WINDOW = SEQ_LEN
RT_STRIDE = 4         # كل كم فريم نطلع prediction
CONF_THRESH = 0.55
