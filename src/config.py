from pathlib import Path

# =============================
# Paths
# =============================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEOS_DIR = PROJECT_ROOT / "data" / "videos" / "squats_all"
LABELS_CSV = PROJECT_ROOT / "data" / "labels" / "squats_labels.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

# =============================
# Labels
# =============================
CLASSES = ["correct", "fast", "uncomplete", "wrong position"]

# =============================
# Video / Dataset
# =============================
IMG_SIZE = 224
SEQ_LEN = 40          # عدد الفريمات لكل عينة
FPS_SAMPLE = 15       # نسحب فريمات أقل لتسريع التدريب
SEED = 42

# =============================
# Training
# =============================
BATCH_SIZE = 8
EPOCHS = 71
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0

# Split
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# EarlyStopping / Scheduler
PATIENCE = 7          # كم epoch نستنى بدون تحسن
MIN_DELTA = 1e-4      # أقل تحسن نعتبره تحسن
LR_PATIENCE = 3       # patience للسكيجولر
LR_FACTOR = 0.5       # يخفّض LR لنصه

# =============================
# Real-time
# =============================
RT_WINDOW = SEQ_LEN
RT_STRIDE = 4
CONF_THRESH = 0.55
