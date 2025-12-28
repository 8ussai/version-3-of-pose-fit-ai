from pathlib import Path
import csv
import shutil

# ================== PATHS ==================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_ROOT = PROJECT_ROOT / "data" / "squats"   # فيها correct/fast/...
OUT_VIDEOS_DIR = PROJECT_ROOT / "data" / "videos" / "squats_all"
OUT_LABELS_DIR = PROJECT_ROOT / "data" / "labels"
OUT_CSV = OUT_LABELS_DIR / "squats_labels.csv"

# ================== CLASSES ==================
# لازم تطابق أسماء الفولدرات عندك حرفيًا
CLASSES = ["correct", "fast", "uncomplete", "wrong position"]

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

def norm_key(cls_name: str) -> str:
    # للـ CSV: wrong position -> wrong_position
    return cls_name.strip().replace(" ", "_")

def main():
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"Source root not found: {SRC_ROOT}")

    # جهّز المجلدات
    OUT_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # filename -> set(labels) + one source path to copy from
    labels_map = {}   # { "v001.mp4": {"fast","wrong position"} }
    src_path_map = {} # { "v001.mp4": Path(...) }

    # 1) اجمع الليبلات حسب اسم الملف
    for cls in CLASSES:
        cls_dir = SRC_ROOT / cls
        if not cls_dir.exists():
            print(f"[WARN] Missing class folder: {cls_dir}")
            continue

        for p in cls_dir.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in VIDEO_EXTS:
                continue

            fname = p.name  # نفس الاسم يعني نفس الفيديو
            labels_map.setdefault(fname, set()).add(cls)
            # احتفظ بمسار واحد للنسخ
            src_path_map.setdefault(fname, p)

    if not labels_map:
        raise RuntimeError(f"No videos found under: {SRC_ROOT}")

    # 2) انسخ كل فيديو مرة وحدة إلى squats_all
    copied = 0
    for fname, srcp in src_path_map.items():
        dstp = OUT_VIDEOS_DIR / fname
        if dstp.exists():
            # إذا موجود، ما نعيد النسخ
            continue
        shutil.copy2(srcp, dstp)
        copied += 1

    print(f"✅ Copied {copied} unique videos -> {OUT_VIDEOS_DIR}")

    # 3) اكتب CSV multi-label
    header = ["filename"] + [norm_key(c) for c in CLASSES]
    rows = []

    for fname in sorted(labels_map.keys()):
        labs = labels_map[fname]
        row = {"filename": fname}
        for c in CLASSES:
            row[norm_key(c)] = 1 if c in labs else 0
        rows.append(row)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

    print(f"✅ Wrote labels CSV -> {OUT_CSV}")
    print(f"✅ Total unique videos: {len(labels_map)}")

    # عرض سريع لأول 10
    print("\nSample rows:")
    for r in rows[:10]:
        print(r)

if __name__ == "__main__":
    main()
