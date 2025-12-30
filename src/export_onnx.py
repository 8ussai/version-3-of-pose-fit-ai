from pathlib import Path
import torch

from src.config import CKPT_DIR, CLASSES, IMG_SIZE, SEQ_LEN
from src.model_twostream import TwoStreamFusionNet

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = CKPT_DIR / "squats_twostream_gru.pt"
    if not ckpt_path.exists():
        raise RuntimeError(f"Checkpoint not found: {ckpt_path}. Train first (python -m src.train_multilabel).")

    ckpt = torch.load(ckpt_path, map_location=device)

    model = TwoStreamFusionNet(num_classes=len(CLASSES), pretrained_backbone=False).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    # Dummy inputs (B,T,3,H,W) and (B,T,132)
    B = 1
    frames = torch.randn(B, SEQ_LEN, 3, IMG_SIZE, IMG_SIZE, device=device)
    pose = torch.randn(B, SEQ_LEN, 132, device=device)

    out_path = CKPT_DIR / "twostream_multilabel.onnx"

    dynamic_axes = {
        "frames": {0: "batch", 1: "time"},
        "pose": {0: "batch", 1: "time"},
        "logits": {0: "batch"},
    }

    torch.onnx.export(
        model,
        (frames, pose),
        str(out_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["frames", "pose"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
    )

    print(f"✅ Exported ONNX -> {out_path}")

if __name__ == "__main__":
    main()
