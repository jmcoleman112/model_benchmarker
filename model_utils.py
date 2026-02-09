import sys
from typing import Optional


def load_model(model_path: str, task: Optional[str]):
    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "error: ultralytics not installed. Run: pip install ultralytics",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if task:
        return YOLO(model_path, task=task)
    return YOLO(model_path)


def maybe_import_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def predict_once(model, img, args):
    kwargs = {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "device": args.device,
        "verbose": False,
    }
    if args.task:
        kwargs["task"] = args.task
    model.predict(img, **kwargs)
