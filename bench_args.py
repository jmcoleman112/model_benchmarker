import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark YOLO detect/pose models on Jetson (inference time + system stats)."
        )
    )
    parser.add_argument("--model", required=True, help="Path to .pt or .engine model.")

    parser.add_argument("--input", required=True, help="Directory with input images for benchmarking.")

    parser.add_argument("--task", choices=["detect", "pose"], default=None, help="Override model task (detect or pose).")

    parser.add_argument("--duration", type=float, default=300.0, help="Benchmark duration in seconds (0 disables duration limit).")

    parser.add_argument("--iters", type=int, default=0, help="Max inference iterations (0 disables iteration limit).")

    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations before timing.")

    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (square).")

    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detect/pose.")

    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for detect/pose.")

    parser.add_argument("--device", default="0", help="Device id (e.g. 0, cpu).")

    parser.add_argument("--stats-interval", type=int, default=1000, help="Stats sampling interval in milliseconds.")

    parser.add_argument("--progress-interval", type=float, default=30.0, help="Progress logging interval in seconds (0 disables).")

    parser.add_argument("--output-dir", default="bench_results", help="Directory to write result JSON files.")

    parser.add_argument("--sync", action="store_true", help="Synchronize CUDA before/after timing for more accurate inference time.")

    return parser.parse_args()
