import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

from bench_io import safe_name, find_input_files
from model_utils import load_model, maybe_import_torch, predict_once
from stats import select_sampler


def run_benchmark(args) -> Path:
    model_path = Path(args.model)
    if not model_path.exists():
        print("error: model not found: {0}".format(model_path), file=sys.stderr)
        raise SystemExit(1)

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print("error: input directory not found: {0}".format(input_dir), file=sys.stderr)
        raise SystemExit(1)

    input_files = find_input_files(input_dir)
    if not input_files:
        print("error: no input images found.", file=sys.stderr)
        raise SystemExit(1)

    model = load_model(str(model_path), args.task)
    model_task = args.task or getattr(model, "task", None)

    try:
        import cv2
    except ImportError:
        print(
            "error: opencv-python not installed. Run: pip install opencv-python",
            file=sys.stderr,
        )
        raise SystemExit(1)

    torch = maybe_import_torch()

    for i in range(max(args.warmup, 0)):
        img_path = input_files[i % len(input_files)]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        predict_once(model, img, args)

    sampler, stats_source = select_sampler(args.stats_interval)

    infer_count = 0
    infer_sum = 0.0
    infer_min = None
    infer_max = None
    skipped = 0
    start_time = time.perf_counter()
    start_time_utc = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    def should_stop() -> bool:
        if args.iters > 0 and infer_count >= args.iters:
            return True
        if args.duration > 0 and (time.perf_counter() - start_time) >= args.duration:
            return True
        return False

    try:
        idx = 0
        while True:
            if should_stop():
                break
            img_path = input_files[idx]
            idx = (idx + 1) % len(input_files)
            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue
            if args.sync and torch and torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            predict_once(model, img, args)
            if args.sync and torch and torch.cuda.is_available():
                torch.cuda.synchronize()
            dt_inf = time.perf_counter() - t0
            infer_sum += dt_inf
            infer_count += 1
            infer_min = dt_inf if infer_min is None else min(infer_min, dt_inf)
            infer_max = dt_inf if infer_max is None else max(infer_max, dt_inf)
    finally:
        if sampler:
            sampler.stop()

    elapsed = time.perf_counter() - start_time
    avg_ms = (infer_sum / infer_count * 1000.0) if infer_count else None
    fps = (infer_count / infer_sum) if infer_sum > 0 else None

    stats = sampler.averages() if sampler else {
        "cpu_avg_pct": None,
        "gpu_avg_pct": None,
        "ram_avg_used_mb": None,
        "ram_total_mb": None,
        "temps_avg_c": None,
        "power_avg_mw": None,
        "samples": 0,
    }

    model_name = safe_name(model_path.stem)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "{0}_benchmark.json".format(model_name)
    if output_path.exists():
        stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / "{0}_benchmark_{1}.json".format(model_name, stamp)

    results: Dict[str, Any] = {
        "model": {
            "name": model_name,
            "path": str(model_path),
            "ext": model_path.suffix.lower(),
        },
        "task": model_task,
        "input": {
            "dir": str(input_dir),
            "num_files": len(input_files),
        },
        "run": {
            "start_time_utc": start_time_utc,
            "duration_s_requested": args.duration,
            "duration_s_actual": elapsed,
            "iterations": infer_count,
            "skipped_files": skipped,
            "warmup_iters": args.warmup,
        },
        "inference": {
            "avg_ms": avg_ms,
            "min_ms": infer_min * 1000.0 if infer_min is not None else None,
            "max_ms": infer_max * 1000.0 if infer_max is not None else None,
            "fps": fps,
        },
        "system": stats,
        "notes": {
            "stats_source": stats_source,
            "stats_interval_ms": args.stats_interval,
        },
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return output_path
