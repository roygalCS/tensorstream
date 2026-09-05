"""
benchmark.py - Throughput + GPU-utilization benchmark for TensorStream.

For each mode it reports, over a *steady-state window only* (pipeline-fill
startup and end-of-stream drain excluded):

    sustained FPS   = frames processed / wall-clock seconds
    mean GPU util % = average of nvidia-smi `utilization.gpu` samples taken
                      during that same window

Modes:
    pipelined  - the real 3-process pipeline (ingestion | batching | inference)
    baseline   - a naive single-threaded version: one loop that reads a frame,
                 preprocesses it, and runs inference, with no multiprocessing
    both       - run baseline then pipelined and print a comparison (default)

Meant to run on a CUDA machine (e.g. a Colab GPU runtime). On CPU it still
reports FPS; the GPU-util columns show "n/a" when nvidia-smi is absent.

Usage:
    python benchmark.py --video sample_video.mp4
    python benchmark.py --video sample_video.mp4 --mode pipelined --loops 5
    python benchmark.py --model resnet50 --batch-size 8 --input-size 224
"""

import argparse
import math
import multiprocessing as mp
import shutil
import subprocess
import threading
import time
from pathlib import Path

# 'spawn' (not 'fork'): the inference stage touches CUDA, and CUDA context +
# fork() in the parent is unsafe. spawn re-imports this module in each child,
# so every process target below is module-level and every arg is picklable.
MP = mp.get_context("spawn")

import cv2
import torch

from config import PipelineConfig
from models import create_model
from process_batching import FramePreprocessor, batching_process


# --------------------------------------------------------------------------- #
# GPU utilization sampler
# --------------------------------------------------------------------------- #
class GpuSampler(threading.Thread):
    """Polls `nvidia-smi` for GPU utilization on a background thread.

    Each sample is (wall_clock_time, util_percent) for GPU 0. `mean_between`
    then averages only the samples inside the steady-state window so the util
    number lines up with the FPS number.

    (`nvidia-smi dmon -s u` gives the same signal; a one-shot query per tick is
    just easier to timestamp and window precisely.)
    """

    def __init__(self, period_s: float = 0.1):
        super().__init__(daemon=True)
        self.period_s = period_s
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self.available = shutil.which("nvidia-smi") is not None

    def run(self) -> None:
        if not self.available:
            return
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    text=True, timeout=2,
                )
                self.samples.append((time.time(), float(out.strip().splitlines()[0])))
            except Exception:
                pass
            self._stop.wait(self.period_s)

    def stop(self) -> None:
        self._stop.set()

    def mean_between(self, t0: float, t1: float) -> float:
        vals = [v for (t, v) in self.samples if t0 <= t <= t1]
        return sum(vals) / len(vals) if vals else math.nan

    def count_between(self, t0: float, t1: float) -> int:
        return sum(1 for (t, _) in self.samples if t0 <= t <= t1)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _warmup_model(mm, config) -> None:
    """A few forward passes so cuDNN autotune / allocation is not timed."""
    dummy = torch.zeros(
        config.batch_size, 3, config.input_size, config.input_size,
        device=config.device,
    )
    with torch.no_grad():
        for _ in range(10):
            mm.model(dummy)
    if config.device == "cuda":
        torch.cuda.synchronize()


def _summarize(records, sampler, warmup_s, cooldown_s):
    """records: list of (t_batch_done, cumulative_frame_count), in order."""
    if len(records) < 3:
        raise RuntimeError(
            "too few batches to measure - use a longer clip or more --loops"
        )
    lo = records[0][0] + warmup_s
    hi = records[-1][0] - cooldown_s
    win = [(t, n) for (t, n) in records if lo <= t <= hi]
    if len(win) < 2:
        raise RuntimeError(
            "steady-state window is empty - lower --warmup/--cooldown or "
            "use a longer clip / more --loops"
        )
    frames = win[-1][1] - win[0][1]
    dur = win[-1][0] - win[0][0]
    return {
        "fps": frames / dur,
        "window_frames": frames,
        "window_seconds": dur,
        "gpu_util": sampler.mean_between(win[0][0], win[-1][0]),
        "gpu_samples": sampler.count_between(win[0][0], win[-1][0]),
        "gpu_available": sampler.available,
        "batches_total": len(records),
        "batches_in_window": len(win),
    }


# --------------------------------------------------------------------------- #
# Mode 1: real 3-process pipeline
# --------------------------------------------------------------------------- #
def _looping_ingestion(video_path, out_queue, stop_event, loops):
    """Like process_ingestion, but replays the file `loops` times before the
    sentinel, so a short clip still produces a long sustained run."""
    for _ in range(loops):
        cap = cv2.VideoCapture(video_path)
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            out_queue.put(frame)
        cap.release()
    out_queue.put(None)


def _bench_inference(in_queue, stop_event, config, result_queue):
    """Instrumented inference stage: same forward path as process_inference,
    plus a (timestamp, cumulative_frames) record after every batch."""
    mm = create_model(config)
    _warmup_model(mm, config)

    records = []
    n_frames = 0
    try:
        while True:
            batch = in_queue.get()
            if batch is None:
                break
            batch = batch.to(config.device, non_blocking=True)
            pred, conf = mm.predict(batch)
            pred.cpu()                       # forces a CUDA sync, mirrors real pipeline
            n_frames += batch.shape[0]
            records.append((time.time(), n_frames))
    finally:
        stop_event.set()
        result_queue.put(records)


def run_pipelined(config, loops, warmup_s, cooldown_s):
    raw_q = MP.Queue(maxsize=config.ingestion_queue_size)
    bat_q = MP.Queue(maxsize=config.batching_queue_size)
    res_q = MP.Queue()
    stop = MP.Event()

    procs = [
        MP.Process(target=_looping_ingestion,
                   args=(config.video_path, raw_q, stop, loops), name="ingestion"),
        MP.Process(target=batching_process,
                   args=(raw_q, bat_q, stop, config.batch_size, config.input_size, False),
                   name="batching"),
        MP.Process(target=_bench_inference,
                   args=(bat_q, stop, config, res_q), name="inference"),
    ]

    sampler = GpuSampler()
    sampler.start()
    for p in procs:
        p.start()
    records = res_q.get()            # blocks until inference stage finishes
    for p in procs:
        p.join()
    sampler.stop()
    sampler.join()
    return _summarize(records, sampler, warmup_s, cooldown_s)


# --------------------------------------------------------------------------- #
# Mode 2: naive single-threaded baseline (no multiprocessing)
# --------------------------------------------------------------------------- #
def run_baseline(config, loops, warmup_s, cooldown_s):
    pre = FramePreprocessor(input_size=config.input_size)
    mm = create_model(config)
    _warmup_model(mm, config)

    sampler = GpuSampler()
    sampler.start()

    records = []
    n_frames = 0
    buf = []
    for _ in range(loops):
        cap = cv2.VideoCapture(config.video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            buf.append(pre.preprocess(frame))          # CPU work, inline
            if len(buf) == config.batch_size:
                batch = torch.stack(buf).to(config.device)
                pred, conf = mm.predict(batch)
                pred.cpu()
                if config.device == "cuda":
                    torch.cuda.synchronize()
                n_frames += config.batch_size
                records.append((time.time(), n_frames))
                buf = []
        cap.release()

    sampler.stop()
    sampler.join()
    return _summarize(records, sampler, warmup_s, cooldown_s)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt_util(r) -> str:
    if not r["gpu_available"] or math.isnan(r["gpu_util"]):
        return "n/a (nvidia-smi not found)"
    return f"{r['gpu_util']:.1f} %  ({r['gpu_samples']} samples)"


def _print_block(title, r) -> None:
    print(f"\n{title}")
    print(f"  sustained FPS:   {r['fps']:8.1f}   "
          f"({r['window_frames']} frames / {r['window_seconds']:.2f} s steady-state, "
          f"{r['batches_in_window']}/{r['batches_total']} batches)")
    print(f"  mean GPU util:   {_fmt_util(r)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TensorStream throughput / GPU-util benchmark")
    parser.add_argument("--video", default="sample_video.mp4")
    parser.add_argument("--mode", choices=["pipelined", "baseline", "both"], default="both")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--loops", type=int, default=3,
                        help="replay the clip N times for a longer sustained run")
    parser.add_argument("--warmup", type=float, default=2.0,
                        help="seconds of pipeline-fill startup to exclude")
    parser.add_argument("--cooldown", type=float, default=1.0,
                        help="seconds of end-of-stream drain to exclude")
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"'{args.video}' not found - generating a synthetic clip...")
        from make_sample_video import make_sample_video
        make_sample_video(args.video)

    config = PipelineConfig(
        video_path=args.video,
        model_name=args.model,
        batch_size=args.batch_size,
        input_size=args.input_size,
        device=args.device,
        verbose=False,
    )

    src_w = src_h = None
    cap = cv2.VideoCapture(config.video_path)
    if cap.isOpened():
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    gpu_name = torch.cuda.get_device_name(0) if config.device == "cuda" else "-"

    print("=" * 64)
    print("TENSORSTREAM BENCHMARK")
    print("=" * 64)
    print(f"Model:                 {config.model_name}  (ImageNet-pretrained, eval)")
    print(f"Source resolution:     {src_w}x{src_h}")
    print(f"Inference resolution:  {config.input_size}x{config.input_size}")
    print(f"Batch size:            {config.batch_size}")
    print(f"Device:                {config.device}  ({gpu_name})")
    print(f"Clip replays (loops):  {args.loops}")
    print(f"Steady-state window:   drop first {args.warmup:.1f}s / last {args.cooldown:.1f}s")

    results = {}
    if args.mode in ("baseline", "both"):
        results["baseline"] = run_baseline(config, args.loops, args.warmup, args.cooldown)
    if args.mode in ("pipelined", "both"):
        results["pipelined"] = run_pipelined(config, args.loops, args.warmup, args.cooldown)

    print("\n" + "=" * 64)
    print("RESULTS")
    print("=" * 64)
    if "pipelined" in results:
        _print_block("Pipelined (3 processes: ingestion | batching | inference)",
                     results["pipelined"])
    if "baseline" in results:
        _print_block("Single-threaded baseline (sequential ingest -> infer)",
                     results["baseline"])

    # compact summary (item 5)
    print("\n" + "-" * 64)
    p = results.get("pipelined")
    b = results.get("baseline")

    def _u(r):
        if r is None or not r["gpu_available"] or math.isnan(r["gpu_util"]):
            return "n/a"
        return f"{r['gpu_util']:.1f}%"

    fields = [
        f"model={config.model_name}",
        f"resolution={config.input_size}x{config.input_size}",
    ]
    if p:
        fields += [f"fps_pipelined={p['fps']:.1f}", f"gpu_util_pipelined={_u(p)}"]
    if b:
        fields += [f"fps_baseline={b['fps']:.1f}", f"gpu_util_baseline={_u(b)}"]
    print("SUMMARY  " + "  ".join(fields))
    print("-" * 64)


if __name__ == "__main__":
    main()
