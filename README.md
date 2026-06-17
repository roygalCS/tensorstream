# TensorStream

**Edge-optimized video inference engine built on a multi-process producer-consumer pipeline.**

TensorStream ingests video frames, preprocesses and batches them in parallel, and runs GPU-accelerated model inference — all across three concurrent processes connected by shared queues. The pipeline is designed for throughput-sensitive workloads where frame ingestion, preprocessing, and inference need to happen simultaneously rather than sequentially.

---

## Architecture

```
Video File
    │
    ▼
┌──────────────────┐
│  Ingestion       │  Process 1 — Decodes frames from video via OpenCV
│  process_ingestion.py │
└────────┬─────────┘
         │  raw_frame_queue (Queue, max N frames)
         ▼
┌──────────────────┐
│  Batching        │  Process 2 — Resizes, normalizes, and batches frames into tensors
│  process_batching.py  │
└────────┬─────────┘
         │  batched_tensor_queue (Queue, max N batches)
         ▼
┌──────────────────┐
│  Inference       │  Process 3 — Loads model, runs forward pass, tracks FPS/latency
│  process_inference.py │
└────────┬─────────┘
         │
         ▼
   results/predictions.json
```

The three processes run concurrently and communicate exclusively through bounded queues. A shared `stop_event` handles graceful shutdown. Sentinel `None` values signal end-of-stream between stages.

---

## Features

- **True parallelism** — each pipeline stage runs in its own Python process, bypassing the GIL
- **Bounded queues** — configurable queue sizes prevent unbounded memory growth under load
- **GPU inference** — model runs on CUDA by default, falls back to CPU
- **Swappable backbones** — ResNet-18, ResNet-50, and MobileNetV2 supported out of the box
- **Live metrics** — FPS, per-frame latency, and batch time tracked and printed during inference
- **Graceful shutdown** — Ctrl+C propagates cleanly across all three processes
- **JSON output** — predictions and confidence scores saved per frame

---

## Project Structure

```
tensorstream/
├── main.py                  # Pipeline orchestrator — spawns and joins all processes
├── config.py                # PipelineConfig dataclass — all tunable parameters
├── models.py                # ModelManager — loads torchvision models, runs inference
├── process_ingestion.py     # Stage 1 — video decode and frame production
├── process_batching.py      # Stage 2 — preprocessing and batch assembly
└── process_inference.py     # Stage 3 — model inference and metrics tracking
```

---

## Setup

```bash
git clone https://github.com/roygalCS/tensorstream.git
cd tensorstream
pip install torch torchvision opencv-python
```

CUDA is required for GPU inference. To run on CPU, pass `--device cpu`.

---

## Usage

```bash
# Basic run (defaults: resnet18, batch_size=4, CUDA)
python main.py --video my_video.mp4

# Custom configuration
python main.py \
  --video my_video.mp4 \
  --model mobilenet_v2 \
  --batch-size 8 \
  --input-size 224 \
  --device cuda \
  --output-dir results/
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--video` | `sample_video.mp4` | Path to input video |
| `--model` | `resnet18` | Backbone (`resnet18`, `resnet50`, `mobilenet_v2`) |
| `--batch-size` | `4` | Frames per inference batch |
| `--input-size` | `224` | Frame resize dimension (NxN) |
| `--device` | `cuda` | Inference device (`cuda` or `cpu`) |
| `--output-dir` | `results/` | Where to write `predictions.json` |
| `--ingestion-queue-size` | `100` | Max raw frames buffered between stages 1→2 |
| `--batching-queue-size` | `20` | Max batched tensors buffered between stages 2→3 |

---

## Output

Predictions are written to `results/predictions.json`:

```json
[
  {
    "frame_id": 0,
    "prediction": 281,
    "confidence": 0.94,
    "batch_id": 1
  },
  ...
]
```

Live metrics are printed to stdout every 10 batches:

```
Inference: Batch   10 | Frames:    40 | FPS:   87.3 | Latency:  11.4ms | Batch time:  45.2ms | Queue:  12
```

---

## Design Notes

**Why three processes?** Video decode, image preprocessing, and GPU inference have very different CPU/memory profiles. Running them concurrently means the GPU stays busy while the CPU is decoding the next segment of frames. A single-threaded pipeline would leave the GPU idle during preprocessing and vice versa.

**Queue sizing** controls the tradeoff between memory usage and pipeline stall tolerance. A larger `ingestion-queue-size` gives the batching stage more slack if inference slows down; shrinking it reduces peak RAM at the cost of more frequent back-pressure stalls.

**Model loading** happens inside the inference process after it forks, so the model weights live only in that process's memory space.

---

## Related Projects

- [TitanGrad](https://github.com/roygalCS/titangrad) — dual-backend autograd engine (NumPy + CuPy + Numba) built from scratch
