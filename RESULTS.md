# Benchmark Results

**Setup:** ResNet-18, 224x224 input, Tesla T4 (Colab). Sustained FPS and mean GPU
utilization are measured over a steady-state window (pipeline-fill startup and
end-of-stream drain excluded). See `benchmark.py` / `benchmark_colab.ipynb`.

| Batch size | Mode | Sustained FPS | Mean GPU util |
|---|---|---|---|
| 4 | Pipelined (3 processes) | 97.0 | 21.7% |
| 4 | Baseline (single-threaded) | 172.7 | 36.1% |
| 32 | Pipelined (3 processes) | 111.4 | 13.0% |
| 32 | Baseline (single-threaded) | 201.3 | 22.7% |

## What was tested

The pipeline splits work across three OS processes — ingestion → batching →
inference — connected by bounded `multiprocessing.Queue`s. The design intent is
to overlap CPU-side work (video decode, resize, normalize, batch assembly) with
GPU-side inference, so the GPU isn't sitting idle while the CPU prepares the next
batch. The naive baseline does the same steps in a single process, one after
another, with no queues and no multiprocessing.

## What the numbers show

The single-process baseline was faster at both batch sizes — by a wide margin
(roughly 1.8x). Going from batch size 4 to 32 *widened* the gap rather than
closing it. That rules out small-batch IPC overhead as a tuning problem you could
fix by batching more aggressively: if per-batch queue/IPC cost were the main
drag, larger batches (fewer batches, fewer queue operations per frame) would have
helped the pipeline catch up. It didn't.

The likely explanation is that ResNet-18 at 224x224 on a T4 is just fast. The GPU
compute time per batch is small compared to the pipeline's constant per-batch
overhead — queue puts/gets, serializing tensors across process boundaries,
process scheduling and wakeups. When GPU compute per batch is that cheap, there
isn't enough GPU idle time for overlapping to reclaim, and the coordination cost
of three processes exceeds whatever it buys back. Lower GPU utilization for the
pipelined runs (and the further drop at batch 32) is consistent with the GPU
spending proportionally more time waiting on cross-process handoffs.

## Where the pipeline would be expected to win

The overlap strategy should pay off when GPU compute time per batch is large
relative to that IPC overhead — a heavier or slower model (larger backbone,
higher input resolution, or a CPU-bound preprocessing step that genuinely
bottlenecks a single thread). That regime wasn't tested here.
