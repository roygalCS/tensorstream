"""
main.py - Main Pipeline Orchestrator

Starts all three processes (ingestion, batching, inference) and manages them.
This is what you run to execute the entire pipeline.

Usage:
    python main.py
    python main.py --video my_video.mp4
    python main.py --batch-size 8 --device cpu
"""

import argparse
from multiprocessing import Process, Queue, Event
import time
import sys
from pathlib import Path

from config import PipelineConfig
from process_ingestion import ingestion_process
from process_batching import batching_process
from process_inference import inference_process


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Edge-Optimized Video Analytics Pipeline"
    )
    
    parser.add_argument(
        "--video",
        type=str,
        default="sample_video.mp4",
        help="Path to input video file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference"
    )
    
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Input image size (224x224, 160x160, etc.)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        help="Model name (resnet18, resnet50, mobilenet_v2)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda, cpu)"
    )
    
    parser.add_argument(
        "--ingestion-queue-size",
        type=int,
        default=100,
        help="Max frames in ingestion queue"
    )
    
    parser.add_argument(
        "--batching-queue-size",
        type=int,
        default=20,
        help="Max batches in batching queue"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed status messages"
    )
    
    return parser


def main():
    """
    Run the complete video analytics pipeline.
    
    This orchestrates three parallel processes:
    1. Ingestion: Read frames from video
    2. Batching: Preprocess and batch frames
    3. Inference: Run model on batches
    """
    
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Create configuration from arguments
    config = PipelineConfig(
        video_path=args.video,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        input_size=args.input_size,
        model_name=args.model,
        device=args.device,
        ingestion_queue_size=args.ingestion_queue_size,
        batching_queue_size=args.batching_queue_size,
        verbose=args.verbose
    )
    
    # Validate video exists
    if not Path(config.video_path).exists():
        print(f"✗ Video not found: {config.video_path}")
        print(f"  Please provide a valid video file with --video argument")
        print(f"  Example: python main.py --video path/to/video.mp4")
        sys.exit(1)
    
    # Print configuration
    print("\n" + "="*60)
    print("VIDEO ANALYTICS PIPELINE - CONFIGURATION")
    print("="*60)
    print(f"Video:              {config.video_path}")
    print(f"Output:             {config.output_dir}")
    print(f"Model:              {config.model_name}")
    print(f"Input size:         {config.input_size}x{config.input_size}")
    print(f"Batch size:         {config.batch_size}")
    print(f"Device:             {config.device}")
    print(f"Ingestion queue:    {config.ingestion_queue_size} frames")
    print(f"Batching queue:     {config.batching_queue_size} batches")
    print("="*60 + "\n")
    
    # Create queues for inter-process communication
    # Queue 1: Raw frames from ingestion to batching
    raw_frame_queue = Queue(maxsize=config.ingestion_queue_size)
    
    # Queue 2: Batched tensors from batching to inference
    batched_tensor_queue = Queue(maxsize=config.batching_queue_size)
    
    # Event to signal all processes to stop
    stop_event = Event()
    
    # Create the three processes
    ingestion_proc = Process(
        target=ingestion_process,
        args=(
            config.video_path,
            raw_frame_queue,
            stop_event,
            config.verbose
        ),
        name="Ingestion"
    )
    
    batching_proc = Process(
        target=batching_process,
        args=(
            raw_frame_queue,
            batched_tensor_queue,
            stop_event,
            config.batch_size,
            config.input_size,
            config.verbose
        ),
        name="Batching"
    )
    
    inference_proc = Process(
        target=inference_process,
        args=(
            batched_tensor_queue,
            stop_event,
            config,
            config.verbose
        ),
        name="Inference"
    )
    
    # Start all processes
    print("Starting pipeline processes...\n")
    start_time = time.time()
    
    ingestion_proc.start()
    batching_proc.start()
    inference_proc.start()
    
    try:
        # Wait for all processes to finish
        ingestion_proc.join()
        batching_proc.join()
        inference_proc.join()
    
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\n\nInterrupt received, shutting down...")
        stop_event.set()
        ingestion_proc.terminate()
        batching_proc.terminate()
        inference_proc.terminate()
        ingestion_proc.join(timeout=5)
        batching_proc.join(timeout=5)
        inference_proc.join(timeout=5)
    
    # Print final summary
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Total time: {elapsed:.2f}s")
    print(f"Results saved to: {config.output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # This is required for multiprocessing on Windows
    # On Linux/Mac, not strictly necessary but good practice
    main()