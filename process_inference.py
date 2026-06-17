"""
process_inference.py - Stage 3: Model Inference and Results

This process takes batched tensors and runs them through the model.
Tracks performance metrics (FPS, latency, etc.)

Runs on: GPU (or CPU if no GPU available)
Input: Queue of batched tensors
Job: Run model inference, track metrics
Output: Results log
"""

import torch
from multiprocessing import Queue, Event
from typing import Dict, List
import time
from models import create_model


class MetricsTracker:
    """
    Tracks performance metrics during inference.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker.
        
        Args:
            window_size: How many frames to average over
        """
        self.window_size = window_size
        self.frame_times = []  # Latency per frame
        self.batch_times = []  # Latency per batch
        self.start_time = time.time()
    
    def add_batch(self, latency_ms: float, batch_size: int) -> None:
        """
        Record a batch inference.
        
        Args:
            latency_ms: Time taken for this batch (milliseconds)
            batch_size: Number of frames in this batch
        """
        self.batch_times.append(latency_ms)
        # Add per-frame latency (divide batch latency by frames)
        for _ in range(batch_size):
            self.frame_times.append(latency_ms / batch_size)
    
    def get_fps(self) -> float:
        """Get frames per second."""
        if not self.frame_times:
            return 0.0
        elapsed = time.time() - self.start_time
        return len(self.frame_times) / elapsed
    
    def get_avg_latency_ms(self) -> float:
        """Get average latency in milliseconds."""
        if not self.frame_times:
            return 0.0
        # Average over last window_size frames
        recent = self.frame_times[-self.window_size:]
        return sum(recent) / len(recent)
    
    def get_avg_batch_time_ms(self) -> float:
        """Get average batch processing time."""
        if not self.batch_times:
            return 0.0
        recent = self.batch_times[-10:]  # Last 10 batches
        return sum(recent) / len(recent)


def inference_process(
    input_queue: Queue,  # From batching
    stop_event: Event,
    config,  # PipelineConfig object
    verbose: bool = True
) -> None:
    """
    Run inference on batches.
    
    Args:
        input_queue: Queue of batched tensors
        stop_event: Event to signal stop
        config: Pipeline configuration
        verbose: Print status messages
    """
    
    # Load model
    model = create_model(config)
    metrics = MetricsTracker()
    
    if verbose:
        print(f"✓ Inference: Ready (device={config.device})")
    
    batch_count = 0
    frame_count = 0
    results = []
    
    try:
        while True:
            # Get batch from batching queue
            batch = input_queue.get()
            
            if batch is None:
                # Sentinel value - we're done
                if verbose:
                    print(f"✓ Inference: Received end signal")
                break
            
            batch_size = batch.shape[0]
            
            # Move batch to device (GPU or CPU)
            batch = batch.to(config.device)
            
            # Run inference and measure time
            batch_start = time.time()
            predictions, confidences = model.predict(batch)
            batch_time = (time.time() - batch_start) * 1000  # Convert to ms
            
            # Move results back to CPU for logging
            predictions = predictions.cpu().numpy()
            confidences = confidences.cpu().numpy()
            
            # Track metrics
            metrics.add_batch(batch_time, batch_size)
            frame_count += batch_size
            batch_count += 1
            
            # Log results
            for frame_idx, (pred, conf) in enumerate(zip(predictions, confidences)):
                results.append({
                    'frame_id': frame_count - batch_size + frame_idx,
                    'prediction': int(pred),
                    'confidence': float(conf),
                    'batch_id': batch_count
                })
            
            # Print metrics periodically
            if verbose and batch_count % 10 == 0:
                fps = metrics.get_fps()
                latency = metrics.get_avg_latency_ms()
                batch_time = metrics.get_avg_batch_time_ms()
                queue_size = input_queue.qsize()
                
                print(f"  Inference: Batch {batch_count:4d} | "
                      f"Frames: {frame_count:5d} | "
                      f"FPS: {fps:6.1f} | "
                      f"Latency: {latency:5.1f}ms | "
                      f"Batch time: {batch_time:5.1f}ms | "
                      f"Queue: {queue_size:3d}")
    
    except Exception as e:
        print(f"✗ Inference error: {e}")
    
    finally:
        # Final metrics
        if verbose:
            elapsed = time.time() - metrics.start_time
            fps = frame_count / elapsed
            avg_latency = metrics.get_avg_latency_ms()
            
            print(f"✓ Inference complete:")
            print(f"  Total frames: {frame_count}")
            print(f"  Total batches: {batch_count}")
            print(f"  Elapsed time: {elapsed:.2f}s")
            print(f"  Average FPS: {fps:.1f}")
            print(f"  Average latency: {avg_latency:.1f}ms")
        
        stop_event.set()
        
        # Save results
        save_results(results, config.output_dir, verbose=verbose)


def save_results(results: List[Dict], output_dir: str, verbose: bool = True) -> None:
    """
    Save inference results to file.
    
    Args:
        results: List of prediction results
        output_dir: Where to save results
        verbose: Print status
    """
    import json
    
    output_file = f"{output_dir}/predictions.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print(f"✓ Results saved to {output_file}")