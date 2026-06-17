"""
process_batching.py - Stage 2: Frame Batching and Preprocessing

This process takes raw frames from ingestion queue,
preprocesses them (resize, normalize), and batches them.

Runs on: CPU (one core)
Input: Queue of raw frames
Job: Resize, normalize, batch frames
Output: Queue of batched tensors
"""

import cv2
import numpy as np
import torch
from multiprocessing import Queue, Event
from typing import Optional
import time


class FramePreprocessor:
    """
    Preprocesses frames for model inference.
    """
    
    def __init__(self, input_size: int = 224):
        """
        Initialize preprocessor.
        
        Args:
            input_size: Target size (input_size x input_size)
        """
        self.input_size = input_size
        
        # ImageNet normalization constants
        # These are standard values used when training ResNet on ImageNet
        self.mean = np.array([0.485, 0.456, 0.406])  # R, G, B
        self.std = np.array([0.229, 0.224, 0.225])   # R, G, B
    
    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single frame.
        
        Args:
            frame: Raw frame from OpenCV (BGR, uint8, range [0, 255])
        
        Returns:
            Preprocessed frame as tensor (C, H, W)
        """
        # Step 1: Convert BGR (OpenCV) to RGB (PyTorch standard)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Step 2: Resize to target size
        frame_resized = cv2.resize(frame_rgb, (self.input_size, self.input_size))
        
        # Step 3: Convert to float and normalize to [0, 1]
        frame_float = frame_resized.astype(np.float32) / 255.0
        
        # Step 4: Apply ImageNet normalization
        # This is critical - models are trained with this normalization
        frame_normalized = (frame_float - self.mean) / self.std
        
        # Step 5: Convert to tensor and transpose to (C, H, W) format
        # PyTorch expects (Channels, Height, Width), OpenCV gives (H, W, C)
        frame_tensor = torch.from_numpy(
            frame_normalized.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
        )
        
        return frame_tensor


def batching_process(
    input_queue: Queue,  # From ingestion
    output_queue: Queue,  # To inference
    stop_event: Event,
    batch_size: int = 4,
    input_size: int = 224,
    verbose: bool = True
) -> None:
    """
    Batch and preprocess frames.
    
    Args:
        input_queue: Queue of raw frames from ingestion
        output_queue: Queue to put batched tensors
        stop_event: Event to signal stop
        batch_size: How many frames per batch
        input_size: Target image size (224x224, etc.)
        verbose: Print status messages
    """
    
    preprocessor = FramePreprocessor(input_size=input_size)
    
    if verbose:
        print(f"✓ Batching: Started (batch_size={batch_size})")
    
    batch = []  # Accumulate frames
    frame_count = 0
    batch_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Get frame from ingestion queue
            frame = input_queue.get()
            
            if frame is None:
                # Sentinel value - video is done
                if verbose:
                    print(f"✓ Batching: Received end signal")
                break
            
            # Preprocess frame
            try:
                frame_tensor = preprocessor.preprocess(frame)
                batch.append(frame_tensor)
                frame_count += 1
                
                # When batch is full, send it to inference
                if len(batch) == batch_size:
                    # Stack frames into a batch tensor
                    batch_tensor = torch.stack(batch)  # Shape: (batch_size, C, H, W)
                    
                    try:
                        output_queue.put(batch_tensor, timeout=5)
                        batch_count += 1
                        batch = []  # Reset for next batch
                        
                        if verbose and batch_count % 10 == 0:
                            elapsed = time.time() - start_time
                            batch_rate = batch_count / elapsed
                            print(f"  Batching: {batch_count} batches "
                                  f"({frame_count} frames, {batch_rate:.1f} batches/s)")
                    
                    except Exception as e:
                        if verbose:
                            print(f"  ⚠ Batching queue full, dropping batch {batch_count}")
            
            except Exception as e:
                print(f"✗ Preprocessing error on frame {frame_count}: {e}")
    
    except Exception as e:
        print(f"✗ Batching error: {e}")
    
    finally:
        # Handle leftover frames (incomplete batch at end)
        if len(batch) > 0:
            if verbose:
                print(f"  Batching: Sending incomplete final batch ({len(batch)} frames)")
            batch_tensor = torch.stack(batch)
            try:
                output_queue.put(batch_tensor, timeout=5)
                batch_count += 1
            except:
                pass
        
        # Signal completion
        output_queue.put(None)
        stop_event.set()
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"✓ Batching complete: {batch_count} batches "
                  f"({frame_count} frames) in {elapsed:.2f}s")