"""
process_ingestion.py - Stage 1: Video Frame Ingestion

This process reads frames from video file as fast as possible
and puts them into a queue for preprocessing.

Runs on: CPU (one core)
Job: Read raw video frames
Output: Queue of raw frames
"""

import cv2
import numpy as np
from multiprocessing import Queue, Event
from typing import Optional
import time


def ingestion_process(
    video_path: str,
    output_queue: Queue,
    stop_event: Event,
    verbose: bool = True
) -> None:
    """
    Read video frames and put them into output queue.
    
    Args:
        video_path: Path to video file
        output_queue: Queue to put frames into
        stop_event: Event to signal stop
        verbose: Print status messages
    """
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"✗ Failed to open video: {video_path}")
        stop_event.set()
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if verbose:
        print(f"✓ Ingestion: Video loaded")
        print(f"  Total frames: {total_frames}")
        print(f"  FPS: {fps:.1f}")
        print(f"  Resolution: {width}x{height}")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while not stop_event.is_set():
            # Read frame from video
            ret, frame = cap.read()
            
            if not ret:
                # Video ended
                if verbose:
                    print(f"✓ Ingestion: Video ended after {frame_count} frames")
                break
            
            # Frame is now a numpy array of shape (height, width, 3) in BGR format
            # Put it in queue for preprocessing
            try:
                # Put frame with timeout to avoid blocking forever
                output_queue.put(frame, timeout=5)
                frame_count += 1
                
                # Print progress every 100 frames
                if verbose and frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    read_fps = frame_count / elapsed
                    print(f"  Ingestion: {frame_count}/{total_frames} frames "
                          f"({read_fps:.1f} FPS)")
                
            except Exception as e:
                # Queue is full, frame is dropped
                if verbose:
                    print(f"  ⚠ Ingestion queue full, dropping frame {frame_count}")
    
    except Exception as e:
        print(f"✗ Ingestion error: {e}")
    
    finally:
        # Clean up
        cap.release()
        
        # Signal completion by putting sentinel value
        output_queue.put(None)  # None signals "no more frames"
        stop_event.set()
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"✓ Ingestion complete: {frame_count} frames in {elapsed:.2f}s "
                  f"({frame_count/elapsed:.1f} FPS)")