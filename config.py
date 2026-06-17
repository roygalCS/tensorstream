"""
config.py - Configuration for Video Pipeline

All parameters in one place. Tune these to optimize for your hardware.
"""

from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class PipelineConfig:
    """Configuration for the video analytics pipeline."""
    
    # ============ VIDEO PARAMETERS ============
    video_path: str = "sample_video.mp4"  # Path to input video
    output_dir: str = "results/"  # Where to save results
    
    # ============ MODEL PARAMETERS ============
    model_name: str = "resnet18"  # Which model to use
    input_size: int = 224  # Input image size (224x224)
    num_classes: int = 1000  # ResNet18 pretrained on ImageNet (1000 classes)
    
    # ============ PROCESSING PARAMETERS ============
    batch_size: int = 4  # How many frames to process together
    # Larger batch = faster but more memory
    # Smaller batch = slower but less memory
    
    # ============ QUEUE SIZES ============
    ingestion_queue_size: int = 100  # Max frames waiting to be preprocessed
    # Larger = more buffer, more memory
    batching_queue_size: int = 20  # Max batches waiting for inference
    # Larger = more buffer, more memory
    
    # ============ DEVICE ============
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # "cuda" = GPU (fast)
    # "cpu" = CPU only (slow, but works)
    
    # ============ PERFORMANCE TARGETS ============
    target_fps: int = 30  # What FPS we aim for
    
    # ============ DISPLAY OPTIONS ============
    verbose: bool = True  # Print metrics every N frames
    verbose_interval: int = 30  # Print every 30 frames
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate batch size
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        
        # Validate queue sizes
        if self.ingestion_queue_size < self.batch_size * 2:
            print(f"Warning: ingestion_queue_size ({self.ingestion_queue_size}) "
                  f"is smaller than 2x batch_size ({self.batch_size * 2})")
        
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = "cpu"


# Default configuration - can be overridden
DEFAULT_CONFIG = PipelineConfig()