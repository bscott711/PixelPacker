# src/pixelpacker/data_models.py

from dataclasses import dataclass
from pathlib import Path

# Make sure TypedDict is imported if not already via other means
from typing import (
    TypedDict,
    Literal,
)  # Add List, Optional, Any if moving type aliases here

# --- Data classes for configuration and layout ---


@dataclass
class VolumeLayout:
    """Dimensions and tiling layout for the volumes."""

    width: int
    height: int
    depth: int  # Depth after global Z-crop (as noted in previous refactor)
    cols: int
    rows: int
    tile_width: int
    tile_height: int


# Define a type for the channel entry dictionary
class ChannelEntry(TypedDict):
    """Represents a single channel file at a specific timepoint."""

    channel: int
    path: Path


# --- Moved Dataclasses (Added from core.py) ---


@dataclass
class PreprocessingConfig:
    """Configuration settings for the preprocessing pipeline."""

    input_folder: Path
    output_folder: Path
    stretch_mode: str
    z_crop_method: str
    z_crop_threshold: int
    dry_run: bool
    debug: bool
    max_threads: int
    use_global_contrast: bool = True
    executor_type: Literal["thread", "process"] = "process"
    input_pattern: str = "*_ch*_stack*.tif*"


@dataclass
class ProcessingTask:
    """Represents a single file (channel at a timepoint) to be processed."""

    time_id: str
    channel_entry: ChannelEntry  # Contains path and channel ID
    config: PreprocessingConfig


@dataclass
class ProcessingResult:
    """Result metadata for a single successfully processed file (output from Pass 2)."""

    time_id: str
    channel: int
    filename: str  # Output filename
    p_low: float  # Applied low percentile/value
    p_high: float  # Applied high percentile/value
