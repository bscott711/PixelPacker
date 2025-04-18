# tiff_preprocessor/data_models.py

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

# --- Data classes for configuration and layout ---

@dataclass
class VolumeLayout:
    """Dimensions and tiling layout for the volumes."""
    width: int
    height: int
    depth: int
    cols: int
    rows: int
    tile_width: int
    tile_height: int

# --- Other shared types ---

# Define a type for the channel entry dictionary (previously in core.py)
ChannelEntry = TypedDict("ChannelEntry", {"channel": int, "path": Path})

