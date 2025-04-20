# src/pixelpacker/core.py

# --- Standard Imports ---
import math
import time
from typing import Any, Dict, List, Optional, Tuple


# --- Centralized Utilities ---
# Imports log, scan_and_parse_files, TimepointsDict from utils
from .utils import log, scan_and_parse_files, TimepointsDict

# --- Refactored Modules ---
from . import crop
from . import limits
from . import manifest
from . import processing

# --- Data Models & Types ---
# Imports needed dataclasses from data_models
from .data_models import (
    PreprocessingConfig,
    ProcessingTask,
    VolumeLayout,
)

# Type alias remains useful here
TimepointResult = Dict[str, Any]

# Configuration setup functions (_load_config_from_file, _setup_configuration)
# have been moved to cli.py


# === Layout Determination ===
def _determine_layout(
    base_width: int, base_height: int, global_z_range: Tuple[int, int]
) -> Optional[VolumeLayout]:
    """Determines the tile layout based on base W/H and globally cropped depth."""
    try:
        global_z_start, global_z_end = global_z_range
        d = global_z_end - global_z_start + 1
        if d <= 0:
            log.error(
                f"Layout failed: Global depth calculation resulted in <= 0 ({d})"
                f" from Z-range {global_z_range}"
            )
            return None
        if base_width <= 0 or base_height <= 0:
            log.error(
                f"Layout failed: Invalid base dimensions W={base_width}, H={base_height}."
            )
            return None
        log.info(
            f"Determining layout for base W={base_width}, H={base_height},"
            f" Global Depth={d} (from Z-range {global_z_range})"
        )
        cols = math.ceil(math.sqrt(d))
        rows = math.ceil(d / cols)
        tile_w = cols * base_width
        tile_h = rows * base_height
        # Uses VolumeLayout imported from data_models
        layout = VolumeLayout(
            width=base_width,
            height=base_height,
            depth=d,
            cols=cols,
            rows=rows,
            tile_width=tile_w,
            tile_height=tile_h,
        )
        log.info(
            f"Layout determined: Input Volume({base_width}x{base_height}x{d}),"
            f" Output Tile({tile_w}x{tile_h}), Grid({cols}x{rows})"
        )
        return layout
    except Exception as e:
        log.error(f"❌ Layout determination failed unexpectedly: {e}", exc_info=True)
        return None


# === Task Preparation ===
def _prepare_tasks_and_layout(
    config: PreprocessingConfig, timepoints_data: TimepointsDict
) -> Tuple[Optional[Tuple[int, int]], Optional[VolumeLayout], List[ProcessingTask]]:
    """
    Orchestrates Pass 0 (via crop module), determines layout, and prepares tasks.
    Args are typed using classes imported from data_models.
    """
    layout: Optional[VolumeLayout] = None
    global_z_range: Optional[Tuple[int, int]] = None
    base_dims: Optional[Tuple[int, int]] = None
    # Type hint uses ProcessingTask imported from data_models
    tasks_to_submit: List[ProcessingTask] = []
    sorted_time_ids = sorted(timepoints_data.keys())
    if not sorted_time_ids:
        log.warning("No timepoints found in the scanned data. Cannot prepare tasks.")
        return None, None, []
    for time_id in sorted_time_ids:
        # Assumes ChannelEntry is handled correctly within TimepointsDict typing
        for entry in timepoints_data[time_id]:
            # Creates instance using ProcessingTask imported from data_models
            tasks_to_submit.append(ProcessingTask(time_id, entry, config))
    if not tasks_to_submit:
        log.error("❌ Failed to create any processing tasks from input files.")
        return None, None, []
    log.info(f"Prepared {len(tasks_to_submit)} initial tasks for Pass 0.")
    try:
        # Calls crop.determine_global_z_crop_and_dims, passing instances
        global_z_range, base_dims = crop.determine_global_z_crop_and_dims(
            tasks_to_submit, config
        )
    except Exception as e:
        # Log error details if debug is enabled in config
        log.error(
            f"❌ Critical error during Pass 0 execution: {e}", exc_info=config.debug
        )
        return None, None, []
    if global_z_range is None:
        log.error("Aborting: Failed to determine global Z-crop range in Pass 0.")
        return None, None, []
    if base_dims is None:
        log.error("Aborting: Failed to determine base dimensions (W, H) in Pass 0.")
        # If Z range is valid but dims failed, return Z range for potential debugging
        return global_z_range, None, []
    # Determine layout using the results from Pass 0
    layout = _determine_layout(base_dims[0], base_dims[1], global_z_range)
    if layout is None:
        log.error("❌ Aborting: Failed to determine valid tile layout.")
        # Return Z range if layout failed
        return global_z_range, None, []
    log.info(
        f"✅ Task preparation complete. Found {len(tasks_to_submit)} tasks."
        f" Global Z Range: {global_z_range}. Layout determined."
    )
    # Return type uses ProcessingTask imported from data_models
    return global_z_range, layout, tasks_to_submit


# === Main Orchestration Function ===
def run_preprocessing(config: PreprocessingConfig):  # Accepts config object
    """
    Runs the main PixelPacker preprocessing pipeline using refactored modules.

    Args:
        config: The fully resolved PreprocessingConfig object.
    """
    start_time = time.time()
    # Pipeline start is logged in cli.py where config is finalized

    try:
        # Config object is passed in directly
        log.debug(f"Core pipeline starting with config: {config}")

        # File Scanning uses pattern from config
        timepoints_data = scan_and_parse_files(
            config.input_folder, config.input_pattern
        )
        if not timepoints_data:
            log.error("Aborting pipeline: No valid input files found or parsed.")
            # Raise error to be caught by cli.py for cleaner exit code
            raise FileNotFoundError("No valid input files found or parsed.")

        # Prepare Tasks, Run Pass 0, Determine Layout
        global_z_range, layout, tasks = _prepare_tasks_and_layout(
            config, timepoints_data
        )
        if layout is None or not tasks or global_z_range is None:
            # Error logged in _prepare_tasks_and_layout
            raise ValueError("Failed during task prep/Z-crop/layout phase.")

        # Run Pass 1 (Calculate Limits)
        global_contrast_ranges, pass1_results = limits.calculate_global_limits(
            tasks, config, global_z_range
        )
        # calculate_global_limits raises ValueError if Pass 1 critically fails

        # Run Pass 2 (Process Channels)
        limits_for_processing_pass = (
            global_contrast_ranges if config.use_global_contrast else None
        )
        final_results = processing.execute_processing_pass(
            pass1_results, config, layout, limits_for_processing_pass
        )
        # Check if Pass 2 failed critically
        if not final_results and len(pass1_results) > 0:
            # Error logged in execute_processing_pass
            raise RuntimeError("Pass 2 failed to process any channels.")

        # Finalize Metadata & Write Manifest
        actual_global_ranges_used = (
            global_contrast_ranges if config.use_global_contrast else None
        )
        metadata = manifest.finalize_metadata(
            final_results, layout, global_z_range, actual_global_ranges_used
        )
        manifest.write_manifest(metadata, config)

    # Errors should propagate up to cli.py for handling there
    finally:
        # Log core processing time
        elapsed_time = time.time() - start_time
        # Using a distinct emoji/prefix for core timing
        log.info(f"⏱️ Core pipeline processing finished in {elapsed_time:.2f} seconds.")
