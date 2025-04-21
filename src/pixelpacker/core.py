# src/pixelpacker/core.py

# --- Standard Imports ---
import math
import time  # Added for timing
import logging  # Added for logger access
from typing import Any, Dict, List, Optional, Tuple

from .utils import scan_and_parse_files, TimepointsDict  # Keep existing imports

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


# --- Centralized Utilities ---
# Imports log, scan_and_parse_files, TimepointsDict from utils
# Get logger instance directly if not already imported/available
log = logging.getLogger(__name__)  # Ensures log is available


# Type alias remains useful here
TimepointResult = Dict[str, Any]


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
                f" from Z-range {global_z_range}",
                extra={"z_range": global_z_range, "calculated_depth": d},
            )
            return None
        if base_width <= 0 or base_height <= 0:
            log.error(
                f"Layout failed: Invalid base dimensions W={base_width}, H={base_height}.",
                extra={"base_width": base_width, "base_height": base_height},
            )
            return None
        log.info(
            "Determining layout",
            extra={
                "base_width": base_width,
                "base_height": base_height,
                "global_depth": d,
                "z_range": global_z_range,
            },
        )
        cols = math.ceil(math.sqrt(d))
        rows = math.ceil(d / cols)
        tile_w = cols * base_width
        tile_h = rows * base_height

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
            "Layout determined",
            extra={
                "input_volume_WHD": (base_width, base_height, d),
                "output_tile_WH": (tile_w, tile_h),
                "grid_ColsRows": (cols, rows),
                "layout_object": layout,  # Consider if logging the object is too verbose
            },
        )
        return layout
    except Exception as e:
        log.error(f"Layout determination failed unexpectedly: {e}", exc_info=True)
        return None


# === Task Preparation ===
def _prepare_tasks_and_layout(
    config: PreprocessingConfig, timepoints_data: TimepointsDict
) -> Tuple[Optional[Tuple[int, int]], Optional[VolumeLayout], List[ProcessingTask]]:
    """
    Orchestrates Pass 0 (via crop module), determines layout, and prepares tasks.
    """
    layout: Optional[VolumeLayout] = None
    global_z_range: Optional[Tuple[int, int]] = None
    base_dims: Optional[Tuple[int, int]] = None
    tasks_to_submit: List[ProcessingTask] = []
    sorted_time_ids = sorted(timepoints_data.keys())

    if not sorted_time_ids:
        log.warning("No timepoints found in the scanned data. Cannot prepare tasks.")
        return None, None, []

    for time_id in sorted_time_ids:
        for entry in timepoints_data[time_id]:
            tasks_to_submit.append(ProcessingTask(time_id, entry, config))

    if not tasks_to_submit:
        log.error("Failed to create any processing tasks from input files.")
        return None, None, []

    log.info(f"Prepared {len(tasks_to_submit)} initial tasks for Pass 0.")

    try:
        # Pass 0: Determine Global Z-Crop Range & Dimensions
        # This internally calls crop.determine_global_z_crop_and_dims
        global_z_range, base_dims = crop.determine_global_z_crop_and_dims(
            tasks_to_submit, config
        )
    except Exception as e:
        log.error(f"Critical error during Pass 0 execution: {e}", exc_info=config.debug)
        return None, None, []

    # Validate results from Pass 0
    if global_z_range is None:
        log.error("Aborting: Failed to determine global Z-crop range in Pass 0.")
        return None, None, []
    if base_dims is None:
        log.error("Aborting: Failed to determine base dimensions (W, H) in Pass 0.")
        return global_z_range, None, []

    # Determine layout using the results from Pass 0
    layout = _determine_layout(base_dims[0], base_dims[1], global_z_range)
    if layout is None:
        log.error("Aborting: Failed to determine valid tile layout.")
        return global_z_range, None, []

    log.info(
        "Task preparation complete",
        extra={
            "num_tasks": len(tasks_to_submit),
            "global_z_range": global_z_range,
            "layout_determined": True,
        },
    )
    return global_z_range, layout, tasks_to_submit


# === Main Orchestration Function ===
def run_preprocessing(config: PreprocessingConfig):
    """
    Runs the main PixelPacker preprocessing pipeline using refactored modules.

    Args:
        config: The fully resolved PreprocessingConfig object.
    """
    pipeline_start_time = time.time()
    # Log config only if debugging to avoid excessive log size
    if config.debug:
        # Use pformat for potentially large/nested config object
        from pprint import pformat

        log.debug("Core pipeline starting", extra={"config": pformat(config)})
    else:
        log.debug("Core pipeline starting")  # Simple message otherwise

    # --- Prepare Timing Variables ---
    timing_metrics: Dict[str, float] = {}  # Initialize dictionary

    try:
        # --- File Scanning ---
        scan_start = time.time()
        timepoints_data = scan_and_parse_files(
            config.input_folder, config.input_pattern
        )
        scan_duration = time.time() - scan_start
        timing_metrics["scan_files_sec"] = scan_duration
        log.debug("Stage 'Scan Files' finished", extra={"duration_sec": scan_duration})

        if not timepoints_data:
            log.error("Aborting pipeline: No valid input files found or parsed.")
            raise FileNotFoundError("No valid input files found or parsed.")

        # --- Task Prep / Pass 0 / Layout ---
        prep_start = time.time()
        global_z_range, layout, tasks = _prepare_tasks_and_layout(
            config, timepoints_data
        )
        prep_duration = time.time() - prep_start
        timing_metrics["prep_pass0_layout_sec"] = prep_duration
        log.debug(
            "Stage 'Prep/Pass0/Layout' finished", extra={"duration_sec": prep_duration}
        )

        # Check results from prep stage
        if layout is None or not tasks or global_z_range is None:
            # Specific error logged within _prepare_tasks_and_layout
            raise ValueError("Failed during task prep/Z-crop/layout phase.")

        # --- Pass 1 (Calculate Limits) ---
        pass1_start = time.time()
        global_contrast_ranges, pass1_results = limits.calculate_global_limits(
            tasks, config, global_z_range
        )
        pass1_duration = time.time() - pass1_start
        timing_metrics["pass1_limits_sec"] = pass1_duration
        log.debug(
            "Stage 'Pass 1 (Limits)' finished", extra={"duration_sec": pass1_duration}
        )
        # calculate_global_limits raises ValueError if it fails critically

        # --- Pass 2 (Process Channels) ---
        pass2_start = time.time()
        limits_for_processing_pass = (
            global_contrast_ranges if config.use_global_contrast else None
        )
        final_results, processing_error_count = processing.execute_processing_pass(
            pass1_results, config, layout, limits_for_processing_pass
        )
        pass2_duration = time.time() - pass2_start
        timing_metrics["pass2_processing_sec"] = pass2_duration
        log.debug(
            "Stage 'Pass 2 (Processing)' finished",
            extra={"duration_sec": pass2_duration},
        )

        if processing_error_count > 0:
            log.error(
                f"❌ Pipeline aborted: Encountered {processing_error_count} error(s) during channel processing."
            )
            raise RuntimeError(
                f"Pipeline completed with {processing_error_count} processing error(s)."
            )

        # Check if Pass 2 failed critically (logged within execute_processing_pass)
        if not final_results and len(pass1_results) > 0:
            raise RuntimeError("Pass 2 failed to process any channels.")

        # --- Manifest ---
        manifest_start = time.time()
        actual_global_ranges_used = (
            global_contrast_ranges if config.use_global_contrast else None
        )
        metadata = manifest.finalize_metadata(
            final_results, layout, global_z_range, actual_global_ranges_used
        )
        manifest.write_manifest(metadata, config)
        manifest_duration = time.time() - manifest_start
        timing_metrics["manifest_sec"] = manifest_duration
        log.debug(
            "Stage 'Manifest' finished", extra={"duration_sec": manifest_duration}
        )

    # Errors should propagate up to cli.py for handling there
    finally:
        # --- Log Final Timings ---
        total_elapsed_time = time.time() - pipeline_start_time
        timing_metrics["total_pipeline_sec"] = total_elapsed_time

        # Always log the total time
        log.info(f"⏱️ Pipeline finished. Total time: {total_elapsed_time:.2f}s.")

        # Conditionally log the detailed stage breakdown if debug is enabled
        # We check config directly, assuming run_preprocessing always receives it
        if config and config.debug:
            log.debug(
                "Detailed stage timings (sec)",
                extra={"stage_timings": timing_metrics},  # Log the dictionary
            )
