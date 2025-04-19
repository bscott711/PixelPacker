# src/pixelpacker/core.py

import math
import time
# REMOVE: from dataclasses import dataclass # No longer needed here
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- Centralized Utilities ---
# Import the logger configured in utils.py
from .utils import log, scan_and_parse_files, TimepointsDict

# --- Refactored Modules ---
from . import crop
from . import limits
from . import manifest
from . import processing

# --- Data Models & Types ---
# Import specific data models used by the pipeline orchestration
# Ensure ALL needed shared dataclasses are imported from data_models
from .data_models import (
    PreprocessingConfig,
    ProcessingTask,
    VolumeLayout,
)

# Type alias remains useful here (or could move to data_models)
TimepointResult = Dict[str, Any]

# === Layout Determination (Stays with Orchestration for now) ===
def _determine_layout(
    base_width: int, base_height: int, global_z_range: Tuple[int, int]
) -> Optional[VolumeLayout]:
    """Determines the tile layout based on base W/H and globally cropped depth."""
    # No changes needed in the function body assuming VolumeLayout is imported correctly
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


# === Task Preparation (Stays with Orchestration) ===
def _prepare_tasks_and_layout(
    config: PreprocessingConfig, timepoints_data: TimepointsDict
) -> Tuple[Optional[Tuple[int, int]], Optional[VolumeLayout], List[ProcessingTask]]:
    """
    Orchestrates Pass 0 (via crop module), determines layout, and prepares tasks.
    Args are typed using classes imported from data_models.
    """
    # No changes needed in the function body assuming types resolve correctly
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
        for entry in timepoints_data[time_id]:
            # Creates instance using ProcessingTask imported from data_models
            tasks_to_submit.append(ProcessingTask(time_id, entry, config))

    if not tasks_to_submit:
        log.error("❌ Failed to create any processing tasks from input files.")
        return None, None, []

    log.info(f"Prepared {len(tasks_to_submit)} initial tasks for Pass 0.")

    try:
        # Calls crop.determine_global_z_crop_and_dims, passing instances
        # whose types should now correctly resolve to data_models types.
        global_z_range, base_dims = crop.determine_global_z_crop_and_dims(
            tasks_to_submit, config
        )
    except Exception as e:
        log.error(f"❌ Critical error during Pass 0 execution: {e}", exc_info=config.debug)
        return None, None, []

    if global_z_range is None:
        log.error("Aborting: Failed to determine global Z-crop range in Pass 0.")
        return None, None, []
    if base_dims is None:
         log.error("Aborting: Failed to determine base dimensions (W, H) in Pass 0.")
         return global_z_range, None, []

    layout = _determine_layout(base_dims[0], base_dims[1], global_z_range)
    if layout is None:
        log.error("❌ Aborting: Failed to determine valid tile layout.")
        return global_z_range, None, []

    log.info(
        f"✅ Task preparation complete. Found {len(tasks_to_submit)} tasks."
        f" Global Z Range: {global_z_range}. Layout determined."
    )
    # Return type uses ProcessingTask imported from data_models
    return global_z_range, layout, tasks_to_submit


# === Config Setup (Stays near entry point) ===
def _setup_configuration(args: Dict[str, Any]) -> PreprocessingConfig:
    """
    Sets up PreprocessingConfig from command-line arguments or defaults.
    Return type uses PreprocessingConfig imported from data_models.
    """
    # No changes needed in the function body assuming types resolve correctly
    try:
        input_folder = Path(args["--input"]).resolve()
        output_folder = Path(args["--output"]).resolve()

        if not input_folder.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_folder}")

        output_folder.mkdir(parents=True, exist_ok=True)
        log.info(f"Ensured output directory exists: {output_folder}")

        # Creates instance using PreprocessingConfig imported from data_models
        config = PreprocessingConfig(
            input_folder=input_folder,
            output_folder=output_folder,
            stretch_mode=args["--stretch"],
            z_crop_method=args["--z-crop-method"],
            z_crop_threshold=int(args["--z-crop-threshold"]),
            use_global_contrast=args["--global-contrast"],
            dry_run=args["--dry-run"],
            debug=args["--debug"],
            max_threads=int(args["--threads"]),
        )

        log.info("Preprocessing configuration loaded successfully.")
        log.debug(f"Configuration details: {config}")
        return config

    except KeyError as e:
        msg = f"Configuration error: Missing required argument '{e}'"
        log.error(msg)
        raise ValueError(msg) from e
    except ValueError as e:
        msg = f"Configuration error: Invalid value provided ({e})"
        log.error(msg)
        raise ValueError(msg) from e
    except FileNotFoundError as e:
        log.error(f"Configuration error: {e}")
        raise
    except Exception as e:
        log.error(f"Unexpected configuration error: {e}", exc_info=True)
        raise ValueError(f"Unexpected configuration error: {e}") from e


# === Main Orchestration Function ===
def run_preprocessing(args: Dict[str, Any]):
    """
    Runs the main PixelPacker preprocessing pipeline using refactored modules.
    """
    # No changes needed in the function body assuming types resolve correctly
    start_time = time.time()
    log.info("🚀 Starting PixelPacker Preprocessing Pipeline...")

    try:
        config = _setup_configuration(args)
        timepoints_data = scan_and_parse_files(config.input_folder)

        if not timepoints_data:
            log.error("❌ Aborting: No valid input files found or parsed.")
            return

        global_z_range, layout, tasks = _prepare_tasks_and_layout(
            config, timepoints_data
        )

        if layout is None or not tasks or global_z_range is None:
            log.error("❌ Aborting: Failed during task preparation, Z-cropping, or layout.")
            return

        global_contrast_ranges, pass1_results = limits.calculate_global_limits(
            tasks, config, global_z_range
        )

        limits_for_processing_pass = (
            global_contrast_ranges if config.use_global_contrast else None
        )
        final_results = processing.execute_processing_pass(
            pass1_results, config, layout, limits_for_processing_pass
        )

        actual_global_ranges_used = (
             global_contrast_ranges if config.use_global_contrast else None
        )
        # Ensure final_results uses ProcessingResult imported from data_models
        metadata = manifest.finalize_metadata(
            final_results, layout, global_z_range, actual_global_ranges_used
        )
        manifest.write_manifest(metadata, config)

    except (ValueError, FileNotFoundError, OSError) as e:
        log.error(f"❌ Preprocessing pipeline aborted due to error: {e}")
        return

    except Exception as e:
        log.critical(f"❌ An unexpected critical error occurred: {e}", exc_info=True)
        return

    finally:
        elapsed_time = time.time() - start_time
        log.info(f"🏁 Pipeline finished in {elapsed_time:.2f} seconds.")