# src/pixelpacker/crop.py

from concurrent.futures import as_completed
from typing import List, Optional, Tuple

from tqdm import tqdm

# Import necessary I/O and config/task definitions
from .data_models import PreprocessingConfig, ProcessingTask, CropRangeInfo
from .io_utils import (  # <<< MODIFIED
    extract_original_volume,
    find_z_crop_range,
    get_dimensions_from_metadata,  # <<< ADDED
)
from .utils import log, get_executor


# CropRangeInfo dataclass is now in data_models.py


# === Pass 0: Determine Global Z-Crop Range ===


def _task_find_local_z_range(task: ProcessingTask) -> Optional[CropRangeInfo]:
    """
    Pass 0 Task: Extracts volume, finds local Z range (ONLY if enabled), and dimensions.
    NOTE: This task now ALWAYS loads the full volume if Z-cropping is enabled,
          as the analysis requires pixel data.

    Args:
        task: The ProcessingTask containing file path and config.

    Returns:
        CropRangeInfo if successful, None otherwise.
    """
    ch_entry = task.channel_entry
    file_path = ch_entry["path"]
    config = task.config

    # This task is only called if Z-cropping is enabled.
    log.debug(
        "Pass 0 Z-Crop task started for T:%s C:%d (%s)",
        task.time_id,
        ch_entry["channel"],
        file_path.name,
    )
    original_volume = None
    try:
        # Extract the full original volume (needed for analysis)
        original_volume = extract_original_volume(file_path)
        if original_volume is None:
            log.warning(
                "Pass 0 - Failed to load volume for T:%s C:%d (Z-crop task). Skipping.",
                task.time_id,
                ch_entry["channel"],
            )
            return None

        # Check dimensions before proceeding
        if original_volume.ndim != 3:
            log.warning(
                "Pass 0 - Expected 3D volume, got shape %s for T:%s C:%d (Z-crop task). Skipping.",
                original_volume.shape,
                task.time_id,
                ch_entry["channel"],
            )
            return None

        depth, h, w = original_volume.shape  # Get dimensions

        # Find the valid Z-range based on content using selected method
        z_start, z_end = find_z_crop_range(
            volume=original_volume,
            method=config.z_crop_method,
            threshold=config.z_crop_threshold,
            debug=config.debug,
            output_folder=config.output_folder,
            filename_prefix=f"T{task.time_id}_C{ch_entry['channel']}",
        )

        log.debug(
            "Pass 0 - Determined Z-crop range [%d-%d], Original Depth=%d, W=%d, H=%d for T:%s C:%d",
            z_start,
            z_end,
            depth,
            w,
            h,
            task.time_id,
            ch_entry["channel"],
        )
        # Return info including the original depth
        return CropRangeInfo(
            path=file_path,
            z_start=z_start,
            z_end=z_end,
            original_depth=depth,
            width=w,  # Still return W/H, though might not be used if first file dims taken
            height=h,
        )
    except Exception as e:
        log.error(
            "❌ Pass 0 - Unexpected error during Z-crop task T:%s C:%d (%s): %s",
            task.time_id,
            ch_entry["channel"],
            file_path.name,
            e,
            exc_info=config.debug,
        )
        return None
    finally:
        # Ensure memory is released
        if original_volume is not None:
            del original_volume


def determine_global_z_crop_and_dims(
    tasks: List[ProcessingTask], config: PreprocessingConfig
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """
    Pass 0 Orchestrator: Finds base W/H and optionally global Z-crop range.

    Reads metadata from the first file to get dimensions.
    If Z-cropping is disabled, returns full Z range based on first file's depth.
    If Z-cropping is enabled, runs analysis on all files and aggregates/fallbacks.

    Args:
        tasks: List of ProcessingTasks for all input files.
        config: The global preprocessing configuration.

    Returns:
        A tuple containing:
        - The global Z-crop range (start, end) or None if failed.
        - The base dimensions (width, height) or None if failed.
    """
    num_tasks = len(tasks)
    if num_tasks == 0:
        log.warning("Pass 0 received no tasks.")
        return None, None

    # --- Get Dimensions from First File's Metadata ---
    log.info("🚀 Pass 0: Reading dimensions from first file's metadata...")
    first_task = tasks[0]
    first_file_path = first_task.channel_entry["path"]
    dims = get_dimensions_from_metadata(first_file_path)

    if dims is None:
        log.error(
            "❌ Pass 0 failed: Could not read dimensions from metadata of first file: %s",
            first_file_path.name,
        )
        return None, None

    initial_depth, base_height, base_width = dims
    base_dims = (base_width, base_height)
    log.info(
        "Pass 0 - Base dims W=%d, H=%d, D=%d determined from metadata of %s",
        base_width,
        base_height,
        initial_depth,
        first_file_path.name,
    )

    # --- Handle Z-Cropping Logic ---
    global_z_start = 0
    global_z_end = initial_depth - 1  # Default to full range of first file

    if not config.enable_z_crop:
        log.info(
            "ℹ️ Pass 0: Z-cropping is disabled. Using full Z-range [%d, %d] from first file.",
            global_z_start,
            global_z_end,
        )
        # Return early - no need to process other files for Pass 0
        log.info(
            "📊 Pass 0 finished (Z-crop disabled). Global Z-crop range: [%d, %d]. Base Dims: W=%d, H=%d",
            global_z_start,
            global_z_end,
            base_width,
            base_height,
        )
        return (global_z_start, global_z_end), base_dims

    # --- Z-Cropping IS Enabled: Run Analysis on All Files ---
    log.info("ℹ️ Pass 0: Z-cropping enabled. Analyzing all files...")
    all_results: List[CropRangeInfo] = []
    error_tasks = 0
    processed_tasks = 0
    max_original_depth = initial_depth  # Initialize with depth from first file

    with get_executor(config) as executor:
        # Submit tasks that actually load data and perform analysis
        futures = {
            executor.submit(_task_find_local_z_range, task): task for task in tasks
        }
        with tqdm(total=num_tasks, desc="🔪 Pass 0/3: Analyze Z-Crop") as pbar:
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result: Optional[CropRangeInfo] = future.result()
                    if (
                        result
                        and result.width is not None  # Check required fields
                        and result.height is not None
                        and result.original_depth is not None
                    ):
                        processed_tasks += 1
                        all_results.append(result)
                        # Update max depth seen across all files analysed
                        max_original_depth = max(
                            max_original_depth, result.original_depth
                        )
                    elif result:
                        log.warning(
                            "Pass 0 - Z-crop task for %s completed but missing info.",
                            task.channel_entry["path"].name,
                        )
                        error_tasks += 1
                    else:
                        log.warning(
                            "Pass 0 - Z-crop task failed for T:%s C:%d",
                            task.time_id,
                            task.channel_entry["channel"],
                        )
                        error_tasks += 1
                except Exception as exc:
                    log.error(
                        "❌ Pass 0 - Uncaught Z-crop worker error T:%s C:%d: %s",
                        task.time_id,
                        task.channel_entry["channel"],
                        exc,
                        exc_info=config.debug,
                    )
                    error_tasks += 1
                finally:
                    pbar.update(1)

    if error_tasks > 0:
        log.warning(
            "Pass 0 Z-crop analysis completed with %d errors/skipped.", error_tasks
        )
    if processed_tasks == 0:
        log.error("❌ Pass 0 failed: No files successfully analyzed for Z-cropping.")
        # Return dims determined from metadata, but None for Z-range
        return None, base_dims

    # --- Aggregate Z-Crop Results / Check Fallback ---
    log.debug("Pass 0 - Analyzing collected Z-crop ranges...")
    trigger_fallback = False
    min_z_start_agg = float("inf")
    max_z_end_agg = float("-inf")

    for result in all_results:
        if (
            result.original_depth is not None
            and result.z_start == 0
            and result.z_end == result.original_depth - 1
        ):
            log.warning(
                "Pass 0 - Fallback detected for file: %s (used full range [0, %d]). Global Z-crop will default to full range across all files.",
                result.path.name,
                result.original_depth - 1,
            )
            trigger_fallback = True
            break  # Stop checking

    if not trigger_fallback:
        # Aggregate if no fallback triggered
        for result in all_results:
            min_z_start_agg = min(min_z_start_agg, result.z_start)
            max_z_end_agg = max(max_z_end_agg, result.z_end)

        # Validate aggregated range
        if min_z_start_agg == float("inf") or max_z_end_agg == float("-inf"):
            log.error("❌ Pass 0 failed: Could not aggregate a valid Z-range.")
            trigger_fallback = True  # Fallback to full range
        elif min_z_start_agg > max_z_end_agg:
            log.error(
                "❌ Pass 0 failed: Aggregated Z-range invalid [%d, %d].",
                int(min_z_start_agg),
                int(max_z_end_agg),
            )
            trigger_fallback = True  # Fallback to full range
        else:
            # Use the valid aggregated range
            global_z_start = int(min_z_start_agg)
            global_z_end = int(max_z_end_agg)
            log.info(
                "✅ Pass 0 determined AGGREGATED Z-crop range: [%d, %d]",
                global_z_start,
                global_z_end,
            )

    # Apply fallback if needed (use max depth seen across all files)
    if trigger_fallback:
        global_z_start = 0
        global_z_end = max_original_depth - 1
        log.info(
            "✅ Pass 0 determined FALLBACK Z-crop range: [%d, %d] (using max observed depth)",
            global_z_start,
            global_z_end,
        )

    # --- Final Log ---
    log.info(
        "📊 Pass 0 finished (Z-crop enabled). Global Z-crop range: [%d, %d]. Base Dims: W=%d, H=%d",
        global_z_start,
        global_z_end,
        base_width,
        base_height,
    )
    return (global_z_start, global_z_end), base_dims
