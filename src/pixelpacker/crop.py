# src/pixelpacker/crop.py

from concurrent.futures import as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm

# Import necessary I/O and config/task definitions
from .data_models import PreprocessingConfig, ProcessingTask
from .io_utils import extract_original_volume, find_z_crop_range
from .utils import log, get_executor


@dataclass
class CropRangeInfo:
    """Holds results from Pass 0 Z-range finding task."""

    path: Path
    z_start: int
    z_end: int
    width: Optional[int] = None
    height: Optional[int] = None


# === Pass 0: Determine Global Z-Crop Range ===


def _task_find_local_z_range(task: ProcessingTask) -> Optional[CropRangeInfo]:
    """
    Pass 0 Task: Extracts volume, finds local Z range, and dimensions.

    Args:
        task: The ProcessingTask containing file path and config.

    Returns:
        CropRangeInfo if successful, None otherwise.
    """
    ch_entry = task.channel_entry
    file_path = ch_entry["path"]
    config = task.config

    log.debug(
        f"Pass 0 task started for T:{task.time_id} C:{ch_entry['channel']}"
        f" ({file_path.name})"
    )
    original_volume = None
    try:
        # Extract the full original volume
        original_volume = extract_original_volume(file_path)
        if original_volume is None:
            log.warning(
                f"Pass 0 - Failed to load volume for T:{task.time_id}"
                f" C:{ch_entry['channel']}. Skipping."
            )
            return None

        # Check dimensions before proceeding
        if original_volume.ndim != 3:
            log.warning(
                f"Pass 0 - Expected 3D volume, got shape {original_volume.shape}"
                f" for T:{task.time_id} C:{ch_entry['channel']}. Skipping."
            )
            return None

        # Find the valid Z-range based on content
        z_start, z_end = find_z_crop_range(
            volume=original_volume,
            method=config.z_crop_method,
            threshold=config.z_crop_threshold,
            debug=config.debug,
            output_folder=config.output_folder,
            filename_prefix=f"T{task.time_id}_C{ch_entry['channel']}",
        )

        _, h, w = original_volume.shape  # Get dimensions

        log.debug(
            f"Pass 0 - Found local range [{z_start}-{z_end}], W={w}, H={h}"
            f" for T:{task.time_id} C:{ch_entry['channel']}"
        )
        return CropRangeInfo(
            path=file_path, z_start=z_start, z_end=z_end, width=w, height=h
        )
    except Exception as e:
        log.error(
            f"❌ Pass 0 - Unexpected error processing T:{task.time_id}"
            f" C:{ch_entry['channel']} ({file_path.name}): {e}",
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
    Pass 0 Orchestrator: Finds the global Z-crop range and base W/H across all tasks.

    Args:
        tasks: List of ProcessingTasks for all input files.
        config: The global preprocessing configuration.

    Returns:
        A tuple containing:
        - The global Z-crop range (start, end) or None if failed.
        - The base dimensions (width, height) or None if failed.
    """
    log.info(" Kicking off Pass 0: Determining global Z-crop range & dimensions...")
    num_tasks = len(tasks)
    if num_tasks == 0:
        log.warning("Pass 0 received no tasks.")
        return None, None

    min_z_start = float("inf")
    max_z_end = float("-inf")
    base_width: Optional[int] = None
    base_height: Optional[int] = None
    error_tasks = 0
    processed_tasks = 0

    with get_executor(config) as executor:
        futures = {
            executor.submit(_task_find_local_z_range, task): task for task in tasks
        }
        with tqdm(total=num_tasks, desc=" 🔪 Pass 0/3: Find Z Range & Dims") as pbar:
            for future in as_completed(futures):
                task = futures[future]  # Get task associated with this future
                try:
                    result: Optional[CropRangeInfo] = future.result()
                    if (
                        result
                        and result.width is not None
                        and result.height is not None
                    ):
                        processed_tasks += 1
                        min_z_start = min(min_z_start, result.z_start)
                        max_z_end = max(max_z_end, result.z_end)

                        # Set base dimensions from the first successful result
                        if base_width is None:
                            base_width = result.width
                            base_height = result.height
                            log.info(
                                f"Pass 0 - Base dims W={base_width}, H={base_height}"
                                f" determined from {result.path.name}"
                            )
                        # TODO: Add consistency check for W/H across files?
                    elif result:
                        log.warning(
                            f"Pass 0 - Task for {task.channel_entry['path'].name}"
                            " completed but missing width/height info."
                        )
                        error_tasks += 1
                    else:
                        # Task failed or returned None explicitly
                        log.warning(
                            f"Pass 0 - Task failed for T:{task.time_id}"
                            f" C:{task.channel_entry['channel']}"
                        )
                        error_tasks += 1
                except Exception as exc:
                    log.error(
                        f"❌ Pass 0 - Uncaught error from worker T:{task.time_id}"
                        f" C:{task.channel_entry['channel']}: {exc}",
                        exc_info=config.debug,
                    )
                    error_tasks += 1
                finally:
                    pbar.update(1)

    if error_tasks > 0:
        log.warning(f"Pass 0 completed with {error_tasks} errors or skipped files.")

    # --- Validate Results ---
    if processed_tasks == 0:
        log.error(
            "❌ Pass 0 failed: No Z-range or dimensions could be determined from any files."
        )
        return None, None

    if base_width is None or base_height is None:
        log.error("❌ Pass 0 failed: Could not determine base width/height.")
        # If we have a Z-range but no dims, return dims as None
        if min_z_start != float("inf") and max_z_end != float("-inf"):
            global_z_start = int(min_z_start)
            global_z_end = int(max_z_end)
            if global_z_start <= global_z_end:
                return (global_z_start, global_z_end), None
        return None, None  # No usable results

    if min_z_start == float("inf") or max_z_end == float("-inf"):
        log.error(
            f"❌ Pass 0 failed: Could not determine valid Z-range"
            f" (min={min_z_start}, max={max_z_end})."
        )
        # Return valid dims if we have them
        return None, (base_width, base_height)

    global_z_start = int(min_z_start)
    global_z_end = int(max_z_end)

    if global_z_start > global_z_end:
        # This case might indicate issues with the z_crop_method logic
        log.error(
            f"❌ Pass 0 failed: Determined Z-range is invalid"
            f" [{global_z_start}, {global_z_end}]. Min start is after Max end."
        )
        return None, (base_width, base_height)  # Return valid dims

    log.info(
        f"✅ Pass 0 finished. Global Z-crop range: [{global_z_start}, {global_z_end}]."
        f" Base Dims: W={base_width}, H={base_height}"
    )
    return (global_z_start, global_z_end), (base_width, base_height)
