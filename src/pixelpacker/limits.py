# src/pixelpacker/limits.py

from collections import defaultdict
from concurrent.futures import as_completed
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Import necessary I/O, stretch functions, and config/task definitions
from .data_models import (
    PreprocessingConfig,
    ProcessingTask,
)  # Import shared dataclasses
from .io_utils import extract_original_volume
from .stretch import ContrastLimits, calculate_limits_only
from .utils import log, get_executor


@dataclass
class LimitsPassResult:
    """Holds results from Pass 1 limit calculation task."""

    time_id: str
    channel: int
    limits: ContrastLimits
    # Keep the cropped volume in memory for Pass 2
    globally_cropped_volume: Optional[np.ndarray]
    # Include global range for context/validation if needed later
    global_z_range: Tuple[int, int]


# === Pass 1: Calculate Contrast Limits ===


def _task_calculate_limits(
    task: ProcessingTask, global_z_range: Tuple[int, int]
) -> Optional[LimitsPassResult]:
    """
    Pass 1 Task: Extracts volume, applies GLOBAL Z-crop, calculates contrast limits.

    Crucially, this task *keeps* the globally Z-cropped volume in memory
    within the LimitsPassResult for use in Pass 2, avoiding reloading.

    Args:
        task: The ProcessingTask containing file path and config.
        global_z_range: The Z-range (start, end) determined in Pass 0.

    Returns:
        LimitsPassResult containing limits and the cropped volume, or None on failure.
    """
    ch_entry = task.channel_entry
    file_path = ch_entry["path"]
    config = task.config

    log.debug(
        f"Pass 1 task started for T:{task.time_id} C:{ch_entry['channel']}"
        f" using Z-range {global_z_range}"
    )
    original_volume = None
    globally_cropped_volume = None
    try:
        # Load the original volume again (necessary for cropping)
        original_volume = extract_original_volume(file_path)
        if original_volume is None:
            log.warning(
                f"Pass 1 - Failed to load volume for T:{task.time_id}"
                f" C:{ch_entry['channel']}. Skipping."
            )
            return None

        if original_volume.ndim != 3:
            log.warning(
                f"Pass 1 - Expected 3D volume, got shape {original_volume.shape}"
                f" for T:{task.time_id} C:{ch_entry['channel']}. Skipping."
            )
            return None

        # Apply the GLOBAL Z-crop determined in Pass 0
        z_start, z_end = global_z_range
        original_depth = original_volume.shape[0]

        # Clamp Z-range to the actual dimensions of *this* volume
        clamped_z_start = max(0, z_start)
        # Use inclusive index for slicing numpy arrays
        clamped_z_end = min(original_depth - 1, z_end)

        # Validate the clamped range before slicing
        if (
            clamped_z_start > clamped_z_end  # Range is inverted
            or clamped_z_start >= original_depth  # Start is out of bounds
            or clamped_z_end < 0  # End is out of bounds (less likely but safe)
        ):
            log.warning(
                f"Pass 1 - Global Z range [{z_start}-{z_end}] is invalid or outside"
                f" the bounds [0-{original_depth - 1}] for T:{task.time_id}"
                f" C:{ch_entry['channel']}. Skipping."
            )
            # Explicitly delete original volume before returning
            del original_volume
            return None

        # Perform the slice to get the globally Z-cropped volume
        globally_cropped_volume = original_volume[
            clamped_z_start : clamped_z_end + 1, :, :
        ]

        # We no longer need the full original volume in memory
        del original_volume
        original_volume = None

        # Check if the cropped volume is empty
        if globally_cropped_volume.size == 0:
            log.warning(
                f"Pass 1 - Cropped volume is empty for T:{task.time_id}"
                f" C:{ch_entry['channel']} (Z-range [{clamped_z_start}-{clamped_z_end}])."
                " Skipping."
            )
            return None  # Do not return a result with an empty volume

        # Calculate contrast limits on the *cropped* volume
        limits = calculate_limits_only(globally_cropped_volume, config.stretch_mode)
        log.debug(
            f"Pass 1 - Calculated limits {limits} for T:{task.time_id}"
            f" C:{ch_entry['channel']}"
        )

        # Return the result, including the cropped volume
        return LimitsPassResult(
            time_id=task.time_id,
            channel=ch_entry["channel"],
            limits=limits,
            globally_cropped_volume=globally_cropped_volume,
            global_z_range=global_z_range,
        )

    except Exception as e:
        log.error(
            f"❌ Pass 1 - Unexpected error processing T:{task.time_id}"
            f" C:{ch_entry['channel']}: {e}",
            exc_info=config.debug,
        )
        # Ensure cleanup even on unexpected errors
        if original_volume is not None:
            del original_volume
        if globally_cropped_volume is not None:
            del globally_cropped_volume
        return None


def calculate_global_limits(
    tasks: List[ProcessingTask],
    config: PreprocessingConfig,
    global_z_range: Tuple[int, int],
) -> Tuple[Dict[int, Tuple[float, float]], List[LimitsPassResult]]:
    """
    Pass 1 Orchestrator: Calculates contrast limits for each task.

    If `config.use_global_contrast` is True, it also aggregates these limits
    to find the overall min/max across all timepoints for each channel.

    Args:
        tasks: List of ProcessingTasks.
        config: The global preprocessing configuration.
        global_z_range: The global Z-range tuple (start, end).

    Returns:
        A tuple containing:
        - A dictionary mapping channel ID to global (low, high) contrast limits.
          If global contrast is not used, this might be based on aggregation anyway
          or could be empty depending on desired manifest output.
        - A list of LimitsPassResult objects, each containing the calculated limits
          and the *globally Z-cropped volume* for that specific task.
    """
    log.info(
        " Kicking off Pass 1: Applying global Z-crop & calculating contrast limits..."
    )
    num_tasks = len(tasks)
    if num_tasks == 0:
        log.warning("Pass 1 received no tasks.")
        return {}, []

    # Dictionary to aggregate global min/max if needed
    global_range_agg: DefaultDict[int, Dict[str, float]] = defaultdict(
        lambda: {"p_low": float("inf"), "p_high": float("-inf")}
    )

    # Store individual results (including cropped volumes) from each task
    pass1_individual_results: List[LimitsPassResult] = []
    error_tasks = 0

    with get_executor(config) as executor:
        # Submit all tasks for limit calculation
        futures = {
            executor.submit(_task_calculate_limits, task, global_z_range): task
            for task in tasks
        }
        with tqdm(total=num_tasks, desc=" 🔭 Pass 1/3: Calc Limits") as pbar:
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result: Optional[LimitsPassResult] = future.result()
                    if result and result.globally_cropped_volume is not None:
                        pass1_individual_results.append(result)
                        # Aggregate for potential global contrast calculation
                        ch_id = result.channel
                        global_range_agg[ch_id]["p_low"] = min(
                            global_range_agg[ch_id]["p_low"], result.limits.p_low
                        )
                        global_range_agg[ch_id]["p_high"] = max(
                            global_range_agg[ch_id]["p_high"], result.limits.p_high
                        )
                    else:
                        # Task failed or returned None / empty volume
                        log.warning(
                            f"Pass 1 - Task failed or produced no valid result for T:{task.time_id}"
                            f" C:{task.channel_entry['channel']}"
                        )
                        error_tasks += 1
                except Exception as exc:
                    log.error(
                        f"❌ Pass 1 - Uncaught error from worker T:{task.time_id}"
                        f" C:{task.channel_entry['channel']}: {exc}",
                        exc_info=config.debug,
                    )
                    error_tasks += 1
                finally:
                    pbar.update(1)

    # Finalize the aggregated global ranges
    global_ranges_final: Dict[int, Tuple[float, float]] = {}
    for ch_id, limits_agg in global_range_agg.items():
        final_low = limits_agg["p_low"] if limits_agg["p_low"] != float("inf") else 0.0
        final_high = (
            limits_agg["p_high"] if limits_agg["p_high"] != float("-inf") else 0.0
        )

        # Handle cases where min/max are identical (e.g., constant image)
        # or invalid (max < min if only one file failed weirdly)
        # Add a small epsilon to ensure valid range for stretching.
        if final_high <= final_low:
            log.warning(
                f"Pass 1 - Aggregated global range for C:{ch_id} is zero or"
                f" inverted [{final_low:.4g}, {final_high:.4g}]."
                " Adjusting high limit slightly."
            )
            # Ensure high is at least slightly larger than low
            final_high = max(final_high, final_low + 1e-6)

            # If range was [0, 0] and became [0, 1e-6], warn about tiny range
            if final_low == 0 and final_high == 1e-6:
                log.warning(
                    f"Pass 1 - Adjusted C:{ch_id} global range is very small"
                    f" [{final_low:.2g}, {final_high:.2g}]."
                    " This might indicate constant image data."
                )

        global_ranges_final[ch_id] = (final_low, final_high)
        log.info(
            f"Pass 1 - Aggregated Global Range C:{ch_id}:"
            f" ({global_ranges_final[ch_id][0]:.4f},"
            f" {global_ranges_final[ch_id][1]:.4f})"
        )

    if error_tasks > 0:
        log.warning(f"Pass 1 completed with {error_tasks} errors or skipped files.")

    if not pass1_individual_results:
        # Raise an error if *no* files could be processed in Pass 1
        # This prevents proceeding to Pass 2 with no data.
        raise ValueError(
            "Pass 1 failed critically: No contrast limits could be calculated for any input files."
        )

    log.info(
        f"✅ Pass 1 finished. Calculated limits for {len(pass1_individual_results)} tasks."
    )
    # Return both the aggregated global ranges AND the list of individual results
    # (which importantly contain the cropped volumes needed for Pass 2)
    return global_ranges_final, pass1_individual_results
