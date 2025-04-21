# src/pixelpacker/processing.py

from concurrent.futures import as_completed
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

# Import necessary I/O, config, data models, and results from previous passes
from .data_models import (
    PreprocessingConfig,
    ProcessingResult,
    VolumeLayout,
)
from .io_utils import process_channel
from .limits import LimitsPassResult  # Result from Pass 1
from .stretch import ContrastLimits
from .utils import log, get_executor


# === Pass 2: Process Channels (Tile & Save) ===


def _task_process_channel(
    pass1_result: LimitsPassResult,
    layout: VolumeLayout,
    config: PreprocessingConfig,
    global_contrast_override: Optional[Tuple[float, float]] = None,
) -> Optional[ProcessingResult]:
    """
    Pass 2 Task: Takes a cropped volume from Pass 1, applies contrast, tiles, and saves.

    Args:
        pass1_result: The result object from Pass 1 containing the cropped volume
                      and per-image contrast limits.
        layout: The calculated VolumeLayout for tiling.
        config: The global preprocessing configuration.
        global_contrast_override: Optional tuple (low, high) to use instead of
                                  the per-image limits from pass1_result.

    Returns:
        ProcessingResult metadata if successful, None otherwise.
    """
    time_id = pass1_result.time_id
    channel = pass1_result.channel
    log.debug(f"Pass 2 task started for T:{time_id} C:{channel}")

    globally_cropped_volume = None
    try:
        # Retrieve the pre-cropped volume from the Pass 1 result
        globally_cropped_volume = pass1_result.globally_cropped_volume

        if globally_cropped_volume is None:
            # This shouldn't happen if Pass 1 filtering works, but check defensively
            log.warning(
                f"Pass 2 - Missing globally cropped volume for T:{time_id} C:{channel}."
                " Skipping."
            )
            return None

        # Determine the final contrast limits to use
        final_limits: ContrastLimits
        if config.use_global_contrast and global_contrast_override:
            log.debug(
                f"Pass 2 - Overriding limits for T:{time_id} C:{channel} with global"
                f" range: {global_contrast_override}"
            )
            # Create a new ContrastLimits object using the global override values
            final_limits = ContrastLimits(
                p_low=global_contrast_override[0], p_high=global_contrast_override[1]
            )
            # Carry over actual min/max from original calculation if needed for metadata/debugging
            # (though process_channel might recalculate if absolutely necessary)
            final_limits.actual_min = pass1_result.limits.actual_min
            final_limits.actual_max = pass1_result.limits.actual_max
        else:
            # Use the per-image limits calculated in Pass 1
            final_limits = pass1_result.limits
            if config.use_global_contrast and not global_contrast_override:
                log.warning(
                    f"Pass 2 - Global contrast requested but no override value found"
                    f" for C:{channel}. Falling back to per-image limits for T:{time_id}."
                )

        # Call the core processing function (from io_utils)
        result_dict = process_channel(
            time_id=time_id,
            ch_id=channel,
            # Pass the already cropped volume
            globally_cropped_vol=globally_cropped_volume,
            layout=layout,
            limits=final_limits,
            stretch_mode=config.stretch_mode,
            dry_run=config.dry_run,
            debug=config.debug,
            output_folder=str(config.output_folder),  # Ensure output path is string
        )

        # Check if processing was successful
        if result_dict:
            log.debug(
                f"Pass 2 - Successfully processed T:{time_id} C:{channel}"
                f" -> {result_dict.get('filename', 'N/A')}"
            )
            # Create the ProcessingResult dataclass for metadata aggregation
            return ProcessingResult(
                time_id=result_dict["time_id"],
                channel=result_dict["channel"],
                filename=result_dict["filename"],
                # Extract final used intensity range from the result dict
                p_low=result_dict["intensity_range"]["p_low"],
                p_high=result_dict["intensity_range"]["p_high"],
            )
        else:
            # process_channel returned None or an empty dict, indicating failure
            log.warning(
                f"Pass 2 - Core processing function failed for T:{time_id} C:{channel}."
            )
            return None

    except Exception as e:
        log.error(
            f"❌ Pass 2 - Unexpected error processing T:{time_id} C:{channel}: {e}",
            exc_info=config.debug,
        )
        return None
    finally:
        # VERY IMPORTANT: Clean up the large volume data from the Pass 1 result
        # after this task is done with it. This prevents memory ballooning.
        if hasattr(pass1_result, "globally_cropped_volume"):
            del pass1_result.globally_cropped_volume
            pass1_result.globally_cropped_volume = None


def execute_processing_pass(
    pass1_results: List[LimitsPassResult],
    config: PreprocessingConfig,
    layout: VolumeLayout,
    global_contrast_ranges: Optional[Dict[int, Tuple[float, float]]] = None,
) -> Tuple[List[ProcessingResult], int]:
    """
    Pass 2 Orchestrator: Executes the main processing (tiling, saving) for each task.

    Uses the pre-cropped volumes and limits determined in Pass 1.

    Args:
        pass1_results: List of results from Pass 1, containing cropped volumes
                       and per-image limits.
        config: The global preprocessing configuration.
        layout: The calculated VolumeLayout.
        global_contrast_ranges: Optional dictionary mapping channel ID to
                                (low, high) global contrast limits, used if
                                `config.use_global_contrast` is True.

    Returns:
        A list of ProcessingResult objects for successfully processed files.
    """
    log.info(" Kicking off Pass 2: Processing channels (tiling & saving)...")
    num_tasks = len(pass1_results)
    if num_tasks == 0:
        log.warning("Pass 2 received no results from Pass 1 to process.")
        return [], 0

    final_results: List[ProcessingResult] = []
    processed_count = 0
    error_count = 0

    with get_executor(config) as executor:
        futures = {}
        # Submit each Pass 1 result to the Pass 2 task function
        for p1_res in pass1_results:
            global_override = None
            # Determine if global contrast override should be used for this channel
            if config.use_global_contrast and global_contrast_ranges:
                global_override = global_contrast_ranges.get(p1_res.channel)
                # Log if override is expected but not found for a specific channel
                if not global_override:
                    log.warning(
                        f"Pass 2 - Global contrast requested but range not found for"
                        f" C:{p1_res.channel}. Will use per-image limits for T:{p1_res.time_id}."
                    )

            # Submit the task
            fut = executor.submit(
                _task_process_channel, p1_res, layout, config, global_override
            )
            # Store the future -> pass1_result mapping for error reporting
            # Note: p1_res object is mutated by _task_process_channel (deletes volume)
            futures[fut] = (p1_res.time_id, p1_res.channel)

        log.info(f"Pass 2 - Submitted {len(futures)} tasks for processing.")
        with tqdm(total=len(futures), desc=" ⚙️ Pass 2/3: Processing") as pbar:
            for fut in as_completed(futures):
                # Retrieve the original time_id and channel for logging,
                # as the p1_res object might be modified/deleted
                orig_time_id, orig_channel = futures[fut]
                try:
                    # Get the result from the task (ProcessingResult or None)
                    result_obj: Optional[ProcessingResult] = fut.result()

                    if result_obj:
                        processed_count += 1
                        final_results.append(result_obj)
                    else:
                        # Task completed but returned None, indicating failure within task
                        log.warning(
                            f"Pass 2 - Task for T:{orig_time_id} C:{orig_channel}"
                            " completed but returned no result (likely failed)."
                        )
                        error_count += 1

                except Exception as exc:
                    # An exception occurred *during* the execution of the future
                    log.error(
                        f"❌ Pass 2 - Uncaught error from worker T:{orig_time_id}"
                        f" C:{orig_channel}: {exc}",
                        exc_info=config.debug,
                    )
                    error_count += 1
                finally:
                    pbar.update(1)
                    # Memory cleanup now happens *inside* _task_process_channel's finally block

    log.info(
        f"📊 Pass 2 complete. Successful: {processed_count},"
        f" Errors/Skipped: {error_count}"
    )
    if processed_count == 0 and num_tasks > 0:
        log.error(
            "❌ Pass 2 critical failure: No channels were successfully processed."
        )

    return final_results, error_count
