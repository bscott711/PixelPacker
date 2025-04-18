# tiff_preprocessor/core.py

import json
import logging
import math
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from tqdm import tqdm

# Import necessary functions/classes
from .io_utils import extract_volume, process_channel # Keep process_channel
from .stretch import calculate_limits_only # Import the new function
from .data_models import VolumeLayout, ChannelEntry

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Types and Config classes (PreprocessingConfig, ProcessingTask, ProcessingResult, TimepointsDict, TimepointResult)
# remain largely the same as the previous version...

@dataclass
class PreprocessingConfig: # Ensure this reflects args accurately
    input_folder: Path
    output_folder: Path
    stretch_mode: str
    use_global_contrast: bool # We will use this flag
    dry_run: bool
    debug: bool
    max_threads: int

@dataclass
class ProcessingTask: # Task now used for both passes potentially
    time_id: str
    channel_entry: ChannelEntry
    config: PreprocessingConfig
    # Layout is known globally, no need to pass per task if consistent

@dataclass
class ProcessingResult:
    time_id: str
    channel: int
    filename: str
    p_low: float # This will be the GLOBAL p_low if global contrast is used
    p_high: float # This will be the GLOBAL p_high if global contrast is used

TimepointsDict = DefaultDict[str, List[ChannelEntry]]
TimepointResult = Dict[str, Any]

def _prepare_tasks(
    config: PreprocessingConfig,
    timepoints_data: TimepointsDict
) -> Tuple[Optional[VolumeLayout], List[ProcessingTask]]:
    """Determines layout and creates a list of tasks to be processed."""
    layout: Optional[VolumeLayout] = None
    tasks_to_submit: List[ProcessingTask] = []
    sorted_time_ids = sorted(timepoints_data.keys())

    if not sorted_time_ids:
        log.warning("No timepoints found after parsing. No tasks to prepare.")
        return None, []

    log.info("🔍 Analyzing layout and preparing tasks...")
    for time_id in sorted_time_ids:
        entries = timepoints_data[time_id]
        if not entries:
            log.warning(f"No valid channels found for timepoint {time_id}, skipping task prep.")
            continue

        # Determine layout from the first valid channel of the first valid timepoint
        if layout is None:
            layout = _determine_layout(entries[0]["path"])
            if layout is None:
                log.warning(f"Could not determine layout from {time_id}. Trying next timepoint.")
                continue # Skip this timepoint if layout calculation failed

        # If layout is determined, add tasks for this timepoint
        if layout:
            for entry in entries:
                task = ProcessingTask(
                    time_id=time_id,
                    channel_entry=entry,
                    config=config,
                )
                tasks_to_submit.append(task)
        else:
            # Should not be reachable if logic is correct
            log.warning(f"Layout still undefined when processing {time_id}. Skipping.")

    if layout is None:
        log.error("❌ Could not determine volume layout from any input files. Aborting task preparation.")
        return None, []

    if not tasks_to_submit:
        log.error("❌ No valid processing tasks generated. Aborting.")
        return layout, []

    log.info(f"✅ Prepared {len(tasks_to_submit)} tasks. Layout determined: {layout}")
    return layout, tasks_to_submit

# --- Helper Function for Pass 1 Task ---
def _task_calculate_limits(task: ProcessingTask) -> Optional[Tuple[str, int, float, float]]:
    """Reads image and calculates limits for Pass 1."""
    try:
        # Only extract volume and calculate limits
        vol = extract_volume(task.channel_entry["path"])
        limits = calculate_limits_only(vol, task.config.stretch_mode)
        # Return time_id, channel, p_low, p_high calculated for THIS file
        return (task.time_id, task.channel_entry["channel"], limits.p_low, limits.p_high)
    except (FileNotFoundError, ValueError) as e:
         log.error(f"❌ Error calculating limits for T:{task.time_id} C:{task.channel_entry['channel']}: {e}")
         return None
    except Exception as e:
        log.error(f"❌ Unexpected error in limit calculation task T:{task.time_id} C:{task.channel_entry['channel']}: {e}", exc_info=task.config.debug)
        return None

# --- Helper Function for Pass 2 Task (Main Processing) ---
def _task_process_channel(
    task: ProcessingTask,
    layout: VolumeLayout,
    global_ranges: Optional[Dict[int, Tuple[float, float]]] = None
) -> Optional[ProcessingResult]:
    """Wrapper to call process_channel with correct global limits for Pass 2."""
    global_limits_tuple: Optional[Tuple[float, float]] = None
    if global_ranges:
        ch = task.channel_entry["channel"]
        global_limits_tuple = global_ranges.get(ch) # Get pre-calculated global range for this channel

    # Call the existing process_channel function from io_utils
    result_dict = process_channel(
        time_id=task.time_id,
        ch_id=task.channel_entry["channel"],
        tiff_path=str(task.channel_entry["path"]),
        layout=layout,
        stretch_mode=task.config.stretch_mode,
        dry_run=task.config.dry_run,
        debug=task.config.debug,
        output_folder=str(task.config.output_folder),
        global_limits_tuple=global_limits_tuple # Pass the correct global limits
    )

    if result_dict:
        # Construct ProcessingResult. Note: p_low/p_high in result_dict
        # should now be the global ones if global_limits_tuple was used.
         return ProcessingResult(
             time_id=result_dict.get("time_id", task.time_id),
             channel=result_dict["channel"],
             filename=result_dict["filename"],
             p_low=result_dict["intensity_range"]["p_low"],
             p_high=result_dict["intensity_range"]["p_high"],
         )
    else:
        return None


# --- Core Logic Functions (_scan_and_parse_files, _determine_layout, _setup_configuration remain same) ---
def _scan_and_parse_files(input_dir: Path) -> TimepointsDict:
    # (Implementation unchanged from previous step)
    # ...
    timepoints_data: TimepointsDict = defaultdict(list)
    tiff_regex = re.compile(r".*_ch(\d+)_stack(\d{4}).*?\.tif(?:f)?$", re.IGNORECASE)
    log.info(f"Scanning for TIFF files in: {input_dir}")
    found_files = list(input_dir.glob("*.tif*"))
    if not found_files:
        log.warning("No TIFF files found in the input directory.")
        return timepoints_data
    log.info(f"Found {len(found_files)} potential TIFF files. Parsing filenames...")
    parsed_count = 0
    skip_count = 0
    for path in found_files:
        match = tiff_regex.match(path.name)
        if match:
            try:
                ch_id = int(match.group(1))
                time_id = f"stack{match.group(2)}"
                timepoints_data[time_id].append({"channel": ch_id, "path": path})
                parsed_count += 1
            except (ValueError, IndexError) as e:
                log.warning(f"Skipping {path.name} — Error parsing captured groups: {e}")
                skip_count += 1
            except Exception as e:
                log.warning(f"Skipping {path.name} — Unexpected error during parsing: {e}")
                skip_count += 1
    log.info(f"Parsed {parsed_count} files successfully.")
    if skip_count > 0:
        log.warning(f"Skipped {skip_count} files due to naming/parsing issues.")
    for time_id in timepoints_data:
        timepoints_data[time_id].sort(key=lambda x: x["channel"])
    return timepoints_data

def _determine_layout(first_valid_path: Path) -> Optional[VolumeLayout]:
    # (Implementation unchanged from previous step)
    # ...
    try:
        log.info(f"Determining layout from: {first_valid_path.name}")
        vol = extract_volume(first_valid_path)
        d, h, w = vol.shape
        cols = math.ceil(math.sqrt(d))
        rows = math.ceil(d / cols)
        tile_w = cols * w
        tile_h = rows * h
        layout = VolumeLayout(
            width=w, height=h, depth=d,
            cols=cols, rows=rows,
            tile_width=tile_w, tile_height=tile_h
        )
        log.info(f"Layout set: Volume({w}x{h}x{d}), Tile({tile_w}x{tile_h}), Grid({cols}x{rows})")
        return layout
    except FileNotFoundError:
        log.error(f"Layout determination failed: File not found {first_valid_path}")
        return None
    except ValueError as e:
        log.error(f"Layout determination failed: Shape error in {first_valid_path.name}: {e}")
        return None
    except Exception as e:
        log.error(f"Error processing layout from {first_valid_path.name}: {e}")
        return None

def _setup_configuration(args: Dict[str, Any]) -> PreprocessingConfig:
    # (Implementation unchanged from previous step)
    # ...
    try:
        config = PreprocessingConfig(
            input_folder=Path(args["--input"]).resolve(),
            output_folder=Path(args["--output"]).resolve(),
            stretch_mode=args["--stretch"],
            use_global_contrast=args["--global-contrast"],
            dry_run=args["--dry-run"],
            debug=args["--debug"],
            max_threads=int(args["--threads"]),
        )
        log.info("Configuration loaded.")
        if config.dry_run:
            log.info("💡 Dry-run mode enabled.")
        config.output_folder.mkdir(parents=True, exist_ok=True)
        return config
    except KeyError as e:
        log.error(f"Missing expected argument: {e}")
        raise ValueError(f"Configuration error: Missing argument {e}") from e
    except ValueError as e:
        log.error(f"Invalid argument value: {e}")
        raise ValueError(f"Configuration error: Invalid value {e}") from e
    except OSError as e:
        log.error(f"❌ Could not create output directory {args['--output']}: {e}")
        raise


# --- Pass 1 Function ---
def _calculate_global_limits(tasks: List[ProcessingTask], config: PreprocessingConfig) -> Dict[int, Tuple[float, float]]:
    """Pass 1: Calculates global min/max contrast limits across all tasks."""
    log.info(" Kicking off Pass 1: Calculating global contrast limits...")
    global_range_agg: DefaultDict[int, Dict[str, float]] = defaultdict(
        lambda: {"p_low": float("inf"), "p_high": float("-inf")}
    )
    num_tasks = len(tasks)
    completed_tasks = 0
    error_tasks = 0

    with ThreadPoolExecutor(max_workers=config.max_threads) as executor:
        # Submit tasks to calculate limits only
        futures = {executor.submit(_task_calculate_limits, task): task for task in tasks}

        with tqdm(total=num_tasks, desc=" 🔭 Pass 1/2: Calc Limits") as pbar:
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result() # Tuple: (time_id, ch_id, p_low, p_high) or None
                    if result:
                        _, ch_id, p_low, p_high = result
                        global_range_agg[ch_id]["p_low"] = min(global_range_agg[ch_id]["p_low"], p_low)
                        global_range_agg[ch_id]["p_high"] = max(global_range_agg[ch_id]["p_high"], p_high)
                        completed_tasks += 1
                    else:
                        error_tasks += 1
                except Exception as exc:
                    log.error(f"❌ Error in Pass 1 future result for T:{task.time_id} C:{task.channel_entry['channel']}: {exc}", exc_info=config.debug)
                    error_tasks += 1
                finally:
                    pbar.update(1)

    # Convert aggregated ranges to the final structure
    global_ranges: Dict[int, Tuple[float, float]] = {}
    for ch_id, limits in global_range_agg.items():
        final_low = limits["p_low"] if limits["p_low"] != float("inf") else 0.0
        final_high = limits["p_high"] if limits["p_high"] != float("-inf") else 0.0
        if final_high <= final_low: # Handle potential collapse
             log.warning(f"Global range for C:{ch_id} collapsed or invalid [{final_low}, {final_high}]. Check input data or stretch mode '{config.stretch_mode}'. Using [0, max] or [0, 0].")
             # Decide on a safe fallback, e.g., use only max or zero if max is also zero
             final_high = max(final_low, final_high) # Ensure high >= low
             if final_high == 0:
                 final_low = 0

        global_ranges[ch_id] = (final_low, final_high)
        log.info(f"Global Range C:{ch_id}: ({global_ranges[ch_id][0]:.2f}, {global_ranges[ch_id][1]:.2f})")

    if error_tasks > 0:
         log.warning(f"Pass 1 completed with {error_tasks} errors during limit calculation.")
    if not global_ranges:
        log.error("❌ Pass 1 failed to determine any global ranges.")
        raise ValueError("Global limit calculation failed.")

    log.info("✅ Pass 1 finished. Global limits determined.")
    return global_ranges


# --- Pass 2 Function ---
def _execute_processing_pass(
    tasks: List[ProcessingTask],
    config: PreprocessingConfig,
    layout: VolumeLayout,
    global_ranges: Optional[Dict[int, Tuple[float, float]]] = None # Pass 1 results
) -> List[ProcessingResult]:
    """Pass 2: Executes the main processing using determined global limits if provided."""
    pass_num = "2/2" if global_ranges else "1/1" # Adjust progress bar label
    log.info(f" Kicking off Pass {pass_num}: Processing channels...")
    results: List[ProcessingResult] = []
    processed_count = 0
    error_count = 0
    num_tasks = len(tasks)

    with ThreadPoolExecutor(max_workers=config.max_threads) as executor:
        # Submit main processing tasks, passing global ranges
        futures = {
            executor.submit(_task_process_channel, task, layout, global_ranges): task
            for task in tasks
        }

        with tqdm(total=num_tasks, desc=f" ⚙️ Pass {pass_num}: Processing") as pbar:
            for fut in as_completed(futures):
                task = futures[fut]
                try:
                    result_obj: Optional[ProcessingResult] = fut.result()
                    if result_obj:
                        processed_count += 1
                        results.append(result_obj)
                    else:
                        log.warning(f"Task for T:{task.time_id} C:{task.channel_entry['channel']} returned no result in Pass {pass_num}.")
                        error_count += 1
                except Exception as exc:
                    log.error(
                        f"❌ Error processing result for T:{task.time_id} C:{task.channel_entry['channel']} in Pass {pass_num}: {exc}",
                        exc_info=config.debug
                    )
                    error_count += 1
                finally:
                    pbar.update(1)

    log.info(f"📊 Pass {pass_num} complete. Successful: {processed_count}, Errors/Skipped: {error_count}")
    return results

# --- Metadata and Manifest Functions (_finalize_metadata, _write_manifest remain similar) ---
def _finalize_metadata(
    results: List[ProcessingResult],
    layout: VolumeLayout,
    global_ranges_used: Optional[Dict[int, Tuple[float, float]]] # Pass in the ranges used
) -> Dict[str, Any]:
    """Aggregates results into the final metadata structure."""
    log.info("📝 Finalizing metadata...")
    metadata: Dict[str, Any] = {
        "tile_layout": {"cols": layout.cols, "rows": layout.rows},
        "volume_size": {"width": layout.width, "height": layout.height, "depth": layout.depth},
        "channels": 0,
        "timepoints": [],
        "global_intensity": {}, # Will populate or confirm from global_ranges_used
    }
    timepoints_results: DefaultDict[str, TimepointResult] = defaultdict(
        lambda: {"time": None, "files": {}}
    )

    max_channel = -1
    for res in results:
        res_time = res.time_id
        res_ch = res.channel
        max_channel = max(max_channel, res_ch)

        timepoints_results[res_time]["time"] = res_time
        # The p_low/p_high here now accurately reflect what was used
        timepoints_results[res_time]["files"][f"c{res_ch}"] = {
            "file": res.filename,
            "p_low": res.p_low,
            "p_high": res.p_high,
        }

    metadata["channels"] = max_channel + 1
    processed_time_ids = sorted(timepoints_results.keys())
    metadata["timepoints"] = [
        timepoints_results[tid] for tid in processed_time_ids if timepoints_results[tid]["files"]
    ]

    # Populate global_intensity section - confirm from passed ranges if available
    final_global_intensity = {}
    if global_ranges_used:
        for ch_id, (low, high) in global_ranges_used.items():
             final_global_intensity[f"c{ch_id}"] = {"p_low": low, "p_high": high}
    else:
        # If not using global contrast, calculate it retrospectively as before
        global_range_agg: DefaultDict[int, Dict[str, float]] = defaultdict(
            lambda: {"p_low": float("inf"), "p_high": float("-inf")}
        )
        for res in results:
            ch = res.channel
            global_range_agg[ch]["p_low"] = min(global_range_agg[ch]["p_low"], res.p_low)
            global_range_agg[ch]["p_high"] = max(global_range_agg[ch]["p_high"], res.p_high)
        for ch in range(metadata["channels"]):
             low = global_range_agg[ch]["p_low"]
             high = global_range_agg[ch]["p_high"]
             final_global_intensity[f"c{ch}"] = {
                 "p_low": low if low != float("inf") else 0.0,
                 "p_high": high if high != float("-inf") else 0.0,
            }

    metadata["global_intensity"] = final_global_intensity
    log.info("Metadata finalized.")
    return metadata

def _write_manifest(metadata: Dict[str, Any], config: PreprocessingConfig):
    # (Implementation unchanged)
    # ...
    if config.dry_run:
        log.info("🧪 Dry run complete — manifest not written.")
        return
    if not metadata["timepoints"]:
        log.info("⚠️ No timepoints were successfully processed — manifest not written.")
        return
    manifest_path = config.output_folder / "manifest.json"
    log.info(f"📄 Writing metadata to {manifest_path}...")
    try:
        with open(manifest_path, "w") as f:
            json.dump(metadata, f, indent=2)
        log.info("Manifest saved successfully.")
    except IOError as e:
        log.error(f"❌ Failed to write manifest file {manifest_path}: {e}")
    except TypeError as e:
        log.error(f"❌ Failed to serialize metadata to JSON: {e}")


# --- Main Orchestration Function ---
def run_preprocessing(args: Dict[str, Any]):
    """Runs the TIFF to WebP preprocessing pipeline, potentially using a two-pass approach for global contrast."""
    start_time = time.time()
    log.info("🚀 Starting TIFF to WebP preprocessing...")

    try:
        config = _setup_configuration(args)
        timepoints_data = _scan_and_parse_files(config.input_folder)

        # Determine layout and prepare initial task list
        layout, tasks = _prepare_tasks(config, timepoints_data)
        if not layout or not tasks:
            log.error("❌ Aborting run: Could not determine layout or no tasks found.")
            return

        global_ranges: Optional[Dict[int, Tuple[float, float]]] = None
        # ****** TWO-PASS LOGIC ******
        if config.use_global_contrast:
            # --- Pass 1: Calculate Global Limits ---
            try:
                global_ranges = _calculate_global_limits(tasks, config)
            except ValueError as e: # Catch failure from limit calculation
                 log.error(f"❌ Aborting run: {e}")
                 return
            # --- Pass 2: Process with Global Limits ---
            results = _execute_processing_pass(tasks, config, layout, global_ranges)
        else:
            # --- Single Pass: Process Individually ---
            log.info("Global contrast not selected. Running single processing pass...")
            results = _execute_processing_pass(tasks, config, layout, None) # Pass None for global_ranges

        # --- Finalize and Save ---
        metadata = _finalize_metadata(results, layout, global_ranges) # Pass global ranges used
        _write_manifest(metadata, config)

    except (ValueError, FileNotFoundError, OSError) as e:
        log.error(f"❌ Preprocessing aborted due to critical error: {e}")
        return
    except Exception as e:
        log.critical(f"❌ An unexpected critical error occurred during orchestration: {e}", exc_info=True)
        return

    elapsed_time = time.time() - start_time
    log.info(f"🏁 Finished in {elapsed_time:.2f}s")