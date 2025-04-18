# src/pixelpacker/core.py

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

import numpy as np
import tifffile
from tqdm import tqdm

from .io_utils import extract_volume, process_channel
from .stretch import calculate_limits_only, ContrastLimits
from .data_models import VolumeLayout, ChannelEntry

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# --- Config and Task Definitions (Unchanged) ---
@dataclass
class PreprocessingConfig:
    input_folder: Path
    output_folder: Path
    stretch_mode: str
    z_crop_threshold: int
    use_global_contrast: bool
    dry_run: bool
    debug: bool
    max_threads: int

@dataclass
class ProcessingTask:
    time_id: str
    channel_entry: ChannelEntry
    config: PreprocessingConfig

@dataclass
class ProcessingResult:
    time_id: str
    channel: int
    filename: str
    p_low: float
    p_high: float
    z_crop_range: List[int]

@dataclass
class LimitsPassResult:
     time_id: str
     channel: int
     limits: ContrastLimits
     cropped_volume: Optional[np.ndarray]
     z_range: Optional[Tuple[int, int]]

TimepointsDict = DefaultDict[str, List[ChannelEntry]]
TimepointResult = Dict[str, Any]


# --- Modified Layout Determination ---
def _determine_layout(first_valid_path: Path) -> Optional[VolumeLayout]:
    """Determines layout based on the shape of the ORIGINAL volume."""
    try:
        log.info(f"Determining layout from ORIGINAL shape of: {first_valid_path.name}")
        # Option 2: Read the volume but DON'T crop it here
        with tifffile.TiffFile(str(first_valid_path)) as tif:
            original_vol: np.ndarray = tif.asarray()
        original_vol = np.squeeze(original_vol)
        # Apply same reshape logic as extract_volume
        if original_vol.ndim == 5:
             # --- FIX E701: Split lines ---
             if original_vol.shape[0] == 1 and original_vol.shape[2] == 1:
                 _, z, _, y, x = original_vol.shape
                 original_vol = original_vol.reshape((z, y, x))
             # --- End FIX ---
             else:
                 raise ValueError("Unsupported 5D for layout")
        elif original_vol.ndim == 4:
             # --- FIX E701: Split lines ---
             if 1 in original_vol.shape:
                 original_vol = original_vol.reshape([s for s in original_vol.shape if s != 1])
                 if original_vol.ndim != 3:
                     raise ValueError("Cannot reshape 4D to 3D for layout")
             else:
                 # --- FIX E701: Split lines ---
                 raise ValueError("Unsupported 4D for layout")
             # --- End FIX ---
        elif original_vol.ndim == 2:
             original_vol = original_vol[np.newaxis, :, :]
        elif original_vol.ndim != 3:
             raise ValueError(f"Unsupported shape for layout: {original_vol.shape}")

        d, h, w = original_vol.shape
        if d == 0:
             raise ValueError("Original volume depth is 0. Cannot determine layout.")

        cols = math.ceil(math.sqrt(d))
        rows = math.ceil(d / cols)
        tile_w = cols * w
        tile_h = rows * h
        layout = VolumeLayout(
            width=w, height=h, depth=d,
            cols=cols, rows=rows,
            tile_width=tile_w, tile_height=tile_h
        )
        log.info(f"Layout set (based on ORIGINAL): Volume({w}x{h}x{d}), Tile({tile_w}x{tile_h}), Grid({cols}x{rows})")
        return layout
    except FileNotFoundError:
        log.error(f"Layout determination failed: File not found {first_valid_path}")
        return None
    except ValueError as e:
        log.error(f"Layout determination failed for {first_valid_path.name}: {e}")
        return None
    except Exception as e:
        log.error(f"Error processing layout from {first_valid_path.name}: {e}", exc_info=True)
        return None

# --- Modified Task Preparation (Unchanged from previous version) ---
def _prepare_tasks(
    config: PreprocessingConfig,
    timepoints_data: TimepointsDict
) -> Tuple[Optional[VolumeLayout], List[ProcessingTask]]:
    """Determines layout (based on original first file) and creates processing tasks."""
    layout: Optional[VolumeLayout] = None
    tasks_to_submit: List[ProcessingTask] = []
    sorted_time_ids = sorted(timepoints_data.keys())

    if not sorted_time_ids:
        log.warning("No timepoints found after parsing. No tasks to prepare.")
        return None, []

    log.info("🔍 Analyzing layout (based on first file's ORIGINAL shape) and preparing tasks...")

    first_file_path: Optional[Path] = None
    for time_id in sorted_time_ids:
        if timepoints_data[time_id]:
            first_file_path = timepoints_data[time_id][0]["path"]
            break

    if first_file_path is None:
        log.error("❌ No valid TIFF files found in any timepoint to determine layout.")
        return None, []

    layout = _determine_layout(first_file_path)
    if layout is None:
        log.error("❌ Could not determine volume layout from first file. Aborting.")
        return None, []

    for time_id in sorted_time_ids:
        for entry in timepoints_data[time_id]:
            task = ProcessingTask(
                time_id=time_id,
                channel_entry=entry,
                config=config,
            )
            tasks_to_submit.append(task)

    if not tasks_to_submit:
        log.error("❌ No valid processing tasks generated despite layout success. Aborting.")
        return layout, []

    log.info(f"✅ Prepared {len(tasks_to_submit)} tasks. Layout determined: {layout}")
    return layout, tasks_to_submit


# --- Helper Function _task_calculate_limits (Unchanged from previous version) ---
def _task_calculate_limits(task: ProcessingTask) -> Optional[LimitsPassResult]:
    """Pass 1: Extracts/crops volume and calculates limits."""
    try:
        cropped_volume, z_range = extract_volume(
            task.channel_entry["path"], task.config.z_crop_threshold
        )
        if cropped_volume is None or z_range is None:
             log.error(f"Failed extraction/crop for T:{task.time_id} C:{task.channel_entry['channel']}")
             return None

        if cropped_volume.size == 0:
             log.warning(f"Volume empty after crop for T:{task.time_id} C:{task.channel_entry['channel']}. Skipping limit calc.")
             return None

        limits = calculate_limits_only(cropped_volume, task.config.stretch_mode)

        return LimitsPassResult(
            time_id=task.time_id,
            channel=task.channel_entry["channel"],
            limits=limits,
            cropped_volume=cropped_volume,
            z_range=z_range
        )
    except Exception as e:
        log.error(f"❌ Unexpected error in limit calculation task T:{task.time_id} C:{task.channel_entry['channel']}: {e}", exc_info=task.config.debug)
        return None

# --- Helper Function _task_process_channel (Unchanged from previous version) ---
def _task_process_channel(
    pass1_result: LimitsPassResult,
    layout: VolumeLayout,
    config: PreprocessingConfig,
    global_limits_per_channel: Optional[Dict[int, Tuple[float, float]]] = None
) -> Optional[ProcessingResult]:
    """Wrapper to call process_channel using data from Pass 1 and global limits."""
    if pass1_result.cropped_volume is None or pass1_result.z_range is None:
         log.warning(f"Missing cropped volume or z_range for T:{pass1_result.time_id} C:{pass1_result.channel}. Skipping processing.")
         return None

    final_limits = pass1_result.limits
    if global_limits_per_channel:
        ch = pass1_result.channel
        global_tuple = global_limits_per_channel.get(ch)
        if global_tuple:
            final_limits.p_low, final_limits.p_high = global_tuple
            log.debug(f"Overriding limits for T:{pass1_result.time_id} C:{ch} with global: {global_tuple}")
        else:
            log.warning(f"Global limits requested but not found for C:{ch}. Using per-image limits.")

    result_dict = process_channel(
        time_id=pass1_result.time_id,
        ch_id=pass1_result.channel,
        cropped_vol=pass1_result.cropped_volume,
        layout=layout,
        limits=final_limits,
        z_range=pass1_result.z_range,
        stretch_mode=config.stretch_mode,
        dry_run=config.dry_run,
        debug=config.debug,
        output_folder=str(config.output_folder),
    )

    if result_dict:
         return ProcessingResult(
             time_id=result_dict["time_id"],
             channel=result_dict["channel"],
             filename=result_dict["filename"],
             p_low=result_dict["intensity_range"]["p_low"],
             p_high=result_dict["intensity_range"]["p_high"],
             z_crop_range=result_dict["z_crop_range"],
         )
    else:
        return None


# --- _scan_and_parse_files (Unchanged from previous version) ---
def _scan_and_parse_files(input_dir: Path) -> TimepointsDict:
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

# --- _setup_configuration (Unchanged from previous version) ---
def _setup_configuration(args: Dict[str, Any]) -> PreprocessingConfig:
    try:
        config = PreprocessingConfig(
            input_folder=Path(args["--input"]).resolve(),
            output_folder=Path(args["--output"]).resolve(),
            stretch_mode=args["--stretch"],
            z_crop_threshold=int(args["--z-crop-threshold"]),
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


# --- _calculate_global_limits (Unchanged from previous version) ---
def _calculate_global_limits(tasks: List[ProcessingTask], config: PreprocessingConfig) -> Tuple[Dict[int, Tuple[float, float]], List[LimitsPassResult]]:
    """Pass 1: Extracts/crops, calculates limits, aggregates for global, returns results for Pass 2."""
    log.info(" Kicking off Pass 1: Extracting volumes and calculating global contrast limits...")
    global_range_agg: DefaultDict[int, Dict[str, float]] = defaultdict(
        lambda: {"p_low": float("inf"), "p_high": float("-inf")}
    )
    num_tasks = len(tasks)
    pass1_results: List[LimitsPassResult] = []
    error_tasks = 0

    with ThreadPoolExecutor(max_workers=config.max_threads) as executor:
        futures = {executor.submit(_task_calculate_limits, task): task for task in tasks}

        with tqdm(total=num_tasks, desc=" 🔭 Pass 1/2: Calc Limits") as pbar:
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result: Optional[LimitsPassResult] = future.result()
                    if result:
                        pass1_results.append(result)
                        ch_id = result.channel
                        p_low = result.limits.p_low
                        p_high = result.limits.p_high
                        global_range_agg[ch_id]["p_low"] = min(global_range_agg[ch_id]["p_low"], p_low)
                        global_range_agg[ch_id]["p_high"] = max(global_range_agg[ch_id]["p_high"], p_high)
                    else:
                        error_tasks += 1
                except Exception as exc:
                    log.error(f"❌ Error in Pass 1 future result for T:{task.time_id} C:{task.channel_entry['channel']}: {exc}", exc_info=config.debug)
                    error_tasks += 1
                finally:
                    pbar.update(1)

    global_ranges: Dict[int, Tuple[float, float]] = {}
    for ch_id, limits_agg in global_range_agg.items():
        final_low = limits_agg["p_low"] if limits_agg["p_low"] != float("inf") else 0.0
        final_high = limits_agg["p_high"] if limits_agg["p_high"] != float("-inf") else 0.0
        # --- FIX E701: Split lines ---
        if final_high <= final_low:
             log.warning(f"Global range for C:{ch_id} collapsed or invalid [{final_low}, {final_high}]. Check input data or stretch mode '{config.stretch_mode}'. Using [0, max] or [0, 0].")
             final_high = max(final_low, final_high)
             if final_high == 0:
                 final_low = 0
        # --- End FIX ---
        global_ranges[ch_id] = (final_low, final_high)
        log.info(f"Global Range C:{ch_id}: ({global_ranges[ch_id][0]:.2f}, {global_ranges[ch_id][1]:.2f})")

    if error_tasks > 0:
         log.warning(f"Pass 1 completed with {error_tasks} errors during extraction/limit calculation.")

    log.info("✅ Pass 1 finished. Global limits determined (if applicable).")
    return global_ranges, pass1_results


# --- _execute_processing_pass (Unchanged from previous version) ---
def _execute_processing_pass(
    pass1_results: List[LimitsPassResult],
    config: PreprocessingConfig,
    layout: VolumeLayout,
    global_ranges: Optional[Dict[int, Tuple[float, float]]] = None
) -> List[ProcessingResult]:
    """Pass 2: Executes the main processing using data from pass 1 and global limits if provided."""
    pass_num = "2/2" if global_ranges else "1/1"
    log.info(f" Kicking off Pass {pass_num}: Processing channels...")
    results: List[ProcessingResult] = []
    processed_count = 0
    error_count = 0
    num_tasks = len(pass1_results)

    with ThreadPoolExecutor(max_workers=config.max_threads) as executor:
        futures = {
            executor.submit(_task_process_channel, p1_res, layout, config, global_ranges): p1_res
            for p1_res in pass1_results
        }

        with tqdm(total=num_tasks, desc=f" ⚙️ Pass {pass_num}: Processing") as pbar:
            for fut in as_completed(futures):
                p1_res = futures[fut]
                try:
                    result_obj: Optional[ProcessingResult] = fut.result()
                    if result_obj:
                        processed_count += 1
                        results.append(result_obj)
                    else:
                        log.warning(f"Task for T:{p1_res.time_id} C:{p1_res.channel} returned no result in Pass {pass_num}.")
                        error_count += 1
                except Exception as exc:
                    log.error(
                        f"❌ Error processing result for T:{p1_res.time_id} C:{p1_res.channel} in Pass {pass_num}: {exc}",
                        exc_info=config.debug
                    )
                    error_count += 1
                finally:
                    pbar.update(1)
                    if p1_res.cropped_volume is not None:
                         del p1_res.cropped_volume
                         p1_res.cropped_volume = None

    log.info(f"📊 Pass {pass_num} complete. Successful: {processed_count}, Errors/Skipped: {error_count}")
    return results


# --- Modified Metadata Finalization ---
def _finalize_metadata(
    results: List[ProcessingResult],
    layout: VolumeLayout,
    global_ranges_used: Optional[Dict[int, Tuple[float, float]]]
) -> Dict[str, Any]:
    """Aggregates results into the final metadata structure, including z_crop_range."""
    log.info("📝 Finalizing metadata...")
    metadata: Dict[str, Any] = {
        "tile_layout": {"cols": layout.cols, "rows": layout.rows},
        "volume_size": {"width": layout.width, "height": layout.height, "depth": layout.depth},
        "channels": 0,
        "timepoints": [],
        "global_intensity": {},
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
        timepoints_results[res_time]["files"][f"c{res_ch}"] = {
            "file": res.filename,
            "p_low": res.p_low,
            "p_high": res.p_high,
            "z_crop_range": res.z_crop_range,
        }

    metadata["channels"] = max_channel + 1
    processed_time_ids = sorted(timepoints_results.keys())
    metadata["timepoints"] = [
        timepoints_results[tid] for tid in processed_time_ids if timepoints_results[tid]["files"]
    ]

    final_global_intensity = {}
    if global_ranges_used:
        for ch_id, (low, high) in global_ranges_used.items():
             final_global_intensity[f"c{ch_id}"] = {"p_low": low, "p_high": high}
    else:
        # --- FIX F841: Remove unused global_range_agg ---
        # global_range_agg: DefaultDict[int, Dict[str, float]] = defaultdict(
        #     lambda: {"p_low": float("inf"), "p_high": float("-inf")}
        # )
        # --- End FIX ---
        if results:
            all_channels = set(res.channel for res in results)
            for ch in all_channels:
                 channel_results = [res for res in results if res.channel == ch]
                 if channel_results:
                     # Calculate directly from results for this channel
                     low = min(res.p_low for res in channel_results)
                     high = max(res.p_high for res in channel_results)
                     final_global_intensity[f"c{ch}"] = {"p_low": low, "p_high": high}
                 else:
                      final_global_intensity[f"c{ch}"] = {"p_low": 0.0, "p_high": 0.0}
        else:
             log.warning("No successful results to calculate retrospective global intensity.")

    metadata["global_intensity"] = final_global_intensity
    log.info("Metadata finalized.")
    return metadata

# --- _write_manifest (Unchanged from previous version) ---
def _write_manifest(metadata: Dict[str, Any], config: PreprocessingConfig):
    if config.dry_run:
        log.info("🧪 Dry run complete — manifest not written.")
        return
    if not metadata.get("timepoints"):
        log.warning("⚠️ No timepoints were successfully processed — manifest not written.")
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


# --- Main Orchestration Function (Unchanged from previous version) ---
def run_preprocessing(args: Dict[str, Any]):
    """Runs the TIFF to WebP preprocessing pipeline with optional Z-cropping."""
    start_time = time.time()
    log.info("🚀 Starting TIFF to WebP preprocessing...")

    try:
        config = _setup_configuration(args)
        timepoints_data = _scan_and_parse_files(config.input_folder)

        layout, tasks = _prepare_tasks(config, timepoints_data)
        if not layout or not tasks:
            log.error("❌ Aborting run: Could not determine layout or no tasks found.")
            return

        global_ranges: Optional[Dict[int, Tuple[float, float]]] = None
        pass1_results: List[LimitsPassResult] = []

        if config.use_global_contrast:
            try:
                global_ranges, pass1_results = _calculate_global_limits(tasks, config)
            except ValueError as e:
                 log.error(f"❌ Aborting run: {e}")
                 return
            results = _execute_processing_pass(pass1_results, config, layout, global_ranges)
        else:
            log.info("Global contrast not selected. Running single processing pass...")
            _, pass1_results = _calculate_global_limits(tasks, config)
            results = _execute_processing_pass(pass1_results, config, layout, None)

        metadata = _finalize_metadata(results, layout, global_ranges)
        _write_manifest(metadata, config)

    except (ValueError, FileNotFoundError, OSError) as e:
        log.error(f"❌ Preprocessing aborted due to critical error: {e}")
        return
    except Exception as e:
        log.critical(f"❌ An unexpected critical error occurred during orchestration: {e}", exc_info=True)
        return

    elapsed_time = time.time() - start_time
    log.info(f"🏁 Finished in {elapsed_time:.2f}s")

