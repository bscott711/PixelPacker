# src/pixelpacker/core.py

import json
import logging
import math
import re  # Keep, used in _scan_and_parse_files
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import numpy as np

# import tifffile # Removed unused import
from tqdm import tqdm

# Import necessary functions/classes
from .io_utils import extract_original_volume
from .io_utils import find_z_crop_range  # Use the wrapper function
from .io_utils import process_channel
from .stretch import ContrastLimits, calculate_limits_only
from .data_models import ChannelEntry, VolumeLayout

# --- Setup & Configuration ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# --- Dataclasses ---
@dataclass
class PreprocessingConfig:
    """Configuration settings for the preprocessing pipeline."""

    input_folder: Path
    output_folder: Path
    stretch_mode: str
    z_crop_method: str
    z_crop_threshold: int
    use_global_contrast: bool
    dry_run: bool
    debug: bool
    max_threads: int


@dataclass
class ProcessingTask:
    """Represents a single file to be processed."""

    time_id: str
    channel_entry: ChannelEntry
    config: PreprocessingConfig


@dataclass
class ProcessingResult:
    """Result metadata for a single successfully processed file."""

    time_id: str
    channel: int
    filename: str
    p_low: float
    p_high: float


@dataclass
class CropRangeInfo:
    """Holds results from Pass 0 Z-range finding task."""

    path: Path
    z_start: int
    z_end: int
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass
class LimitsPassResult:
    """Holds results from Pass 1 limit calculation task."""

    time_id: str
    channel: int
    limits: ContrastLimits
    globally_cropped_volume: Optional[np.ndarray]
    global_z_range: Tuple[int, int]


TimepointsDict = DefaultDict[str, List[ChannelEntry]]
TimepointResult = Dict[str, Any]

# === Pass 0: Determine Global Z-Crop Range ===


def _task_find_local_z_range(task: ProcessingTask) -> Optional[CropRangeInfo]:
    """Pass 0 Task: Extracts original volume, finds local Z range using selected method, and dimensions."""

    ch_entry = task.channel_entry
    file_path = ch_entry["path"]

    log.debug(
        f"Pass 0 task started for T:{task.time_id} C:{ch_entry['channel']} ({file_path.name})"
    )
    original_volume = None
    try:
        original_volume = extract_original_volume(file_path)
        if original_volume is None:
            return None

        z_start, z_end = find_z_crop_range(
            volume=original_volume,
            method=task.config.z_crop_method,
            threshold=task.config.z_crop_threshold,
            debug=task.config.debug,
            output_folder=task.config.output_folder,
            filename_prefix=f"T{task.time_id}_C{ch_entry['channel']}",
        )

        # --- Ruff Fix: E701 ---
        if original_volume.ndim != 3:
            log.warning(f"Pass 0 - Vol shape {original_volume.shape} not 3D.")
            return None

        _, h, w = original_volume.shape

        log.debug(
            f"Pass 0 - Found local range [{z_start}-{z_end}], W={w}, H={h} for T:{task.time_id} C:{ch_entry['channel']}"
        )
        return CropRangeInfo(
            path=file_path, z_start=z_start, z_end=z_end, width=w, height=h
        )
    except Exception as e:
        log.error(
            f"❌ Pass 0 - Unexpected error T:{task.time_id} C:{ch_entry['channel']}: {e}",
            exc_info=task.config.debug,
        )
        return None
    finally:
        # --- Ruff Fix: E701 ---
        if original_volume is not None:
            del original_volume


def _determine_global_z_crop_and_dims(
    tasks: List[ProcessingTask], config: PreprocessingConfig
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """Pass 0 Orchestrator: Finds global Z-crop range and base W/H."""
    log.info(" Kicking off Pass 0: Determining global Z-crop range & dimensions...")
    num_tasks = len(tasks)

    min_z_start = float("inf")
    max_z_end = float("-inf")

    base_width: Optional[int] = None
    base_height: Optional[int] = None

    error_tasks = 0
    processed_tasks = 0

    with ThreadPoolExecutor(
        max_workers=config.max_threads, thread_name_prefix="Pass0_ZRange"
    ) as executor:
        futures = {
            executor.submit(_task_find_local_z_range, task): task for task in tasks
        }
        with tqdm(total=num_tasks, desc=" 🔪 Pass 0/3: Find Z Range & Dims") as pbar:
            for future in as_completed(futures):
                task = futures[future]
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

                        # --- Ruff Fix: E701 ---
                        if base_width is None:
                            base_width = result.width
                            base_height = result.height
                            log.info(
                                f"Pass 0 - Base dims W={base_width}, H={base_height} from {result.path.name}"
                            )

                    elif result:
                        log.warning(
                            f"Pass 0 - Task for {task.channel_entry['path'].name} missing W/H."
                        )
                        error_tasks += 1

                    else:
                        error_tasks += 1
                except Exception as exc:
                    log.error(
                        f"❌ Pass 0 - Error T:{task.time_id} C:{task.channel_entry['channel']}: {exc}",
                        exc_info=config.debug,
                    )
                    error_tasks += 1

                finally:
                    pbar.update(1)

    # --- Ruff Fix: E701 ---
    if error_tasks > 0:
        log.warning(f"Pass 0 completed with {error_tasks} errors.")

    # --- Ruff Fix: E701 ---
    if processed_tasks == 0:
        log.error("❌ Pass 0 failed: No Z-range/dims determined.")
        return None, None

    # --- Ruff Fix: E701 ---
    if base_width is None or base_height is None:
        log.error("❌ Pass 0 failed: No base W/H determined.")
        return None, None

    # --- Ruff Fix: E701 ---
    if min_z_start == float("inf") or max_z_end == float("-inf"):
        log.error(
            f"❌ Pass 0 failed: Invalid Z-range (inf values) [{min_z_start}, {max_z_end}]."
        )
        return None, (base_width, base_height)

    global_z_start = int(min_z_start)
    global_z_end = int(max_z_end)

    # --- Ruff Fix: E701 ---
    if global_z_start > global_z_end:
        log.error(
            f"❌ Pass 0 failed: Final Z-range invalid [{global_z_start}, {global_z_end}]."
        )
        return None, (base_width, base_height)

    log.info(
        f"✅ Pass 0 finished. Global Z-crop range: [{global_z_start}, {global_z_end}]. Base Dims: W={base_width}, H={base_height}"
    )
    return (global_z_start, global_z_end), (base_width, base_height)


# === Layout Determination ===
def _determine_layout(
    base_width: int, base_height: int, global_z_range: Tuple[int, int]
) -> Optional[VolumeLayout]:
    """Determines tile layout based on base W/H and globally cropped depth."""
    try:
        global_z_start, global_z_end = global_z_range
        d = global_z_end - global_z_start + 1
        # --- Ruff Fix: E701 ---
        if d <= 0:
            log.error(f"Layout failed: Global depth <= 0 ({d})")
            return None

        # --- Ruff Fix: E701 ---
        if base_width <= 0 or base_height <= 0:
            log.error(
                f"Layout failed: Invalid base dims W={base_width}, H={base_height}."
            )
            return None

        log.info(
            f"Determining layout for W={base_width}, H={base_height}, Global Depth={d}"
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
            f"Layout determined: Volume({base_width}x{base_height}x{d}), Tile({tile_w}x{tile_h}), Grid({cols}x{rows})"
        )
        return layout
    except Exception as e:
        log.error(f"❌ Layout determination error: {e}", exc_info=True)
        return None


# === Task Preparation ===
def _prepare_tasks_and_layout(
    config: PreprocessingConfig, timepoints_data: TimepointsDict
) -> Tuple[Optional[Tuple[int, int]], Optional[VolumeLayout], List[ProcessingTask]]:
    """Orchestrates Pass 0, determines layout, returns final tasks."""
    layout: Optional[VolumeLayout] = None
    global_z_range: Optional[Tuple[int, int]] = None
    base_dims: Optional[Tuple[int, int]] = None
    tasks_to_submit: List[ProcessingTask] = []
    sorted_time_ids = sorted(timepoints_data.keys())

    # --- Ruff Fix: E701 ---
    if not sorted_time_ids:
        log.warning("No timepoints found.")
        return None, None, []

    for time_id in sorted_time_ids:
        for entry in timepoints_data[time_id]:
            tasks_to_submit.append(ProcessingTask(time_id, entry, config))

    # --- Ruff Fix: E701 ---
    if not tasks_to_submit:
        log.error("❌ No tasks created.")
        return None, None, []

    try:
        global_z_range, base_dims = _determine_global_z_crop_and_dims(
            tasks_to_submit, config
        )
    except Exception as e:
        log.error(f"❌ Critical error during Pass 0: {e}", exc_info=config.debug)
        return None, None, []

    # --- Ruff Fix: E701 ---
    if global_z_range is None or base_dims is None:
        log.error("Aborting: Failed Pass 0.")
        return global_z_range, None, []

    layout = _determine_layout(base_dims[0], base_dims[1], global_z_range)
    # --- Ruff Fix: E701 ---
    if layout is None:
        log.error("❌ Aborting: Failed layout determination.")
        return global_z_range, None, []

    log.info(
        f"✅ Prepared {len(tasks_to_submit)} tasks. Layout determined. Global Z Range: {global_z_range}"
    )
    return global_z_range, layout, tasks_to_submit


# === Pass 1: Calculate Contrast Limits ===
def _task_calculate_limits(
    task: ProcessingTask, global_z_range: Tuple[int, int]
) -> Optional[LimitsPassResult]:
    """Pass 1 Task: Extracts original, applies GLOBAL crop, calculates limits, KEEPS cropped volume."""

    ch_entry = task.channel_entry
    file_path = ch_entry["path"]

    log.debug(f"Pass 1 task started for T:{task.time_id} C:{ch_entry['channel']}")
    original_volume = None
    globally_cropped_volume = None
    try:
        original_volume = extract_original_volume(file_path)
        if original_volume is None:
            return None

        z_start, z_end = global_z_range
        original_depth = original_volume.shape[0]

        clamped_z_start = max(0, z_start)
        clamped_z_end = min(original_depth - 1, z_end)

        # --- Ruff Fix: E701 ---
        if (
            clamped_z_start > clamped_z_end
            or clamped_z_start >= original_depth
            or clamped_z_end < 0
        ):
            log.warning(
                f"Pass 1 - Z range invalid T:{task.time_id} C:{ch_entry['channel']}. Skipping."
            )
            return None

        globally_cropped_volume = original_volume[
            clamped_z_start : clamped_z_end + 1, :, :
        ]

        del original_volume
        original_volume = None  # Delete now

        # --- Ruff Fix: E701 ---
        if globally_cropped_volume.size == 0:
            log.warning(
                f"Pass 1 - Empty volume T:{task.time_id} C:{ch_entry['channel']}. Skip."
            )
            return None

        limits = calculate_limits_only(
            globally_cropped_volume, task.config.stretch_mode
        )
        log.debug(f"Pass 1 - Limits {limits} T:{task.time_id} C:{ch_entry['channel']}")
        return LimitsPassResult(
            time_id=task.time_id,
            channel=ch_entry["channel"],
            limits=limits,
            globally_cropped_volume=globally_cropped_volume,
            global_z_range=global_z_range,
        )
    except Exception as e:
        log.error(
            f"❌ Pass 1 - Error T:{task.time_id} C:{ch_entry['channel']}: {e}",
            exc_info=task.config.debug,
        )
        return None
    finally:
        # --- Ruff Fix: E701 ---
        if original_volume is not None:
            del original_volume


def _calculate_global_limits(
    tasks: List[ProcessingTask],
    config: PreprocessingConfig,
    global_z_range: Tuple[int, int],
) -> Tuple[Dict[int, Tuple[float, float]], List[LimitsPassResult]]:
    """Pass 1 Orchestrator: Calculates limits and returns results including cropped volumes."""
    log.info(
        " Kicking off Pass 1: Applying global crop & calculating contrast limits..."
    )
    global_range_agg: DefaultDict[int, Dict[str, float]] = defaultdict(
        lambda: {"p_low": float("inf"), "p_high": float("-inf")}
    )
    num_tasks = len(tasks)

    pass1_individual_results: List[LimitsPassResult] = []
    error_tasks = 0

    with ThreadPoolExecutor(
        max_workers=config.max_threads, thread_name_prefix="Pass1_Limits"
    ) as executor:
        futures = {
            executor.submit(_task_calculate_limits, task, global_z_range): task
            for task in tasks
        }
        with tqdm(total=num_tasks, desc=" 🔭 Pass 1/3: Calc Limits") as pbar:
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result: Optional[LimitsPassResult] = future.result()
                    if result:
                        pass1_individual_results.append(result)
                        ch_id = result.channel
                        global_range_agg[ch_id]["p_low"] = min(
                            global_range_agg[ch_id]["p_low"], result.limits.p_low
                        )
                        global_range_agg[ch_id]["p_high"] = max(
                            global_range_agg[ch_id]["p_high"], result.limits.p_high
                        )
                    else:
                        error_tasks += 1
                except Exception as exc:
                    log.error(
                        f"❌ Pass 1 - Error T:{task.time_id} C:{task.channel_entry['channel']}: {exc}",
                        exc_info=config.debug,
                    )
                    error_tasks += 1

                finally:
                    pbar.update(1)
    global_ranges_final: Dict[int, Tuple[float, float]] = {}
    for ch_id, limits_agg in global_range_agg.items():
        final_low = limits_agg["p_low"] if limits_agg["p_low"] != float("inf") else 0.0
        final_high = (
            limits_agg["p_high"] if limits_agg["p_high"] != float("-inf") else 0.0
        )
        if final_high <= final_low:
            log.warning(
                f"Aggregated global range C:{ch_id} collapsed [{final_low}, {final_high}]. Adjusting."
            )
            final_high = max(final_high, final_low + 1e-6)
            # --- Ruff Fix: E701 ---
            if final_low == 0 and final_high == 1e-6:
                log.warning(
                    f"Adjusted C:{ch_id} range tiny [{final_low}, {final_high:.2g}]"
                )

        global_ranges_final[ch_id] = (final_low, final_high)
        log.info(
            f"Aggregated Global Range C:{ch_id}: ({global_ranges_final[ch_id][0]:.4f}, {global_ranges_final[ch_id][1]:.4f})"
        )
    # --- Ruff Fix: E701 ---
    if error_tasks > 0:
        log.warning(f"Pass 1 completed with {error_tasks} errors.")

    # --- Ruff Fix: E701 ---
    if not pass1_individual_results:
        raise ValueError("Pass 1 failed: No limits calculated.")

    log.info(
        f"✅ Pass 1 finished. Calculated limits for {len(pass1_individual_results)} tasks."
    )
    return global_ranges_final, pass1_individual_results


# === Pass 2: Process Channels (Tile & Save) ===
def _task_process_channel(
    pass1_result: LimitsPassResult,
    layout: VolumeLayout,
    config: PreprocessingConfig,
    global_contrast_override: Optional[Tuple[float, float]] = None,
) -> Optional[ProcessingResult]:
    """Pass 2 Task: Uses globally cropped volume from Pass 1, applies contrast, tiles, saves."""
    log.debug(
        f"Pass 2 task started for T:{pass1_result.time_id} C:{pass1_result.channel}"
    )
    globally_cropped_volume = None
    try:
        globally_cropped_volume = pass1_result.globally_cropped_volume
        # --- Ruff Fix: E701 ---
        if globally_cropped_volume is None:
            log.warning(
                f"Pass 2 - Missing cropped volume T:{pass1_result.time_id} C:{pass1_result.channel}. Skipping."
            )
            return None

        final_limits = pass1_result.limits
        if global_contrast_override:
            log.debug(
                f"Pass 2 - Overriding T:{pass1_result.time_id} C:{pass1_result.channel} with global: {global_contrast_override}"
            )
            final_limits = ContrastLimits(
                p_low=global_contrast_override[0], p_high=global_contrast_override[1]
            )

            final_limits.actual_min = pass1_result.limits.actual_min
            final_limits.actual_max = pass1_result.limits.actual_max

        result_dict = process_channel(
            time_id=pass1_result.time_id,
            ch_id=pass1_result.channel,
            globally_cropped_vol=globally_cropped_volume,
            layout=layout,
            limits=final_limits,
            stretch_mode=config.stretch_mode,
            dry_run=config.dry_run,
            debug=config.debug,
            output_folder=str(config.output_folder),
        )
        # --- Ruff Fix: E701 ---
        if result_dict:
            log.debug(
                f"Pass 2 - Success T:{pass1_result.time_id} C:{pass1_result.channel}"
            )
            return ProcessingResult(
                time_id=result_dict["time_id"],
                channel=result_dict["channel"],
                filename=result_dict["filename"],
                p_low=result_dict["intensity_range"]["p_low"],
                p_high=result_dict["intensity_range"]["p_high"],
            )

        else:
            # --- Ruff Fix: E701 ---
            log.warning(
                f"Pass 2 - Processing function failed T:{pass1_result.time_id} C:{pass1_result.channel}."
            )
            return None

    except Exception as e:
        log.error(
            f"❌ Pass 2 - Unexpected error T:{pass1_result.time_id} C:{pass1_result.channel}: {e}",
            exc_info=config.debug,
        )
        return None


def _execute_processing_pass(
    pass1_results: List[LimitsPassResult],
    config: PreprocessingConfig,
    layout: VolumeLayout,
    global_contrast_ranges: Optional[Dict[int, Tuple[float, float]]] = None,
) -> List[ProcessingResult]:
    """Pass 2 Orchestrator: Executes main processing using data from Pass 1."""
    pass_desc = "Pass 2/3"
    log.info(f" Kicking off {pass_desc}: Processing channels...")
    results: List[ProcessingResult] = []

    processed_count = 0
    error_count = 0

    tasks_submitted = len(pass1_results)

    with ThreadPoolExecutor(
        max_workers=config.max_threads, thread_name_prefix="Pass2_Process"
    ) as executor:
        futures = {}
        for p1_res in pass1_results:
            global_override = None
            if config.use_global_contrast and global_contrast_ranges:
                global_override = global_contrast_ranges.get(p1_res.channel)
                # --- Ruff Fix: E701 ---
                if not global_override:
                    log.warning(
                        f"{pass_desc} - Global contrast requested but not found C:{p1_res.channel} T:{p1_res.time_id}. Using per-image."
                    )

            fut = executor.submit(
                _task_process_channel, p1_res, layout, config, global_override
            )
            futures[fut] = p1_res
        log.info(f"{pass_desc} - Submitted {tasks_submitted} tasks.")
        with tqdm(total=tasks_submitted, desc=f" ⚙️ {pass_desc}: Processing") as pbar:
            for fut in as_completed(futures):
                p1_res = futures[fut]
                try:
                    result_obj: Optional[ProcessingResult] = fut.result()
                    # --- Ruff Fix: E701 ---
                    if result_obj:
                        processed_count += 1
                        results.append(result_obj)

                    else:
                        # --- Ruff Fix: E701 ---
                        log.warning(
                            f"{pass_desc} - Task T:{p1_res.time_id} C:{p1_res.channel} returned no result."
                        )
                        error_count += 1

                except Exception as exc:
                    log.error(
                        f"❌ {pass_desc} - Error T:{p1_res.time_id} C:{p1_res.channel}: {exc}",
                        exc_info=config.debug,
                    )
                    error_count += 1

                finally:
                    pbar.update(1)
                    # Memory Cleanup
                    # --- Ruff Fix: E701 ---
                    if p1_res.globally_cropped_volume is not None:
                        del p1_res.globally_cropped_volume
                        p1_res.globally_cropped_volume = None

    log.info(
        f"📊 {pass_desc} complete. Successful: {processed_count}, Errors/Skipped: {error_count}"
    )
    return results


# === Metadata Finalization & Manifest Writing ===
def _finalize_metadata(
    results: List[ProcessingResult],
    layout: VolumeLayout,
    global_z_range: Tuple[int, int],
    global_contrast_ranges_used: Optional[Dict[int, Tuple[float, float]]],
) -> Dict[str, Any]:
    """Aggregates results into the final metadata structure including global Z range."""
    log.info("📝 Finalizing metadata...")
    metadata: Dict[str, Any] = {
        "tile_layout": {"cols": layout.cols, "rows": layout.rows},
        "volume_size": {
            "width": layout.width,
            "height": layout.height,
            "depth": layout.depth,
        },
        "channels": 0,
        "global_z_crop_range": list(global_z_range),
        "timepoints": [],
        "global_intensity": {},
    }
    timepoints_results: DefaultDict[str, TimepointResult] = defaultdict(
        lambda: {"time": None, "files": {}}
    )
    max_channel = -1
    # --- Ruff Fix: E701 ---
    if not results:
        log.warning("No results to finalize.")
        return metadata

    for res in results:
        res_time = res.time_id
        res_ch = res.channel
        max_channel = max(max_channel, res_ch)

        timepoints_results[res_time]["time"] = res_time
        timepoints_results[res_time]["files"][f"c{res_ch}"] = {
            "file": res.filename,
            "p_low": res.p_low,
            "p_high": res.p_high,
        }
    metadata["channels"] = max_channel + 1
    processed_time_ids = sorted(timepoints_results.keys())
    metadata["timepoints"] = [
        timepoints_results[tid]
        for tid in processed_time_ids
        if timepoints_results[tid]["files"]
    ]

    final_global_intensity = {}
    if global_contrast_ranges_used:
        log.info("Populating global_intensity from pre-calculated ranges.")
        for ch_id, (low, high) in global_contrast_ranges_used.items():
            final_global_intensity[f"c{ch_id}"] = {"p_low": low, "p_high": high}
    else:
        log.info("Calculating retrospective global_intensity.")
        all_channels = set(res.channel for res in results)
        for ch in all_channels:
            channel_results = [res for res in results if res.channel == ch]
            # --- Ruff Fix: E701 ---
            if channel_results:
                low = min(res.p_low for res in channel_results)
                high = max(res.p_high for res in channel_results)
                final_global_intensity[f"c{ch}"] = {"p_low": low, "p_high": high}

            else:
                final_global_intensity[f"c{ch}"] = {"p_low": 0.0, "p_high": 0.0}
    metadata["global_intensity"] = final_global_intensity
    log.info("Metadata finalized.")
    return metadata


def _write_manifest(metadata: Dict[str, Any], config: PreprocessingConfig):
    """Writes the finalized metadata to manifest.json."""
    # --- Ruff Fix: E701 ---
    if config.dry_run:
        log.info("--dry-run enabled, manifest not written.")
        return

    # --- Ruff Fix: E701 ---
    if not metadata.get("timepoints"):
        log.warning("No timepoints in metadata, manifest not written.")
        return

    manifest_path = config.output_folder / "manifest.json"
    log.info(f"Writing metadata to {manifest_path}...")

    try:
        with open(manifest_path, "w") as f:
            json.dump(metadata, f, indent=2)
        log.info("Manifest saved successfully.")
    except Exception as e:
        log.error(f"❌ Failed to write manifest: {e}", exc_info=config.debug)


# === File Scanning and Config Setup ===
def _scan_and_parse_files(input_dir: Path) -> TimepointsDict:
    """Scans input directory for TIFF files and parses timepoint/channel info."""
    timepoints_data: DefaultDict[str, List[ChannelEntry]] = defaultdict(list)
    tiff_regex = re.compile(r".*_ch(\d+)_stack(\d{4}).*?\.tiff?$", re.IGNORECASE)
    log.info(f"Scanning for TIFF files in: {input_dir}")
    found_files: List[Path] = []
    try:
        found_files = list(input_dir.glob("*.tif*"))
    except OSError as e:
        log.error(f"Error scanning {input_dir}: {e}")
        return timepoints_data

    # --- Ruff Fix: E701 ---
    if not found_files:
        log.warning(f"No TIFF files found in {input_dir}.")
        return timepoints_data

    log.info(f"Found {len(found_files)} files. Parsing...")
    parsed_count = 0
    skip_count = 0

    for path in found_files:
        # --- Ruff Fix: E701 ---
        if not path.is_file():
            log.debug(f"Skipping non-file: {path.name}")
            continue

        match = tiff_regex.match(path.name)
        if match:
            try:
                ch_id = int(match.group(1))
                time_id_num = int(match.group(2))

                time_id = f"stack{time_id_num:04d}"
                timepoints_data[time_id].append(
                    {"channel": ch_id, "path": path.resolve()}
                )
                parsed_count += 1
            except ValueError:
                log.warning(f"Skipping {path.name}: Cannot parse numbers.")
                skip_count += 1

            except Exception as e:
                log.warning(f"Skipping {path.name} due to error: {e}")
                skip_count += 1

        else:
            log.debug(f"Skipping {path.name}: No pattern match.")
            skip_count += 1

    log.info(f"Parsed {parsed_count} files.")
    # --- Ruff Fix: E701 ---
    if skip_count > 0:
        log.warning(f"Skipped {skip_count} files.")

    for time_id in timepoints_data:
        timepoints_data[time_id].sort(key=lambda x: x["channel"])
    return timepoints_data


def _setup_configuration(args: Dict[str, Any]) -> PreprocessingConfig:
    """Sets up PreprocessingConfig from command-line arguments."""
    try:
        input_folder = Path(args["--input"]).resolve()
        output_folder = Path(args["--output"]).resolve()
        # --- Ruff Fix: E701 ---
        if not input_folder.is_dir():
            raise FileNotFoundError(f"Input dir not found: {input_folder}")

        output_folder.mkdir(parents=True, exist_ok=True)
        log.info(f"Using Output folder: {output_folder}")

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

        log.info("Configuration loaded.")
        log.debug(f"Config details: {config}")
        return config

    except KeyError as e:
        log.error(f"Config error: Missing {e}")
        raise ValueError(f"Missing: {e}")

    except ValueError as e:
        log.error(f"Config error: Invalid value ({e})")
        raise ValueError(f"Invalid value: {e}")

    except FileNotFoundError as e:
        log.error(f"Config error: {e}")
        raise
    except Exception as e:
        log.error(f"Unexpected config error: {e}", exc_info=True)
        raise ValueError(f"Unexpected config error: {e}")


# === Main Orchestration Function ===
def run_preprocessing(args: Dict[str, Any]):
    """Runs the main TIFF preprocessing pipeline with 3 passes."""

    start_time = time.time()
    log.info("🚀 Starting PixelPacker Preprocessing Pipeline...")

    try:
        config = _setup_configuration(args)
        timepoints_data = _scan_and_parse_files(config.input_folder)
        # --- Ruff Fix: E701 ---
        if not timepoints_data:
            log.error("❌ Aborting: No input files found.")
            return

        global_z_range, layout, tasks = _prepare_tasks_and_layout(
            config, timepoints_data
        )
        # --- Ruff Fix: E701 ---
        if not layout or not tasks or global_z_range is None:
            log.error("❌ Aborting: Failed setup.")
            return

        global_contrast_ranges, pass1_results = _calculate_global_limits(
            tasks, config, global_z_range
        )
        # --- Ruff Fix: E701 ---
        if not pass1_results:
            log.error("❌ Aborting: Failed Pass 1 (Limit Calc).")
            return

        limits_for_processing = (
            global_contrast_ranges if config.use_global_contrast else None
        )
        final_results = _execute_processing_pass(
            pass1_results, config, layout, limits_for_processing
        )

        actual_global_ranges_used = (
            global_contrast_ranges if config.use_global_contrast else None
        )
        metadata = _finalize_metadata(
            final_results, layout, global_z_range, actual_global_ranges_used
        )
        _write_manifest(metadata, config)

    except (ValueError, FileNotFoundError, OSError) as e:
        log.error(f"❌ Preprocessing aborted: {e}")
        return

    except Exception as e:
        log.critical(f"❌ Unexpected critical error: {e}", exc_info=True)
        return

    finally:
        elapsed_time = time.time() - start_time
        log.info(f"🏁 Pipeline finished in {elapsed_time:.2f}s")
