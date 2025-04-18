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

# Import from io_utils and the RENAMED data_models module
from .io_utils import extract_volume, process_channel
from .data_models import VolumeLayout, ChannelEntry

# --- Configure basic logging ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# (Rest of the file remains the same as the previous version...)
# ...

# --- Helper Types ---
TimepointsDict = DefaultDict[str, List[ChannelEntry]]
TimepointResult = Dict[str, Any]

@dataclass
class PreprocessingConfig:
    input_folder: Path
    output_folder: Path
    stretch_mode: str
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

# --- Helper Functions ---

def _scan_and_parse_files(input_dir: Path) -> TimepointsDict:
    # (Implementation unchanged)
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
                timepoints_data[time_id].append({"channel": ch_id, "path": path}) # Uses ChannelEntry from data_models implicitly
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
    # (Implementation unchanged, returns VolumeLayout from data_models)
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
    # ... (exception handling unchanged) ...
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
    # (Implementation unchanged)
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
    # ... (exception handling unchanged) ...
    except KeyError as e:
        log.error(f"Missing expected argument: {e}")
        raise ValueError(f"Configuration error: Missing argument {e}") from e
    except ValueError as e:
        log.error(f"Invalid argument value: {e}")
        raise ValueError(f"Configuration error: Invalid value {e}") from e
    except OSError as e:
        log.error(f"❌ Could not create output directory {args['--output']}: {e}")
        raise


def _prepare_tasks(
    config: PreprocessingConfig,
    timepoints_data: TimepointsDict
) -> Tuple[Optional[VolumeLayout], List[ProcessingTask]]:
    # (Implementation unchanged)
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
        if layout is None:
            layout = _determine_layout(entries[0]["path"])
            if layout is None:
                log.warning(f"Could not determine layout from {time_id}. Trying next timepoint.")
                continue
        if layout:
            for entry in entries:
                task = ProcessingTask(
                    time_id=time_id,
                    channel_entry=entry,
                    config=config,
                )
                tasks_to_submit.append(task)
        else:
            log.warning(f"Layout still undefined when processing {time_id}. Skipping.")
    if layout is None:
        log.error("❌ Could not determine volume layout from any input files. Aborting task preparation.")
        return None, []
    if not tasks_to_submit:
        log.error("❌ No valid processing tasks generated. Aborting.")
        return layout, []
    log.info(f"✅ Prepared {len(tasks_to_submit)} tasks. Layout determined: {layout}")
    return layout, tasks_to_submit


def _execute_tasks_parallel(
    tasks: List[ProcessingTask],
    config: PreprocessingConfig,
    layout: VolumeLayout,
) -> List[ProcessingResult]:
    # (Implementation unchanged - call to process_channel already uses layout object)
    log.info(f"⚙️ Processing {len(tasks)} tasks using up to {config.max_threads} threads...")
    results: List[ProcessingResult] = []
    processed_count = 0
    error_count = 0
    with ThreadPoolExecutor(max_workers=config.max_threads) as executor:
        futures: Dict[Any, ProcessingTask] = {}
        for task in tasks:
            global_limits_tuple: Optional[Tuple[float, float]] = None
            future = executor.submit(
                process_channel, # Call signature unchanged here
                task.time_id,
                task.channel_entry["channel"],
                str(task.channel_entry["path"]),
                layout,
                config.stretch_mode,
                config.dry_run,
                config.debug,
                str(config.output_folder),
                global_limits_tuple,
            )
            futures[future] = task
        # ... (result processing unchanged) ...
        with tqdm(total=len(futures), desc=" ⏳ Processing Channels") as pbar:
            for fut in as_completed(futures):
                task = futures[fut]
                try:
                    result_dict: Optional[Dict[str, Any]] = fut.result()
                    if result_dict:
                        processed_count += 1
                        result_obj = ProcessingResult(
                            time_id=result_dict.get("time_id", task.time_id),
                            channel=result_dict["channel"],
                            filename=result_dict["filename"],
                            p_low=result_dict["intensity_range"]["p_low"],
                            p_high=result_dict["intensity_range"]["p_high"],
                        )
                        results.append(result_obj)
                    else:
                        log.warning(f"Task for T:{task.time_id} C:{task.channel_entry['channel']} returned no result.")
                        error_count += 1
                except Exception as exc:
                    log.error(
                        f"❌ Error processing result for T:{task.time_id} C:{task.channel_entry['channel']}: {exc}",
                        exc_info=config.debug
                    )
                    error_count += 1
                finally:
                    pbar.update(1)

    log.info(f"📊 Processing complete. Successful: {processed_count}, Errors/Skipped: {error_count}")
    return results


def _finalize_metadata(results: List[ProcessingResult], layout: VolumeLayout) -> Dict[str, Any]:
    # (Implementation unchanged)
    log.info("📝 Finalizing metadata...")
    metadata: Dict[str, Any] = {
        "tile_layout": {"cols": layout.cols, "rows": layout.rows},
        "volume_size": {"width": layout.width, "height": layout.height, "depth": layout.depth},
        "channels": 0,
        "timepoints": [],
        "global_intensity": {},
    }
    # ... (rest of implementation unchanged) ...
    timepoints_results: DefaultDict[str, TimepointResult] = defaultdict(
        lambda: {"time": None, "files": {}}
    )
    global_range: DefaultDict[int, Dict[str, float]] = defaultdict(
        lambda: {"p_low": float("inf"), "p_high": float("-inf")}
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
        }
        global_range[res_ch]["p_low"] = min(global_range[res_ch]["p_low"], res.p_low)
        global_range[res_ch]["p_high"] = max(global_range[res_ch]["p_high"], res.p_high)
    metadata["channels"] = max_channel + 1
    processed_time_ids = sorted(timepoints_results.keys())
    metadata["timepoints"] = [
        timepoints_results[tid] for tid in processed_time_ids if timepoints_results[tid]["files"]
    ]
    final_global_intensity = {}
    for ch in range(metadata["channels"]):
        low = global_range[ch]["p_low"]
        high = global_range[ch]["p_high"]
        final_global_intensity[f"c{ch}"] = {
            "p_low": low if low != float("inf") else 0.0,
            "p_high": high if high != float("-inf") else 0.0,
        }
    metadata["global_intensity"] = final_global_intensity
    log.info("Metadata finalized.")
    return metadata


def _write_manifest(metadata: Dict[str, Any], config: PreprocessingConfig):
    # (Implementation unchanged)
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


def run_preprocessing(args: Dict[str, Any]):
    # (Implementation unchanged)
    start_time = time.time()
    log.info("🚀 Starting TIFF to WebP preprocessing...")
    try:
        config = _setup_configuration(args)
        timepoints_data = _scan_and_parse_files(config.input_folder)
        layout, tasks = _prepare_tasks(config, timepoints_data)
        if not layout or not tasks:
            log.error("❌ Aborting run due to issues in task preparation.")
            return
        results = _execute_tasks_parallel(tasks, config, layout)
        metadata = _finalize_metadata(results, layout)
        _write_manifest(metadata, config)
    # ... (exception handling unchanged) ...
    except (ValueError, FileNotFoundError, OSError) as e:
        log.error(f"❌ Preprocessing aborted due to critical error: {e}")
        return
    except Exception as e:
        log.critical(f"❌ An unexpected critical error occurred during orchestration: {e}", exc_info=True)
        return
    elapsed_time = time.time() - start_time
    log.info(f"🏁 Finished in {elapsed_time:.2f}s")