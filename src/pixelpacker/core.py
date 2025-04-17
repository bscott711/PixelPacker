# tiff_preprocessor/core.py

import json
import logging
import math
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path # <-- Add
from typing import Any, Dict, List, Optional, Tuple, DefaultDict

from tqdm import tqdm

from .io_utils import extract_volume, process_channel

# --- Configure basic logging  ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

# --- Helper Types ---
# Define a type for the channel entry dictionary
ChannelEntry = Dict[str, Any] # Could be more specific: TypedDict('ChannelEntry', {'channel': int, 'path': Path})

# Define a type for the timepoint data dictionary
TimepointsDict = DefaultDict[str, List[ChannelEntry]]

# Define a type for the results dictionary per timepoint
TimepointResult = Dict[str, Any] # Could be TypedDict

# --- Helper Functions ---

def _scan_and_parse_files(input_dir: Path) -> TimepointsDict:
    """Scans input directory for TIFF files and parses timepoint/channel info."""
    timepoints_data: TimepointsDict = defaultdict(list)
    # Regex to capture channel number and stack (timepoint) identifier
    # Making it slightly more specific to capture digits
    tiff_regex = re.compile(r".*_ch(\d+)_stack(\d{4}).*?\.tif(?:f)?$", re.IGNORECASE)
    log.info(f"Scanning for TIFF files in: {input_dir}")

    found_files = list(input_dir.glob('*.tif*')) # Use glob for finding files
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
                # Use the full stack identifier as time_id for robustness
                time_id = f"stack{match.group(2)}"
                timepoints_data[time_id].append({"channel": ch_id, "path": path})
                parsed_count += 1
            except (ValueError, IndexError) as e:
                 log.warning(f"Skipping {path.name} — Error parsing captured groups: {e}")
                 skip_count += 1
            except Exception as e:
                 log.warning(f"Skipping {path.name} — Unexpected error during parsing: {e}")
                 skip_count += 1
        # else: # Optionally log files that didn't match the regex
        #     log.debug(f"Skipping {path.name} - does not match expected pattern.")

    log.info(f"Parsed {parsed_count} files successfully.")
    if skip_count > 0:
         log.warning(f"Skipped {skip_count} files due to naming/parsing issues.")

    return timepoints_data


def _determine_layout(
    first_valid_path: Path,
) -> Optional[Tuple[int, int, int, int, int, int, int]]:
    """Determines volume dimensions and tile layout from a sample TIFF file."""
    try:
        log.info(f"Determining layout from: {first_valid_path.name}")
        vol = extract_volume(first_valid_path)
        d, h, w = vol.shape
        cols = math.ceil(math.sqrt(d))
        rows = math.ceil(d / cols)
        tile_w = cols * w
        tile_h = rows * h
        log.info(f"Layout set: Volume({w}x{h}x{d}), Tile({tile_w}x{tile_h}), Grid({cols}x{rows})")
        return d, h, w, cols, rows, tile_w, tile_h
    except Exception as e:
        log.error(f"Error processing layout from {first_valid_path.name}: {e}")
        return None


# --- Main Function ---

def run_preprocessing(args: Dict[str, Any]):
    """
    Runs the TIFF to WebP preprocessing pipeline.

    Orchestrates file discovery, layout determination, parallel channel processing,
    and manifest generation based on command-line arguments.

    Args:
        args: Dictionary of arguments parsed by docopt from the CLI.
              Expected keys: '--input', '--output', '--stretch', '--global-contrast',
                             '--dry-run', '--debug', '--threads'.
    """
    # --- Configuration Setup ---
    try:
        input_folder = Path(args["--input"]).resolve() # Use Path and resolve
        output_folder = Path(args["--output"]).resolve()
        stretch_mode: str = args["--stretch"]
        use_global: bool = args["--global-contrast"]
        dry_run: bool = args["--dry-run"]
        debug: bool = args["--debug"]
        max_threads: int = int(args["--threads"])
    except KeyError as e:
        log.error(f"Missing expected argument: {e}")
        return
    except ValueError as e:
         log.error(f"Invalid argument value: {e}")
         return

    log.info("🚀 Starting TIFF to WebP preprocessing...")
    if dry_run:
        log.info("💡 Dry-run mode: No files will be written.")

    try:
        output_folder.mkdir(parents=True, exist_ok=True) # Use Path.mkdir
    except OSError as e:
        log.error(f"❌ Could not create output directory {output_folder}: {e}")
        return

    # --- Pass 1: File Scanning, Parsing, Layout Determination ---
    start_time = time.time()

    timepoints_data = _scan_and_parse_files(input_folder)
    sorted_time_ids = sorted(timepoints_data.keys())

    if not sorted_time_ids:
        log.error("❌ No valid timepoints found after parsing filenames. Aborting.")
        return

    # Prepare metadata structure
    metadata: Dict[str, Any] = {
        "tile_layout": {},
        "volume_size": {},
        "channels": 0,
        "timepoints": [],
        "global_intensity": {} # Initialize sub-dict
    }
    timepoints_results: DefaultDict[str, TimepointResult] = defaultdict(lambda: {"time": None, "files": {}})
    global_range: DefaultDict[int, Dict[str, float]] = defaultdict(
        lambda: {"p_low": float("inf"), "p_high": float("-inf")}
    )

    # Layout variables initialization
    layout_info: Optional[Tuple[int, int, int, int, int, int, int]] = None
    tasks_to_submit: List[Tuple[str, ChannelEntry]] = []
    total_tasks = 0

    log.info("🔍 Analyzing layout and preparing tasks...")
    for time_id in sorted_time_ids:
        # Sort channel entries by channel number
        entries = sorted(timepoints_data[time_id], key=lambda x: x["channel"])
        if not entries:
            log.warning(f"No valid channels found for timepoint {time_id}, skipping.")
            continue

        # Determine layout from the first valid timepoint found
        if layout_info is None:
            layout_info = _determine_layout(entries[0]["path"])
            if layout_info is None:
                 log.warning(f"Could not determine layout from {time_id}. Trying next timepoint.")
                 continue # Skip this timepoint if layout calculation failed
            else:
                 d, h, w, cols, rows, tile_w, tile_h = layout_info
                 metadata["volume_size"] = {"width": w, "height": h, "depth": d}
                 metadata["tile_layout"] = {"cols": cols, "rows": rows}

        # If layout is determined, add tasks for this timepoint
        if layout_info:
            for entry in entries:
                # Check if volume shape matches layout (optional, requires reading file again)
                # Could add a check here if needed, but process_channel handles it
                tasks_to_submit.append((time_id, entry))
                total_tasks += 1
        else:
            # This case should ideally not be reached if the loop structure is correct
            log.warning(f"Skipping task preparation for {time_id} as layout is still undefined.")


    if layout_info is None:
        log.error("❌ Could not determine volume layout from any input files. Aborting.")
        return
    # Unpack layout info for use in submission loop
    d, h, w, cols, rows, tile_w, tile_h = layout_info

    if total_tasks == 0:
        log.error("❌ No valid processing tasks found after filtering. Aborting.")
        return

    log.info(f"✅ Ready to process {total_tasks} channel(s) across {len(timepoints_results)} potential timepoint(s).")


    # --- Pass 2: Execute tasks in parallel ---
    log.info(f"⚙️ Processing tasks using up to {max_threads} threads...")
    processed_count = 0
    error_count = 0
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures: Dict[Any, Tuple[str, int]] = {} # Future -> (time_id, ch_id)
        for time_id, entry in tasks_to_submit:
            ch = entry["channel"]
            # Determine global limits based on *currently known* values
            current_global_low = global_range[ch]["p_low"]
            current_global_high = global_range[ch]["p_high"]
            global_limits: Optional[Tuple[float, float]] = None
            if use_global and current_global_high > current_global_low:
                # Only pass limits if they seem valid (inf/-inf means not set yet)
                global_limits = (current_global_low, current_global_high)

            future = executor.submit(
                process_channel,
                time_id,
                ch,
                str(entry["path"]), # Ensure path is string for io_utils
                tile_w, tile_h, w, h, d, cols, rows,
                stretch_mode,
                dry_run,
                debug,
                str(output_folder), # Ensure path is string
                global_limits,
            )
            futures[future] = (time_id, ch)

        # Process results as they complete
        # Use TQDM context manager for cleaner handling
        with tqdm(total=len(futures), desc=" ⏳ Processing Channels") as pbar:
            for fut in as_completed(futures):
                time_id, ch_id = futures[fut]
                try:
                    result: Optional[Dict[str, Any]] = fut.result()
                    if result:
                        processed_count += 1
                        # Aggregate results
                        res_time = result.get("time_id", time_id) # Use result time if available
                        res_ch = result["channel"]
                        timepoints_results[res_time]["time"] = res_time
                        timepoints_results[res_time]["files"][f"c{res_ch}"] = {
                            "file": result["filename"],
                            "p_low": result["intensity_range"]["p_low"],
                            "p_high": result["intensity_range"]["p_high"],
                        }
                        metadata["channels"] = max(metadata["channels"], res_ch + 1)

                        # Update global intensity range tracking immediately
                        global_range[res_ch]["p_low"] = min(
                            global_range[res_ch]["p_low"], result["intensity_range"]["p_low"]
                        )
                        global_range[res_ch]["p_high"] = max(
                            global_range[res_ch]["p_high"], result["intensity_range"]["p_high"]
                        )
                    # else: # process_channel returned None, likely due to shape mismatch or error
                    #     log.warning(f"No result returned for T{time_id} C{ch_id}.")
                    #     error_count += 1 # Count as error if None means failure

                except Exception as exc:
                     log.error(f"❌ Error processing result for T{time_id} C{ch_id}: {exc}", exc_info=debug) # Show stack trace if debug
                     error_count += 1
                finally:
                    pbar.update(1) # Ensure progress bar updates even on error/None result


    log.info(f"📊 Processing complete. Successful: {processed_count}, Errors/Skipped: {error_count}")

    # --- Finalize Metadata ---
    processed_time_ids = sorted(timepoints_results.keys())
    # Filter out timepoints that ended up with no successfully processed files
    metadata["timepoints"] = [
        timepoints_results[tid] for tid in processed_time_ids if timepoints_results[tid]["files"]
    ]

    # Calculate final global intensity dictionary
    final_global_intensity = {}
    for ch in range(metadata["channels"]):
         low = global_range[ch]["p_low"]
         high = global_range[ch]["p_high"]
         final_global_intensity[f"c{ch}"] = {
             "p_low": low if low != float("inf") else 0.0,
             "p_high": high if high != float("-inf") else 0.0,
         }
    metadata["global_intensity"] = final_global_intensity


    # --- Write Manifest ---
    if metadata["timepoints"] and not dry_run:
        manifest_path = output_folder / "manifest.json" # Use Path object
        try:
            with open(manifest_path, "w") as f:
                json.dump(metadata, f, indent=2)
            log.info(f"📄 Metadata saved to {manifest_path}")
        except IOError as e:
             log.error(f"❌ Failed to write manifest file {manifest_path}: {e}")
        except TypeError as e:
             log.error(f"❌ Failed to serialize metadata to JSON: {e}")

    # --- Final Status Reporting ---
    elif dry_run:
        log.info("🧪 Dry run complete — manifest not written.")
    elif not metadata["timepoints"]:
         log.info("⚠️ No timepoints were successfully processed — manifest not written.")
    # The 'else' case from before is unlikely now due to explicit checks

    elapsed_time = time.time() - start_time
    log.info(f"🏁 Finished in {elapsed_time:.2f}s")