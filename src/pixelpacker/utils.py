# src/pixelpacker/utils.py

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, List

# Import data model - adjust if ChannelEntry moves later
from .data_models import ChannelEntry

# --- Centralized Logging Setup ---
# Configure logging once
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: [%(name)s] %(message)s"
)
# Get a logger instance for this module
log = logging.getLogger(__name__)

# Define custom exception classes here if needed later
# class PixelPackerError(Exception): ...


TimepointsDict = DefaultDict[str, List[ChannelEntry]]


# --- File System Utilities ---
def scan_and_parse_files(input_dir: Path) -> TimepointsDict:
    """
    Scans input directory for TIFF files and parses timepoint/channel info.

    Note: This currently uses a hardcoded regex. Roadmap item #3 involves
    making this pattern configurable.
    """
    timepoints_data: TimepointsDict = defaultdict(list)
    # TODO: Replace with user-supplied pattern (Roadmap #3)
    tiff_regex = re.compile(r".*_ch(\d+)_stack(\d{4}).*?\.tiff?$", re.IGNORECASE)
    log.info(f"Scanning for TIFF files in: {input_dir}")
    found_files: List[Path] = []
    try:
        # Use rglob for potentially nested structures? For now, keeping glob.
        found_files = list(input_dir.glob("*.tif*"))
    except OSError as e:
        log.error(f"Error scanning {input_dir}: {e}")
        return timepoints_data

    if not found_files:
        log.warning(f"No TIFF files found matching '*.tif*' in {input_dir}.")
        return timepoints_data

    log.info(f"Found {len(found_files)} potential files. Parsing filenames...")
    parsed_count = 0
    skip_count = 0

    for path in found_files:
        if not path.is_file():
            log.debug(f"Skipping non-file: {path.name}")
            continue

        match = tiff_regex.match(path.name)
        if match:
            try:
                ch_id = int(match.group(1))
                time_id_num = int(match.group(2))

                # Consistent time ID format
                time_id = f"stack{time_id_num:04d}"
                timepoints_data[time_id].append(
                    {"channel": ch_id, "path": path.resolve()}
                )
                parsed_count += 1
            except ValueError:
                log.warning(f"Skipping {path.name}: Cannot parse channel/stack numbers.")
                skip_count += 1
            except Exception as e:
                # Catch other potential errors during parsing
                log.warning(f"Skipping {path.name} due to error: {e}")
                skip_count += 1
        else:
            log.debug(f"Skipping {path.name}: Filename does not match expected pattern.")
            skip_count += 1

    log.info(f"Successfully parsed {parsed_count} files.")
    if skip_count > 0:
        log.warning(f"Skipped {skip_count} files due to naming pattern or errors.")

    # Sort channels within each timepoint for consistent processing order
    for time_id in timepoints_data:
        timepoints_data[time_id].sort(key=lambda x: x["channel"])

    return timepoints_data