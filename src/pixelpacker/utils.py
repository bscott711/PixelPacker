# src/pixelpacker/utils.py

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, List, Optional, Iterator
from contextlib import contextmanager
from concurrent.futures import Executor, ThreadPoolExecutor, ProcessPoolExecutor

from .data_models import ChannelEntry, PreprocessingConfig

# --- Centralized Logging Setup ---
# Configure logging once
logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: [%(name)s] %(message)s"
)
# Get a logger instance for this module
log = logging.getLogger(__name__)

@contextmanager
def get_executor(config: PreprocessingConfig) -> Iterator[Executor]:
    """Provides the configured executor (Thread or Process) as a context manager."""
    executor_instance: Optional[Executor] = None
    ExecutorClass = ThreadPoolExecutor # Default
    executor_name = "thread"
    # Prefix for thread names for easier debugging
    thread_prefix = "PixelPacker"

    if config.executor_type == "process":
        # Use INFO level as this is a significant config choice
        log.info(f"Creating ProcessPoolExecutor (max_workers={config.max_threads})")
        ExecutorClass = ProcessPoolExecutor
        executor_name = "process"
        # Add warning about potential issues
        log.debug("Note: ProcessPoolExecutor has higher overhead and requires arguments/results to be pickleable.")
    else:
        log.info(f"Creating ThreadPoolExecutor (max_workers={config.max_threads})")
        # Pass thread_name_prefix only to ThreadPoolExecutor where it's supported
        kwargs = {"max_workers": config.max_threads, "thread_name_prefix": thread_prefix}
        executor_instance = ExecutorClass(**kwargs) # type: ignore # Handle potential type checking issue with kwargs

    try:
        # Create instance only if not already created (for ThreadPool case)
        if executor_instance is None:
             executor_instance = ExecutorClass(max_workers=config.max_threads)

        yield executor_instance # Provide the executor to the 'with' block
    finally:
        # Ensure shutdown occurs reliably
        if executor_instance:
            log.debug(f"Shutting down {executor_name} executor.")
            executor_instance.shutdown(wait=True) # Wait for tasks to complete

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