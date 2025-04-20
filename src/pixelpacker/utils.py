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
logging.basicConfig(level=logging.INFO, format="%(levelname)s: [%(name)s] %(message)s")
# Get a logger instance for this module
log = logging.getLogger(__name__)


@contextmanager
def get_executor(config: PreprocessingConfig) -> Iterator[Executor]:
    """Provides the configured executor (Thread or Process) as a context manager."""
    executor_instance: Optional[Executor] = None
    ExecutorClass = ThreadPoolExecutor  # Default
    executor_name = "thread"
    # Prefix for thread names for easier debugging
    thread_prefix = "PixelPacker"

    if config.executor_type == "process":
        # Use INFO level as this is a significant config choice
        log.info(f"Creating ProcessPoolExecutor (max_workers={config.max_threads})")
        ExecutorClass = ProcessPoolExecutor
        executor_name = "process"
        # Add warning about potential issues
        log.debug(
            "Note: ProcessPoolExecutor has higher overhead and requires arguments/results to be pickleable."
        )
    else:
        log.info(f"Creating ThreadPoolExecutor (max_workers={config.max_threads})")
        # Pass thread_name_prefix only to ThreadPoolExecutor where it's supported
        kwargs = {
            "max_workers": config.max_threads,
            "thread_name_prefix": thread_prefix,
        }
        executor_instance = ExecutorClass(**kwargs)  # type: ignore # Handle potential type checking issue with kwargs

    try:
        # Create instance only if not already created (for ThreadPool case)
        if executor_instance is None:
            executor_instance = ExecutorClass(max_workers=config.max_threads)

        yield executor_instance  # Provide the executor to the 'with' block
    finally:
        # Ensure shutdown occurs reliably
        if executor_instance:
            log.debug(f"Shutting down {executor_name} executor.")
            executor_instance.shutdown(wait=True)  # Wait for tasks to complete


TimepointsDict = DefaultDict[str, List[ChannelEntry]]


def scan_and_parse_files(input_dir: Path, input_pattern: str) -> TimepointsDict:
    """
    Scans input directory using the provided glob pattern and parses
    filenames matching the expected '_chN_stackN' convention.

    Args:
        input_dir: The directory to scan.
        input_pattern: The glob pattern provided by the user.

    Returns:
        A dictionary mapping time IDs to lists of ChannelEntry, or empty if
        no files are found or none can be parsed.
    """
    timepoints_data: TimepointsDict = defaultdict(list)
    # Regex for *parsing* channel/stack numbers remains the same
    tiff_regex = re.compile(r".*_ch(\d+)_stack(\d{4}).*?\.tiff?$", re.IGNORECASE)

    log.info(f"Scanning for files in: {input_dir} using pattern: '{input_pattern}'")
    found_files: List[Path] = []
    try:
        # Use the provided input_pattern for globbing
        found_files = list(input_dir.glob(input_pattern))
    except OSError as e:
        log.error(f"Error scanning {input_dir} using pattern '{input_pattern}': {e}")
        return timepoints_data  # Return empty on scanning error

    # --- Added Validation: No files found ---
    if not found_files:
        log.error(
            f"❌ No files found in '{input_dir}' matching pattern '{input_pattern}'."
        )
        return timepoints_data  # Return empty
    # --- End Validation ---

    log.info(
        f"Found {len(found_files)} potential files matching pattern. Parsing filenames..."
    )
    parsed_count = 0
    skip_count = 0

    for path in found_files:
        if not path.is_file():
            log.debug(f"Skipping non-file entry: {path.name}")
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
                log.warning(
                    f"Skipping {path.name}: Could not parse channel/stack numbers from filename despite regex match."
                )
                skip_count += 1
            except Exception as e:
                log.warning(
                    f"Skipping {path.name} due to unexpected parsing error: {e}"
                )
                skip_count += 1
        else:
            # File matched glob pattern but not the parsing regex
            log.warning(
                f"Skipping {path.name}: Does not match expected '_chX_stackY' parsing convention."
            )
            skip_count += 1

    log.info(f"Successfully parsed {parsed_count} files.")
    if skip_count > 0:
        # This log now covers files skipped due to non-matching parsing *or* parsing errors
        log.warning(
            f"Skipped {skip_count} files found by pattern due to naming convention or parsing errors."
        )

    # --- Added Validation: No files parsed ---
    # Check if we found files but couldn't parse any valid ones
    if parsed_count == 0 and len(found_files) > 0:
        log.error(
            f"❌ Found {len(found_files)} files matching pattern, but none could be parsed using the expected '_chX_stackY' naming convention."
        )
        return defaultdict(list)  # Return empty to signal failure
    # --- End Validation ---

    # Sort channels within each timepoint for consistent processing order
    for time_id in timepoints_data:
        timepoints_data[time_id].sort(key=lambda x: x["channel"])

    return timepoints_data
