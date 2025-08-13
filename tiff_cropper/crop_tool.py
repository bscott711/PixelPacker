import concurrent.futures
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Attempt to import tkinter for optional GUI prompts
try:
    import tkinter
    from tkinter import filedialog

    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

import tifffile
import typer
from tqdm import tqdm

# --- Configuration ---

# Set up basic logging (File handler added in main if log_file specified)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stderr)],  # Log INFO+ to stderr
)
logger = logging.getLogger(__name__)

# Supported TIFF extensions
TIFF_EXTENSIONS: Set[str] = {".tif", ".tiff"}

# Create a Typer application
app = typer.Typer(
    name="tiff-crop-tool",
    help=(
        "CLI tool to efficiently crop TIFF stacks (Stack, ..., X, ...) dimensions"
        " using slice-by-slice processing."
    ),
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


# --- Helper Function for Directory Prompt ---
def prompt_for_directory(title: str) -> Optional[Path]:
    """Uses Tkinter (if available) to ask the user to select a directory.

    Args:
        title: The title for the directory selection dialog window.

    Returns:
        The selected Path object, or None if the user cancels or Tkinter is unavailable.
    """
    if not TKINTER_AVAILABLE:
        logger.error(
            "Tkinter is not available in this environment. "
            "Cannot show GUI directory prompt."
        )
        typer.secho(
            "Error: GUI prompts require Tkinter, which is not available. "
            "Please provide paths via command-line arguments.",
            fg=typer.colors.RED,
            err=True,
        )
        return None

    root: Optional[tkinter.Tk] = None  # type: ignore[name-defined]
    try:
        # Hide the root window
        root = tkinter.Tk()  # type: ignore[name-defined]
        root.withdraw()
        # Bring window to front
        root.attributes("-topmost", True)

        # Explicitly lifting can help on some platforms like Linux
        if sys.platform.startswith("linux"):
            try:
                # This might fail on some headless systems, hence the inner try
                root.lift()
            except tkinter.TclError:  # type: ignore[name-defined]
                logger.warning(
                    "Could not 'lift' Tkinter window, dialog might be hidden."
                )

        selected_path_str = filedialog.askdirectory(title=title, parent=root)  # type: ignore[name-defined]

        if selected_path_str:  # User selected a path
            return Path(selected_path_str)
        # User cancelled
        logger.warning(f"Directory selection cancelled by user for: {title}")
        typer.echo("Directory selection cancelled.")
        return None
    except Exception as e:
        logger.error(f"Error during Tkinter directory prompt: {e}", exc_info=True)
        typer.secho(
            f"Error showing directory dialog: {e}", fg=typer.colors.RED, err=True
        )
        return None
    finally:
        if root:
            try:
                root.destroy()  # Clean up the hidden root window
            except Exception:
                logger.warning(
                    "Exception during Tkinter root window cleanup.",
                    exc_info=True,
                )


# --- Core Cropping Logic (Memory Efficient & Flexible Axes) ---
def crop_z_x_stack_mem_efficient(
    input_path: Path,
    output_path: Path,
    z_start: Optional[int],
    z_end: Optional[int],
    x_start: Optional[int],
    x_end: Optional[int],
) -> Tuple[bool, Optional[str]]:
    """
    Reads a TIFF stack slice by slice, crops the first (stack/Z) dimension
    and the 'X' dimension (last spatial dim if not named 'X'), saves the result.

    Args:
        input_path: Path to the input TIFF file.
        output_path: Path where the cropped TIFF file will be saved.
        z_start: Starting slice index of the first dimension (inclusive).
                 Defaults to 0 if None.
        z_end: Ending slice index of the first dimension (exclusive).
               Defaults to stack depth if None.
        x_start: Starting pixel index of the X dimension (inclusive).
                 Defaults to 0 if None.
        x_end: Ending pixel index of the X dimension (exclusive).
               Defaults to stack width if None.

    Returns:
        A tuple containing:
            - bool: True if cropping was successful, False otherwise.
            - Optional[str]: An error message if unsuccessful, otherwise None.
    """
    try:
        with tifffile.TiffFile(input_path) as tif:
            # --- Get Metadata and Validate ---
            if not tif.series or not tif.series[0].shape:
                msg = "Could not find image series or shape."
                logger.warning(f"Skipping {input_path.name}: {msg}")
                return False, msg

            series = tif.series[0]
            original_shape = series.shape
            original_axes = (
                series.axes.upper() if series.axes else ""
            )  # Ensure uppercase, handle None axes

            # Basic validation: Require >= 3 dimensions for cropping first (Z/stack) and one spatial (X)
            if len(original_shape) < 3:
                msg = (
                    f"Unsupported shape {original_shape}. Requires at least 3"
                    f" dimensions for Z/Stack and X cropping (axes:"
                    f" '{series.axes}')."
                )
                logger.warning(f"Skipping {input_path.name}: {msg}")
                return False, msg

            # --- Identify Dimension Indices ---
            # Assume first dimension is Z/stack dimension
            z_dim_index = 0
            num_z_or_stack = original_shape[z_dim_index]
            logger.debug(
                f"'{input_path.name}': Assuming first dimension (index"
                f" {z_dim_index}, size {num_z_or_stack}) is the stack/Z"
                f" dimension. Original axes: '{original_axes}'."
            )

            # Find X dimension index in the original axes string
            x_dim_original_index = original_axes.find("X")
            if x_dim_original_index == -1:
                # Assumption: If 'X' not explicitly named, assume it's the LAST dimension
                x_dim_original_index = len(original_shape) - 1
                logger.debug(
                    f"'{input_path.name}': 'X' axis not found in"
                    f" '{original_axes}'. Assuming last dimension (index"
                    f" {x_dim_original_index}) is X."
                )
            elif x_dim_original_index == z_dim_index:
                # Check if 'X' conflicts with the assumed stack dimension
                msg = (
                    f"Detected 'X' axis ('{original_axes}') conflicts with"
                    f" assumed Z/stack axis (index {z_dim_index})."
                )
                logger.warning(f"Skipping {input_path.name}: {msg}")
                return False, msg
            else:
                logger.debug(
                    f"'{input_path.name}': Found 'X' axis at original index"
                    f" {x_dim_original_index} in '{original_axes}'."
                )

            width_x = original_shape[x_dim_original_index]

            # Determine the index of X within the slice data (shape after removing Z/stack dim)
            # If X's original index was > 0, its index in the slice data (shape[1:]) is original_index - 1.
            if x_dim_original_index > z_dim_index:  # Ensure X wasn't the stack dim
                x_dim_slice_index = x_dim_original_index - 1
                logger.debug(
                    f"'{input_path.name}': Calculated X index within slice data:"
                    f" {x_dim_slice_index}."
                )
            else:
                # This case should have been caught by the conflict check, but log defensively.
                msg = (
                    f"Internal logic error: X dimension index"
                    f" ({x_dim_original_index}) is not greater than Z/stack"
                    f" dimension index ({z_dim_index}). Axes:"
                    f" '{original_axes}'"
                )
                logger.error(f"Error processing {input_path.name}: {msg}")
                return False, msg

            # --- Determine Effective Crop Ranges ---
            eff_z_start = max(0, z_start if z_start is not None else 0)
            eff_z_end = min(
                num_z_or_stack, z_end if z_end is not None else num_z_or_stack
            )
            eff_x_start = max(0, x_start if x_start is not None else 0)
            eff_x_end = min(width_x, x_end if x_end is not None else width_x)

            if eff_z_start >= eff_z_end or eff_x_start >= eff_x_end:
                msg = (
                    f"Invalid calculated crop range resulting in zero size: "
                    f"Stack=[{eff_z_start}:{eff_z_end}), X=[{eff_x_start}:{eff_x_end}). "
                    f"Original Stack={num_z_or_stack}, X={width_x}."
                )
                logger.warning(f"Skipping {input_path.name}: {msg}")
                return False, msg

            # --- Process Slice by Slice ---
            with tifffile.TiffWriter(output_path, bigtiff=True) as writer:
                for z_or_stack_index in range(eff_z_start, eff_z_end):
                    try:
                        # Use the actual page corresponding to the stack index
                        page = tif.pages[z_or_stack_index]
                        slice_data = page.asarray()

                        # --- Dynamic Slicing for X Crop ---
                        expected_slice_ndim = len(original_shape) - 1
                        if slice_data.ndim != expected_slice_ndim:
                            msg = (
                                f"Slice {z_or_stack_index} dimension mismatch."
                                f" Expected {expected_slice_ndim} dims, got"
                                f" {slice_data.ndim}. Original shape:"
                                f" {original_shape}, axes: '{original_axes}'"
                            )
                            logger.error(f"Error processing {input_path.name}: {msg}")
                            output_path.unlink(missing_ok=True)
                            return False, msg

                        # Ensure calculated slice index is valid for the actual slice data shape
                        if not (0 <= x_dim_slice_index < slice_data.ndim):
                            msg = (
                                f"Calculated X dimension index ({x_dim_slice_index})"
                                f" is out of bounds for slice data dimensions"
                                f" ({slice_data.ndim}). Slice shape:"
                                f" {slice_data.shape}. Original axes:"
                                f" '{original_axes}'"
                            )
                            logger.error(f"Error processing {input_path.name}: {msg}")
                            output_path.unlink(missing_ok=True)
                            return False, msg

                        # Build the slicing tuple dynamically
                        crop_slice_tuple = [slice(None)] * slice_data.ndim
                        crop_slice_tuple[x_dim_slice_index] = slice(
                            eff_x_start, eff_x_end
                        )

                        cropped_slice = slice_data[tuple(crop_slice_tuple)]
                        # --- End Dynamic Slicing ---

                    except IndexError as e:  # Catch reading page error
                        msg = (
                            f"Could not read stack slice index"
                            f" {z_or_stack_index}. Error: {e}"
                        )
                        logger.error(f"Error processing {input_path.name}: {msg}")
                        output_path.unlink(missing_ok=True)
                        return False, msg
                    except (
                        Exception
                    ) as e:  # Catch other errors during read/slice generation
                        msg = f"Error preparing slice {z_or_stack_index}: {e}"
                        logger.error(
                            f"Error processing {input_path.name}: {msg}",
                            exc_info=True,
                        )
                        output_path.unlink(missing_ok=True)
                        return False, msg

                    # --- Get Photometric Interpretation ---
                    try:
                        photometric_tag = page.tags.get(  # type: ignore
                            "PhotometricInterpretation"
                        )
                        input_photometric = (
                            getattr(photometric_tag, "value", None)
                            if photometric_tag
                            else None
                        )

                        if isinstance(input_photometric, int):
                            photometric_map = {
                                0: "miniswhite",
                                1: "minisblack",
                                2: "rgb",
                                3: "palette",
                                5: "cmyk",
                                # Add others if needed based on TIFF spec
                            }
                            photometric = photometric_map.get(
                                input_photometric, "minisblack"
                            )  # Fallback
                        elif isinstance(input_photometric, str):
                            photometric = input_photometric.lower()
                        else:
                            # Default if tag missing or has unexpected type
                            photometric = "minisblack"
                            if input_photometric is not None:
                                logger.debug(
                                    f"'{input_path.name}': Slice"
                                    f" {z_or_stack_index} has unexpected"
                                    f" PhotometricInterpretation type:"
                                    f" {type(input_photometric)}. Using default."
                                )

                    except Exception as e:
                        logger.warning(
                            f"Could not determine photometric interpretation for"
                            f" slice {z_or_stack_index} in"
                            f" {input_path.name}, using default 'minisblack'."
                            f" Error: {e}"
                        )
                        photometric = "minisblack"
                    # --- End Photometric ---

                    if cropped_slice.size == 0:
                        msg = (
                            f"Cropped slice at index {z_or_stack_index}"
                            f" resulted in empty data (shape:"
                            f" {cropped_slice.shape}). Skipping write for this"
                            " slice."
                        )
                        logger.warning(f"Issue processing {input_path.name}: {msg}")
                        continue

                    # --- Simple Write Call ---
                    try:
                        writer.write(
                            cropped_slice,
                            photometric=photometric,
                            contiguous=True,
                        )
                    except Exception as e:
                        msg = (
                            f"Error writing cropped slice {z_or_stack_index}"
                            f" to {output_path.name}: {e}"
                        )
                        logger.error(
                            f"Failed to process {input_path.name}: {msg}",
                            exc_info=True,
                        )
                        output_path.unlink(missing_ok=True)  # Attempt cleanup
                        return False, msg
                    # --- END Simple Write ---

            logger.debug(
                f"Successfully wrote cropped data from {input_path.name} to"
                f" {output_path.name}"
            )
            return True, None

    # --- Error handling ---
    except tifffile.TiffFileError as e:
        msg = f"Tifffile error opening/reading {input_path.name}: {e}"
        logger.error(f"Failed to process {input_path.name}: {msg}")
        return False, msg
    except ValueError as e:  # Catch potential errors during write/close
        msg = f"ValueError during processing/writing {output_path.name}: {e}"
        logger.error(f"Failed to process {input_path.name}: {msg}", exc_info=True)
        output_path.unlink(missing_ok=True)  # Attempt cleanup
        return False, msg
    except MemoryError:
        msg = "MemoryError during processing."
        logger.error(f"Failed to process {input_path.name}: {msg}")
        return False, msg
    except Exception as e:
        msg = (
            f"Unexpected error processing {input_path.name}:"
            f" {e.__class__.__name__}: {e}"
        )
        logger.error(f"Failed to process {input_path.name}: {msg}", exc_info=True)
        # Attempt cleanup if error is not file format/read related
        if not isinstance(e, (tifffile.TiffFileError, IndexError, ValueError)):
            output_path.unlink(missing_ok=True)
        return False, msg


# --- Worker Task ---
def worker_task(
    args: Tuple[Path, Path, Optional[int], Optional[int], Optional[int], Optional[int]],
) -> Tuple[Path, bool, Optional[str]]:
    """Helper function to unpack arguments for map and return input path + result."""
    input_path, output_path, z_start, z_end, x_start, x_end = args
    success, error_msg = crop_z_x_stack_mem_efficient(
        input_path, output_path, z_start, z_end, x_start, x_end
    )
    return input_path, success, error_msg


# --- Cropping Runner ---
def run_cropping(
    input_dir: Path,
    output_dir: Path,
    start_z: Optional[int],  # Can be None
    end_z: Optional[int],  # Can be None
    start_x: Optional[int],  # Can be None
    end_x: Optional[int],  # Can be None
    max_workers: int,
):
    """Discovers TIFF files, manages parallel cropping, and reports summary."""
    start_perf_time = time.perf_counter()

    # --- Ensure final output directory exists ---
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured output directory exists: {output_dir}")
    except OSError as e:
        logger.error(
            f"Error creating output directory {output_dir}: {e}", exc_info=True
        )
        typer.secho(
            f"Error: Could not create output directory {output_dir}: {e}",
            fg=typer.colors.RED,
            err=True,
        )
        # Exit here if directory cannot be created
        raise typer.Exit(code=1) from e

    # Use effective start values (defaulting to 0) for logging/display
    # Note: these 'eff' values are for display only, the worker gets the original (potentially None) values
    eff_start_z_disp = start_z if start_z is not None else 0
    eff_start_x_disp = start_x if start_x is not None else 0
    z_range_str = f"[{eff_start_z_disp}:{end_z if end_z is not None else 'end'})"
    x_range_str = f"[{eff_start_x_disp}:{end_x if end_x is not None else 'end'})"

    log_msg = (
        f"Starting Stack={z_range_str}, X={x_range_str} crop process using up"
        f" to {max_workers} workers."
    )
    logger.info(log_msg)
    typer.echo(log_msg)

    # --- Collect tasks using glob for efficiency ---
    logger.info(f"Scanning input directory for TIFF files: {input_dir}")
    tiff_files = []
    for ext in TIFF_EXTENSIONS:
        try:
            tiff_files.extend(input_dir.glob(f"*{ext}"))
        except OSError as e:
            logger.error(
                f"Error scanning input directory {input_dir}: {e}", exc_info=True
            )
            typer.secho(
                f"Error scanning input directory: {e}", fg=typer.colors.RED, err=True
            )
            raise typer.Exit(code=1) from e

    tiff_files = [f for f in tiff_files if f.is_file()]

    if not tiff_files:
        msg = (
            f"No TIFF files ({', '.join(TIFF_EXTENSIONS)}) found in the input"
            f" directory: {input_dir}"
        )
        logger.warning(msg)
        typer.echo(msg)
        raise typer.Exit()  # No files to process

    # Pass original start/end values (which might be None) to worker
    tasks_to_process = [
        (entry, output_dir / entry.name, start_z, end_z, start_x, end_x)
        for entry in tiff_files
    ]

    processed_count = 0
    error_count = 0
    failed_files_details: List[Tuple[str, str]] = []
    num_tasks = len(tasks_to_process)
    log_msg = f"Found {num_tasks} TIFF files. Starting parallel processing..."
    logger.info(log_msg)
    typer.echo(log_msg)

    # --- Execute tasks in parallel with tqdm ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results_iterator = executor.map(worker_task, tasks_to_process)

        for input_path, success, error_msg in tqdm(
            results_iterator,
            total=num_tasks,
            desc="Cropping Files",
            unit="file",
            ncols=100,
            dynamic_ncols=True,
        ):
            if success:
                processed_count += 1
            else:
                error_count += 1
                failed_files_details.append(
                    (input_path.name, error_msg or "Unknown processing error")
                )

    logger.info("Parallel execution finished.")
    typer.echo("\nParallel execution finished.")

    # --- Final Summary ---
    typer.echo("\n" + "=" * 30)
    typer.secho("Processing Summary:", bold=True)
    log_summary = ["Processing Summary:"]

    summary_color = typer.colors.GREEN if error_count == 0 else typer.colors.YELLOW
    success_msg = f"Successfully processed: {processed_count} files."
    typer.secho(success_msg, fg=summary_color)
    log_summary.append(success_msg)

    if error_count > 0:
        error_msg_summary = f"Failed/Skipped during processing: {error_count} files."
        typer.secho(error_msg_summary, fg=typer.colors.RED)
        log_summary.append(error_msg_summary)

        typer.secho(
            "Failed/Skipped file details (first 10 shown):", fg=typer.colors.YELLOW
        )
        log_summary.append("Failed/Skipped file details:")
        for i, (filename, msg) in enumerate(failed_files_details):
            log_line = f"  - {filename}: {msg}"
            log_summary.append(log_line)
            if i < 10:  # Only print first 10 to console
                typer.secho(log_line, fg=typer.colors.YELLOW)
        if len(failed_files_details) > 10:
            more_msg = (
                f"  (...and {len(failed_files_details) - 10} more - check log"
                " file for full details if configured)"
            )
            typer.secho(more_msg, fg=typer.colors.YELLOW)
            log_summary.append(more_msg)

    end_perf_time = time.perf_counter()
    total_time = end_perf_time - start_perf_time
    time_log = f"Total wall-clock time: {total_time:.2f} seconds"
    typer.echo(time_log)
    log_summary.append(time_log)
    typer.echo("=" * 30)

    logger.info("\n".join(log_summary))


# --- Typer CLI Command Definition ---
@app.command(
    context_settings={"help_option_names": ["-h", "--help"]},
)
def main(
    input_dir: Optional[Path] = typer.Option(
        None,
        "--input",
        "-i",
        help=(
            "Path to the directory containing input TIFF stacks. If omitted"
            " and --no-gui is not used, prompts for selection."
        ),
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        show_default=False,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help=(
            "Path to the directory where cropped TIFF stacks will be saved. If"
            " omitted and --no-gui is not used, prompts for selection."
        ),
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        show_default=False,
    ),
    start_z: Optional[int] = typer.Option(
        None,  # Default start Z to None initially
        "--start-z",
        "-sz",
        help=(
            "Starting slice index of the first dimension (inclusive)."
            " Defaults to 0 if not in config."
        ),
        min=0,  # Keep validation constraint
        show_default="0 (or config)",  # Clarify default source
    ),
    end_z: Optional[int] = typer.Option(
        None,
        "--end-z",
        "-ez",
        help=(
            "Ending slice index of the first dimension (exclusive). Defaults"
            " to the end of the stack."
        ),
        min=1,
        show_default="stack depth",
    ),
    start_x: Optional[int] = typer.Option(
        None,  # Default start X to None initially
        "--start-x",
        "-sx",
        help=("Starting X pixel index (inclusive). Defaults to 0 if not in config."),
        min=0,  # Keep validation constraint
        show_default="0 (or config)",  # Clarify default source
    ),
    end_x: Optional[int] = typer.Option(
        None,
        "--end-x",
        "-ex",
        help=(
            "Ending X pixel index (exclusive). Defaults to the full width of the stack."
        ),
        min=1,
        show_default="stack width",
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help=(
            "Path to a JSON configuration file (CLI arguments override config"
            " settings)."
        ),
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    workers: Optional[int] = typer.Option(
        None,
        "--workers",
        "-w",
        help=(
            "Number of worker threads for parallel processing. Defaults to"
            f" system CPU count ({os.cpu_count() or 'N/A'}). Adjust based on"
            " I/O vs CPU bottleneck."
        ),
        min=1,
        show_default="CPU count",
    ),
    log_file: Optional[Path] = typer.Option(
        None,
        "--log-file",
        help="Optional file path to write detailed logs.",
        dir_okay=False,
        writable=True,
        resolve_path=True,
    ),
    no_gui: bool = typer.Option(
        False,
        "--no-gui",
        help=(
            "Disable graphical directory selection prompts (requires paths via"
            " CLI or config)."
        ),
        is_flag=True,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging (DEBUG level) to console and log file.",
        is_flag=True,
    ),
):
    """
    Efficiently crops TIFF image stacks in the first (Stack/Z) and X dimensions
    using slice-by-slice processing to minimize memory usage. Assumes the first
    dimension is the stack dimension and identifies the X dimension (or assumes
    the last dimension if 'X' axis is not specified).

    Requires Input (--input) and Output (--output) directories, either via
    these flags, a config file, or GUI prompts (unless --no-gui is used).

    Examples:
        # Specify paths via CLI, crop Stack 10-49, X 100-599
        python crop_tool.py -i /in -o /out -sz 10 -ez 50 -sx 100 -ex 600

        # Run without paths to trigger GUI prompts (if available)
        python crop_tool.py -sz 10 -ez 50

        # Use config file, disable GUI prompts, 8 workers
        python crop_tool.py --config settings.json --no-gui -w 8
    """
    # --- Setup Logging File Handler and Verbosity ---
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(log_level)

    if log_file:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d"
                " - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(formatter)
            # Log DEBUG level to file regardless of console verbosity
            file_handler.setLevel(logging.DEBUG)
            logging.getLogger().addHandler(file_handler)
            logger.info(f"Logging detailed output (DEBUG level) to: {log_file}")
        except Exception as e:
            typer.secho(
                f"Warning: Could not configure log file at {log_file}: {e}."
                " Continuing without file logging.",
                fg=typer.colors.YELLOW,
                err=True,
            )
            logger.error(f"Failed to configure log file handler: {e}", exc_info=True)

    logger.info("--- TIFF Cropping Tool Started ---")
    logger.debug(
        f"Raw arguments: input={input_dir}, output={output_dir}, sz={start_z},"
        f" ez={end_z}, sx={start_x}, ex={end_x}, config={config_file},"
        f" workers={workers}, log={log_file}, no_gui={no_gui},"
        f" verbose={verbose}"
    )

    # --- Load Configuration File ---
    config: Dict[str, Any] = {}
    if config_file:
        if not config_file.is_file():
            msg = (
                "Error: Specified config file not found or is not a file:"
                f" {config_file}"
            )
            logger.error(msg)
            typer.secho(msg, fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        try:
            with config_file.open("r", encoding="utf-8") as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from: {config_file}")
            typer.echo(f"Loaded configuration from: {config_file}")
            logger.debug(f"Config file contents: {json.dumps(config, indent=2)}")
        except json.JSONDecodeError as e:
            msg = f"Error: Could not decode JSON from config file: {config_file}\n{e}"
            logger.error(msg)
            typer.secho(msg, fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from e
        except OSError as e:
            msg = f"Error reading config file {config_file}: {e}"
            logger.error(msg, exc_info=True)
            typer.secho(msg, fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from e

    # --- Determine Input/Output Directories ---
    final_input_dir = input_dir
    if final_input_dir is None:
        config_input = config.get("input_dir")
        if config_input and isinstance(config_input, str):
            final_input_dir = Path(config_input)
            logger.info(f"Using input directory from config file: {final_input_dir}")
        elif not no_gui:
            typer.echo("Input directory not specified, prompting for selection...")
            final_input_dir = prompt_for_directory(
                "Select Input Directory (containing TIFFs)"
            )
            if final_input_dir is None:
                logger.error("No input directory selected or provided. Exiting.")
                raise typer.Exit(code=1)
            logger.info(f"Input directory selected via dialog: {final_input_dir}")
            typer.echo(f"Using input directory: {final_input_dir}")
        else:
            msg = (
                "Error: Input directory must be provided via --input option or"
                " config file when --no-gui is used."
            )
            logger.error(msg)
            typer.secho(msg, fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)

    try:
        # Ensure final_input_dir is resolved and validated before proceeding
        if final_input_dir is None:  # Should be caught above, but safety check
            raise ValueError("Input directory could not be determined.")
        final_input_dir = final_input_dir.resolve()
        if not final_input_dir.is_dir():
            msg = f"Error: Final input path is not a valid directory: {final_input_dir}"
            logger.error(msg)
            typer.secho(msg, fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        if not os.access(final_input_dir, os.R_OK):
            msg = f"Error: Final input directory is not readable: {final_input_dir}"
            logger.error(msg)
            typer.secho(msg, fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
    except Exception as e:
        msg = f"Error resolving or validating input directory '{final_input_dir}': {e}"
        logger.error(msg, exc_info=True)
        typer.secho(msg, fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e

    final_output_dir = output_dir
    if final_output_dir is None:
        config_output = config.get("output_dir")
        if config_output and isinstance(config_output, str):
            final_output_dir = Path(config_output)
            logger.info(f"Using output directory from config file: {final_output_dir}")
        elif not no_gui:
            typer.echo("Output directory not specified, prompting for selection...")
            final_output_dir = prompt_for_directory(
                "Select Output Directory (for cropped files)"
            )
            if final_output_dir is None:
                logger.error("No output directory selected or provided. Exiting.")
                raise typer.Exit(code=1)
            logger.info(f"Output directory selected via dialog: {final_output_dir}")
            typer.echo(f"Using output directory: {final_output_dir}")
        else:
            msg = (
                "Error: Output directory must be provided via --output option"
                " or config file when --no-gui is used."
            )
            logger.error(msg)
            typer.secho(msg, fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)

    try:
        # Resolve output dir (don't require it to exist yet with strict=False)
        if final_output_dir is None:  # Should be caught above
            raise ValueError("Output directory could not be determined.")
        final_output_dir = final_output_dir.resolve(strict=False)
        # Check write access to the parent directory where it will be created
        parent_dir = final_output_dir.parent
        parent_dir.mkdir(parents=True, exist_ok=True)  # Ensure parent exists
        if not os.access(parent_dir, os.W_OK | os.X_OK):
            # Use warning as mkdir might have fixed it, but good to note potential issue
            logger.warning(
                f"Parent directory '{parent_dir}' might not have full"
                f" write/execute permissions needed for creating output"
                f" directory '{final_output_dir.name}'."
            )
    except Exception as e:
        msg = (
            f"Error resolving or checking parent for output directory"
            f" '{final_output_dir}': {e}"
        )
        logger.error(msg, exc_info=True)
        typer.secho(msg, fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e

    # --- Parameter Determination Logic (CLI > Config > Default) ---

    # Helper to get int from config, returning None if invalid
    def get_int_from_config(key: str) -> Optional[int]:
        value = config.get(key)
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            logger.warning(f"Invalid value '{value}' for '{key}' in config. Ignoring.")
            return None

    # Start Z
    final_start_z: Optional[int]
    if start_z is not None:
        final_start_z = start_z
        logger.debug(f"Using start_z={final_start_z} from CLI.")
    else:
        config_start_z = get_int_from_config("start_z")
        if config_start_z is not None and config_start_z >= 0:
            final_start_z = config_start_z
            logger.debug(f"Using start_z={final_start_z} from config.")
        else:
            final_start_z = 0  # Default
            if config_start_z is not None:  # Log if config value was invalid
                logger.warning(
                    f"Invalid config start_z={config_start_z}. Using default 0."
                )
            else:
                logger.debug(f"Using default start_z={final_start_z}.")

    # End Z
    final_end_z: Optional[int]
    if end_z is not None:
        final_end_z = end_z
        logger.debug(f"Using end_z={final_end_z} from CLI.")
    else:
        config_end_z = get_int_from_config("end_z")
        if config_end_z is not None and config_end_z > final_start_z:
            final_end_z = config_end_z
            logger.debug(f"Using end_z={final_end_z} from config.")
        else:
            final_end_z = None  # Default (full depth)
            if config_end_z is not None:  # Log if config value was invalid
                logger.warning(
                    f"Invalid or non-positive config end_z={config_end_z}. Using default (full depth)."
                )
            else:
                logger.debug("Using default end_z=None (full depth).")

    # Start X
    final_start_x: Optional[int]
    if start_x is not None:
        final_start_x = start_x
        logger.debug(f"Using start_x={final_start_x} from CLI.")
    else:
        config_start_x = get_int_from_config("start_x")
        if config_start_x is not None and config_start_x >= 0:
            final_start_x = config_start_x
            logger.debug(f"Using start_x={final_start_x} from config.")
        else:
            final_start_x = 0  # Default
            if config_start_x is not None:
                logger.warning(
                    f"Invalid config start_x={config_start_x}. Using default 0."
                )
            else:
                logger.debug(f"Using default start_x={final_start_x}.")

    # End X
    final_end_x: Optional[int]
    if end_x is not None:
        final_end_x = end_x
        logger.debug(f"Using end_x={final_end_x} from CLI.")
    else:
        config_end_x = get_int_from_config("end_x")
        if config_end_x is not None and config_end_x > final_start_x:
            final_end_x = config_end_x
            logger.debug(f"Using end_x={final_end_x} from config.")
        else:
            final_end_x = None  # Default (full width)
            if config_end_x is not None:
                logger.warning(
                    f"Invalid or non-positive config end_x={config_end_x}. Using default (full width)."
                )
            else:
                logger.debug("Using default end_x=None (full width).")

    # Workers parameter
    default_workers = os.cpu_count() or 4
    final_workers: int
    if workers is not None:  # CLI highest priority
        final_workers = workers  # Already validated by Typer's min=1
        logger.debug(f"Using workers={final_workers} from CLI.")
    else:
        config_workers = get_int_from_config("workers")
        if config_workers is not None and config_workers >= 1:
            final_workers = config_workers
            logger.debug(f"Using workers={final_workers} from config.")
        else:
            final_workers = default_workers
            if config_workers is not None:
                logger.warning(
                    f"Invalid config workers value ({config_workers}). Using"
                    f" default: {default_workers}"
                )
            else:
                logger.debug(f"Using default workers={final_workers}.")

    # --- Log Effective Parameters ---
    effective_z_end_str = final_end_z if final_end_z is not None else "end"
    effective_x_end_str = final_end_x if final_end_x is not None else "end"
    logger.info(
        f"Effective parameters: Input='{final_input_dir}',"
        f" Output='{final_output_dir}', Stack/Z=[{final_start_z}:{effective_z_end_str}),"
        f" X=[{final_start_x}:{effective_x_end_str}), Workers={final_workers}"
    )

    # --- Validation Parameter Logic ---
    if final_end_z is not None and final_start_z >= final_end_z:
        msg = (
            f"Error: start_z ({final_start_z}) must be less than end_z"
            f" ({final_end_z}) if end_z is specified."
        )
        logger.error(msg)
        typer.secho(msg, fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    if final_end_x is not None and final_start_x >= final_end_x:
        msg = (
            f"Error: start_x ({final_start_x}) must be less than end_x"
            f" ({final_end_x}) if end_x is specified."
        )
        logger.error(msg)
        typer.secho(msg, fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # --- Run the Main Cropping Logic ---
    try:
        run_cropping(
            input_dir=final_input_dir,
            output_dir=final_output_dir,
            start_z=final_start_z,  # Pass final determined values
            end_z=final_end_z,
            start_x=final_start_x,
            end_x=final_end_x,
            max_workers=final_workers,
        )
        logger.info("--- TIFF Cropping Tool Finished Successfully ---")
        typer.secho("Processing complete.", fg=typer.colors.GREEN)

    except typer.Exit:
        logger.warning("--- TIFF Cropping Tool Exited ---")
        raise  # Re-raise to ensure Typer handles the exit code
    except Exception as e:
        logger.critical(
            "--- TIFF Cropping Tool Failed Unexpectedly During Execution ---",
            exc_info=True,
        )
        typer.secho(
            f"An unexpected error occurred during processing: {e}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from e


# --- Main execution guard ---
if __name__ == "__main__":
    app()
