# -*- coding: utf-8 -*-
"""
Command-line tool to efficiently crop TIFF image stacks in Z and X dimensions.
Uses a memory-efficient slice-by-slice processing approach.
"""

import concurrent.futures
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

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
    help="CLI tool to efficiently crop TIFF stacks (Z, Y, X, ...) dimensions using slice-by-slice processing.",
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
        else:  # User cancelled
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


# --- Core Cropping Logic (Memory Efficient) ---
def crop_z_x_stack_mem_efficient(
    input_path: Path,
    output_path: Path,
    z_start: Optional[int],
    z_end: Optional[int],
    x_start: Optional[int],
    x_end: Optional[int],
) -> Tuple[bool, Optional[str]]:
    """
    Reads a TIFF stack slice by slice, crops in Z/I and X, saves the result
    as a standard multi-page TIFF without explicit OME-XML generation.

    Args:
        input_path: Path to the input TIFF file.
        output_path: Path where the cropped TIFF file will be saved.
        z_start: Starting Z/I slice index (inclusive). Defaults to 0 if None.
        z_end: Ending Z/I slice index (exclusive). Defaults to stack depth if None.
        x_start: Starting X pixel index (inclusive). Defaults to 0 if None.
        x_end: Ending X pixel index (exclusive). Defaults to stack width if None.

    Returns:
        A tuple containing:
            - bool: True if cropping was successful, False otherwise.
            - Optional[str]: An error message if unsuccessful, otherwise None.
    """
    try:
        with tifffile.TiffFile(input_path) as tif:
            # --- Get Metadata and Validate ---
            if not tif.series or len(tif.series) == 0:
                msg = "Could not find image series."
                logger.warning(f"Skipping {input_path.name}: {msg}")
                return False, msg

            series = tif.series[0]
            original_shape = series.shape
            axes = series.axes.upper()

            is_valid_axes = axes.startswith("ZYX") or axes.startswith("IYX")
            if len(original_shape) < 3 or not is_valid_axes:
                msg = (
                    f"Unsupported shape {original_shape} or axes '{series.axes}'. "
                    "Requires at least 3 dimensions starting with ZYX or IYX."
                )
                logger.warning(f"Skipping {input_path.name}: {msg}")
                return False, msg

            num_z_or_i, _, width_x = original_shape[:3]

            # --- Determine Effective Crop Ranges ---
            eff_z_start = max(0, z_start if z_start is not None else 0)
            eff_z_end = min(num_z_or_i, z_end if z_end is not None else num_z_or_i)
            eff_x_start = max(0, x_start if x_start is not None else 0)
            eff_x_end = min(width_x, x_end if x_end is not None else width_x)

            if eff_z_start >= eff_z_end or eff_x_start >= eff_x_end:
                msg = (
                    f"Invalid calculated crop range resulting in zero size: "
                    f"Z/I=[{eff_z_start}:{eff_z_end}), X=[{eff_x_start}:{eff_x_end}). "
                    f"Original Z/I={num_z_or_i}, X={width_x}."
                )
                logger.warning(f"Skipping {input_path.name}: {msg}")
                return False, msg

            # --- Process Slice by Slice (No ome=True) ---
            # Initialize TiffWriter without ome=True.
            # Let tifffile create a basic multi-page TIFF.
            with tifffile.TiffWriter(output_path, bigtiff=True) as writer:
                for z_or_i_index in range(eff_z_start, eff_z_end):
                    try:
                        page = tif.pages[z_or_i_index]
                        slice_data = page.asarray()
                        # Attempt to get photometric interpretation
                        input_photometric = getattr(
                            page.tags.get("PhotometricInterpretation"),  # type: ignore
                            "value",
                            "minisblack",
                        )
                        if isinstance(input_photometric, int):
                            photometric_map = {
                                0: "miniswhite",
                                1: "minisblack",
                                2: "rgb",
                                3: "palette",
                            }
                            photometric = photometric_map.get(
                                input_photometric, "minisblack"
                            )
                        elif isinstance(input_photometric, str):
                            photometric = input_photometric.lower()
                        else:
                            photometric = "minisblack"  # Fallback
                    except IndexError:
                        msg = f"Could not read Z/I-slice index {z_or_i_index}."
                        logger.error(f"Error processing {input_path.name}: {msg}")
                        output_path.unlink(missing_ok=True)
                        return False, msg
                    except Exception as e:
                        msg = f"Error reading Z/I-slice {z_or_i_index}: {e}"
                        logger.error(
                            f"Error processing {input_path.name}: {msg}",
                            exc_info=True,
                        )
                        output_path.unlink(missing_ok=True)
                        return False, msg

                    # --- Slice Cropping ---
                    try:
                        # Crop the X dimension (index 1 of YX slice)
                        cropped_slice = slice_data[:, eff_x_start:eff_x_end, ...]
                    except IndexError as e:
                        msg = (
                            f"Error cropping slice at Z/I={z_or_i_index}. Slice shape:"
                            f" {slice_data.shape}. X-crop range:"
                            f" [{eff_x_start}:{eff_x_end}). Error: {e}"
                        )
                        logger.error(f"Error processing {input_path.name}: {msg}")
                        output_path.unlink(missing_ok=True)
                        return False, msg

                    if cropped_slice.size == 0:
                        msg = (
                            f"Cropped slice at Z/I={z_or_i_index} resulted in empty"
                            " data. Skipping write."
                        )
                        logger.warning(f"Issue processing {input_path.name}: {msg}")
                        continue

                    # --- Simple Write Call ---
                    # Write just the data slice and photometric info.
                    # Tifffile will create a standard multi-page TIFF.
                    writer.write(
                        cropped_slice,
                        photometric=photometric,
                        contiguous=True,
                    )
                    # --- END Simple Write ---

            logger.debug(
                f"Successfully wrote cropped data from {input_path.name} to {output_path.name}"
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
        msg = f"Unexpected error processing {input_path.name}: {e.__class__.__name__}: {e}"
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
        f"Starting Z={z_range_str}, X={x_range_str} crop process using up to"
        f" {max_workers} workers."
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
        msg = f"No TIFF files ({', '.join(TIFF_EXTENSIONS)}) found in the input directory: {input_dir}"
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
        help="Path to the directory containing input TIFF stacks. If omitted and --no-gui is not used, prompts for selection.",
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
        help="Path to the directory where cropped TIFF stacks will be saved. If omitted and --no-gui is not used, prompts for selection.",
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        show_default=False,
    ),
    # --- MODIFIED: start_z default is None ---
    start_z: Optional[int] = typer.Option(
        None,  # Default start Z to None initially
        "--start-z",
        "-sz",
        help="Starting Z slice index (inclusive). Defaults to 0 if not in config.",
        min=0,  # Keep validation constraint
        show_default="0 (or config)",  # Clarify default source
    ),
    end_z: Optional[int] = typer.Option(
        None,
        "--end-z",
        "-ez",
        help="Ending Z slice index (exclusive). Defaults to the end of the stack.",
        min=1,
        show_default="stack depth",
    ),
    # --- MODIFIED: start_x default is None ---
    start_x: Optional[int] = typer.Option(
        None,  # Default start X to None initially
        "--start-x",
        "-sx",
        help="Starting X pixel index (inclusive). Defaults to 0 if not in config.",
        min=0,  # Keep validation constraint
        show_default="0 (or config)",  # Clarify default source
    ),
    end_x: Optional[int] = typer.Option(
        None,
        "--end-x",
        "-ex",
        help="Ending X pixel index (exclusive). Defaults to the full width of the stack.",
        min=1,
        show_default="stack width",
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to a JSON configuration file (CLI arguments override config settings).",
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
            "Number of worker threads for parallel processing. "
            f"Defaults to system CPU count ({os.cpu_count() or 'N/A'}). "
            "Adjust based on I/O vs CPU bottleneck."
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
        help="Disable graphical directory selection prompts (requires paths via CLI or config).",
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
    Efficiently crops TIFF image stacks in Z (slice) and X (width) dimensions
    using slice-by-slice processing to minimize memory usage.

    Requires Input (--input) and Output (--output) directories, either via
    these flags, a config file, or GUI prompts (unless --no-gui is used).

    Examples:
        # Specify paths via CLI, crop Z 10-49, X 100-599
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
                "%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)
            logging.getLogger().addHandler(file_handler)
            logger.info(f"Logging detailed output (DEBUG level) to: {log_file}")
        except Exception as e:
            typer.secho(
                f"Warning: Could not configure log file at {log_file}: {e}. Continuing without file logging.",
                fg=typer.colors.YELLOW,
                err=True,
            )
            logger.error(f"Failed to configure log file handler: {e}", exc_info=True)

    logger.info("--- TIFF Cropping Tool Started ---")
    logger.debug(
        f"Raw arguments: input={input_dir}, output={output_dir}, sz={start_z}, ez={end_z}, sx={start_x}, ex={end_x}, config={config_file}, workers={workers}, log={log_file}, no_gui={no_gui}, verbose={verbose}"
    )

    # --- Load Configuration File ---
    config: Dict[str, Any] = {}
    if config_file:
        if not config_file.is_file():
            msg = f"Error: Specified config file not found or is not a file: {config_file}"
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
    # ... (Input/Output directory determination logic remains the same) ...
    final_input_dir = input_dir
    if final_input_dir is None:
        config_input = config.get("input_dir")
        if config_input:
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
            msg = "Error: Input directory must be provided via --input option or config file when --no-gui is used."
            logger.error(msg)
            typer.secho(msg, fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)

    try:
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
        if config_output:
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
            msg = "Error: Output directory must be provided via --output option or config file when --no-gui is used."
            logger.error(msg)
            typer.secho(msg, fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)

    try:
        final_output_dir = final_output_dir.resolve(strict=False)
        parent_dir = final_output_dir.parent
        if not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)
        if not os.access(parent_dir, os.W_OK | os.X_OK):
            logger.warning(
                f"Parent directory '{parent_dir}' might not be writable/accessible."
            )
    except Exception as e:
        msg = f"Error resolving or checking parent for output directory '{final_output_dir}': {e}"
        logger.error(msg, exc_info=True)
        typer.secho(msg, fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from e

    # --- CORRECTED Parameter Determination Logic ---
    # Priority: CLI > Config > Default(0 for start, None for end)

    # Start Z
    if start_z is not None:  # Check if CLI value was provided (Typer default is None)
        final_start_z = start_z
        logger.debug(f"Using start_z={final_start_z} from CLI.")
    elif "start_z" in config:  # Check config if CLI was None
        final_start_z = config.get("start_z")
        # Validate config value type
        if not isinstance(final_start_z, int) or final_start_z < 0:
            logger.warning(
                f"Invalid 'start_z' in config ({final_start_z}). Using default 0."
            )
            final_start_z = 0
        else:
            logger.debug(f"Using start_z={final_start_z} from config.")
    else:  # Use default 0 if neither CLI nor config provided it
        final_start_z = 0
        logger.debug(f"Using default start_z={final_start_z}.")

    # End Z
    if end_z is not None:  # Check if CLI value was provided
        final_end_z = end_z
        logger.debug(f"Using end_z={final_end_z} from CLI.")
    else:  # Check config if CLI was None
        final_end_z = config.get("end_z")  # Can be None if not in config
        # Validate config value type
        if final_end_z is not None and (
            not isinstance(final_end_z, int) or final_end_z <= final_start_z
        ):
            logger.warning(
                f"Invalid 'end_z' in config ({final_end_z}) or not > start_z. Treating as unspecified."
            )
            final_end_z = None
        elif final_end_z is None:
            logger.debug("Using default end_z=None (full depth) from config/default.")
        else:
            logger.debug(f"Using end_z={final_end_z} from config.")

    # Start X (similar logic to Start Z)
    if start_x is not None:  # Check if CLI value was provided (Typer default is None)
        final_start_x = start_x
        logger.debug(f"Using start_x={final_start_x} from CLI.")
    elif "start_x" in config:  # Check config if CLI was None
        final_start_x = config.get("start_x")
        # Validate config value type
        if not isinstance(final_start_x, int) or final_start_x < 0:
            logger.warning(
                f"Invalid 'start_x' in config ({final_start_x}). Using default 0."
            )
            final_start_x = 0
        else:
            logger.debug(f"Using start_x={final_start_x} from config.")
    else:  # Use default 0 if neither CLI nor config provided it
        final_start_x = 0
        logger.debug(f"Using default start_x={final_start_x}.")

    # End X (similar logic to End Z)
    if end_x is not None:  # Check if CLI value was provided
        final_end_x = end_x
        logger.debug(f"Using end_x={final_end_x} from CLI.")
    else:  # Check config if CLI was None
        final_end_x = config.get("end_x")  # Can be None if not in config
        # Validate config value type
        if final_end_x is not None and (
            not isinstance(final_end_x, int) or final_end_x <= final_start_x
        ):
            logger.warning(
                f"Invalid 'end_x' in config ({final_end_x}) or not > start_x. Treating as unspecified."
            )
            final_end_x = None
        elif final_end_x is None:
            logger.debug("Using default end_x=None (full width) from config/default.")
        else:
            logger.debug(f"Using end_x={final_end_x} from config.")

    # Workers parameter (refined logic)
    default_workers = os.cpu_count() or 4
    final_workers = default_workers  # Start with default

    if workers is not None:  # CLI highest priority
        final_workers = workers  # Already validated by Typer's min=1
        logger.debug(f"Using workers={final_workers} from CLI.")
    elif "workers" in config:  # Config next
        config_workers = config.get("workers")
        # --- ADDED Check for None before try block ---
        if config_workers is not None:
            try:
                # Attempt conversion to int
                parsed_config_workers = int(
                    config_workers
                )  # Now None case is explicitly handled before this line
                if parsed_config_workers >= 1:
                    final_workers = parsed_config_workers
                    logger.debug(f"Using workers={final_workers} from config.")
                else:
                    # Value is int but < 1
                    logger.warning(
                        f"Config 'workers' value ({config_workers}) is invalid (< 1). Using default: {default_workers}"
                    )
                    # final_workers already set to default_workers
            except (ValueError, TypeError):
                # Value is not None but cannot be converted to int
                logger.warning(
                    f"Config 'workers' value ('{config_workers}') is not a valid integer. Using default: {default_workers}"
                )
                # final_workers already set to default_workers
        else:
            # Handle case where config has "workers": null
            logger.warning(
                f"Config 'workers' value is null. Using default: {default_workers}"
            )
            # final_workers already set to default_workers
        # --- END Added Check ---
    else:  # Neither CLI nor config key exists
        logger.debug(f"Using default workers={final_workers}.")

    # --- Log Effective Parameters ---
    # This log message now reflects the correctly determined final values
    effective_z_end_str = final_end_z if final_end_z is not None else "end"
    effective_x_end_str = final_end_x if final_end_x is not None else "end"
    logger.info(
        f"Effective parameters: Input='{final_input_dir}',"
        f" Output='{final_output_dir}', Z=[{final_start_z}:{effective_z_end_str}),"
        f" X=[{final_start_x}:{effective_x_end_str}), Workers={final_workers}"
    )

    # --- Validation Parameter Logic ---
    # Add type ignores as Pylance might still complain after None checks
    if final_end_z is not None and final_start_z >= final_end_z:  # type: ignore[operator]
        msg = (
            f"Error: start_z ({final_start_z}) must be less than end_z"
            f" ({final_end_z}) if end_z is specified."
        )
        logger.error(msg)
        typer.secho(msg, fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    if final_end_x is not None and final_start_x >= final_end_x:  # type: ignore[operator]
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
        raise
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
