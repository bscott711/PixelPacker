# src/pixelpacker/cli.py

import logging
from pathlib import Path
from typing import Optional
from typing_extensions import Annotated
import enum

import typer

from .core import run_preprocessing
from . import __version__

log = logging.getLogger(__name__)
app = typer.Typer(
    name="pixelpacker",
    help="Processes multi-channel, multi-timepoint TIFF stacks into tiled WebP volumes with contrast stretching and automatic Z-cropping.",
    add_completion=False,
)


# --- Define Enum for choices ---
class ZCropMethod(str, enum.Enum):
    slope = "slope"
    threshold = "threshold"


class ExecutorChoice(str, enum.Enum):
    thread = "thread"
    process = "process"


# --- End Enum ---


def version_callback(value: bool):
    """Prints the version and exits."""
    if value:
        print(f"PixelPacker Version: {__version__}")
        raise typer.Exit()


@app.command()
def main(
    input_folder: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Input folder containing TIFF files (e.g., *_chN_stackNNNN*.tif).",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
        ),
    ] = Path("./Input_TIFFs"),
    # --- Add input_pattern Option ---
    input_pattern: Annotated[
        str,
        typer.Option(
            "--input-pattern",
            help="Glob pattern to find input TIFF files (e.g., '*.tif', 'exp1_*.tiff'). Requires '_chN_stackN' for parsing.",
        ),
    ] = "*_ch*_stack*.tif*",  # Default matches config default
    # --- End Add ---
    output_folder: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output folder for WebP volumes and manifest.json.",
            file_okay=False,
            dir_okay=True,
            writable=True,
            resolve_path=True,
        ),
    ] = Path("./volumes"),
    stretch_mode: Annotated[
        str, typer.Option("--stretch", "-s", help="Contrast stretch method.")
    ] = "smart",
    z_crop_method: Annotated[
        ZCropMethod,
        typer.Option(
            "--z-crop-method",
            case_sensitive=False,  # Allow 'slope' or 'SLOPE' etc.
            help="Method for automatic Z-cropping.",
        ),
    ] = ZCropMethod.slope,
    z_crop_threshold: Annotated[
        int,
        typer.Option(
            "--z-crop-threshold",
            min=0,
            help="Intensity threshold used ONLY if --z-crop-method=threshold.",
        ),
    ] = 0,
    # --- Add new per_image_contrast flag ---
    per_image_contrast: Annotated[
        bool,
        typer.Option(
            "--per-image-contrast",  # Flag to *enable* per-image (disable global)
            help="Use per-image contrast limits instead of the default global limits across timepoints.",
        ),
    ] = False,  # Default is False (meaning global is default)
    # --- End Add ---
    executor: Annotated[
        ExecutorChoice,
        typer.Option(
            "--executor",
            case_sensitive=False,
            help="Concurrency execution model ('thread' or 'process').",
            # Default changed to process
        ),
    ] = ExecutorChoice.process,
    threads: Annotated[
        int, typer.Option("--threads", "-t", min=1, help="Number of worker threads.")
    ] = 8,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Simulate without reading/writing image files."),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Enable detailed debug logging and save intermediate images.",
        ),
    ] = False,
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = None,
):
    """
    Main command function executed by Typer.
    Sets up logging and calls the core processing function.
    """
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log.debug("Starting PixelPacker with arguments:")
    log.debug(f"  Input Folder: {input_folder}")
    log.debug(f"  Output Folder: {output_folder}")
    log.debug(f"  Stretch Mode: {stretch_mode}")
    log.debug(f"  Z-Crop Method: {z_crop_method.value}")
    if z_crop_method == ZCropMethod.threshold:
        log.debug(f"  Z-Crop Threshold: {z_crop_threshold}")
    # --- Update args_dict generation ---
    # Calculate the effective setting for internal use based on the CLI flag
    # If --per-image-contrast is False (default), use_global_contrast is True.
    # If --per-image-contrast is True, use_global_contrast is False.
    actual_use_global_contrast = not per_image_contrast
    log.debug(f"  Per-Image Contrast Flag: {per_image_contrast}")  # Log the flag value
    log.debug(
        f"  Effective Use Global Contrast: {actual_use_global_contrast}"
    )  # Log effective setting

    log.debug(f"  Threads: {threads}")
    log.debug(f"  Executor Type: {executor.value}")
    log.debug(f"  Dry Run: {dry_run}")
    log.debug(f"  Debug: {debug}")

    # Prepare arguments dictionary for core function
    args_dict = {
        "--input": str(input_folder),
        "--input-pattern": input_pattern,
        "--output": str(output_folder),
        "--stretch": stretch_mode,
        "--z-crop-method": z_crop_method.value,
        "--z-crop-threshold": z_crop_threshold,
        "--use-global-contrast": actual_use_global_contrast,
        "--executor": executor.value,
        "--threads": str(threads),
        "--dry-run": dry_run,
        "--debug": debug,
        "--help": False,
        "--version": False,
    }

    # --- Run Core Processing ---
    try:
        run_preprocessing(args_dict)
        log.info("✅ Preprocessing finished successfully.")
    except FileNotFoundError as e:
        log.error(f"❌ Error: {e}")
        raise typer.Exit(code=1)
    except ValueError as e:
        log.error(f"❌ Invalid configuration or data error: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        log.critical(f"❌ An unexpected critical error occurred: {e}", exc_info=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
