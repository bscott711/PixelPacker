# src/pixelpacker/cli.py

import logging
from pathlib import Path
from typing import Optional
from typing_extensions import Annotated

import typer

from .core import run_preprocessing
from . import __version__

log = logging.getLogger(__name__)
app = typer.Typer(
    name="pixelpacker",
    help="Processes multi-channel, multi-timepoint TIFF stacks into tiled WebP volumes with contrast stretching options.",
    add_completion=False,
)

def version_callback(value: bool):
    """Prints the version and exits."""
    if value:
        print(f"PixelPacker Version: {__version__}")
        raise typer.Exit()

@app.command()
def main(
    input_folder: Annotated[Path, typer.Option(
        "--input", "-i",
        help="Input folder containing TIFF files (e.g., *_chN_stackNNNN*.tif).",
        exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True,
    )] = Path("./Input_TIFFs"),

    output_folder: Annotated[Path, typer.Option(
        "--output", "-o",
        help="Output folder for WebP volumes and manifest.json.",
        file_okay=False, dir_okay=True, writable=True, resolve_path=True,
    )] = Path("./volumes"),

    stretch_mode: Annotated[str, typer.Option(
        "--stretch", "-s", help="Contrast stretch method."
    )] = "smart-late",

    # --- ADD Z-CROP OPTION ---
    z_crop_threshold: Annotated[int, typer.Option(
        "--z-crop-threshold",
        min=0, # Threshold cannot be negative
        help="Crop Z-slices where max intensity is <= this threshold. Default 0 crops only fully black slices."
    )] = 0, # Default to 0
    # --- END ADDED OPTION ---

    global_contrast: Annotated[bool, typer.Option(
        "--global-contrast / --no-global-contrast", "-g / ",
        help="Apply contrast range globally (default). Use --no-global-contrast to disable."
    )] = True,

    threads: Annotated[int, typer.Option(
        "--threads", "-t", min=1, help="Number of worker threads."
    )] = 8,

    dry_run: Annotated[bool, typer.Option(
        "--dry-run", help="Simulate without reading/writing image files."
    )] = False,

    debug: Annotated[bool, typer.Option(
        "--debug", help="Enable detailed debug logging and save intermediate images."
    )] = False,

    version: Annotated[Optional[bool], typer.Option(
        "--version", callback=version_callback, is_eager=True, help="Show version and exit."
    )] = None,
):
    """
    Main command function executed by Typer.
    Sets up logging and calls the core processing function.
    """
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Prepare arguments dictionary for core function
    args_dict = {
        "--input": str(input_folder),
        "--output": str(output_folder),
        "--stretch": stretch_mode,
        "--z-crop-threshold": z_crop_threshold, # Pass the new arg
        "--global-contrast": global_contrast,
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
