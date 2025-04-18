# src/pixelpacker/cli.py

import logging
from pathlib import Path
from typing import Optional
from typing_extensions import Annotated # Use typing_extensions for wider compatibility if needed

import typer # Import typer

# Assuming core.py is in the same package directory
from .core import run_preprocessing
from . import __version__ # Assuming you have __version__ in __init__.py

# Configure logging (can stay similar, or be enhanced by Typer callbacks if needed)
log = logging.getLogger(__name__)

# Create a Typer app instance
app = typer.Typer(
    name="pixelpacker",
    help="Processes multi-channel, multi-timepoint TIFF stacks into tiled WebP volumes with contrast stretching options.",
    add_completion=False # Disable shell completion for simplicity unless desired
)

def version_callback(value: bool):
    """Prints the version and exits."""
    if value:
        print(f"PixelPacker Version: {__version__}")
        raise typer.Exit()

# Define the main command function using Typer decorators and type hints
@app.command()
def main(
    input_folder: Annotated[Path, typer.Option(
        "--input", "-i", # Short option added
        help="Input folder containing TIFF files (e.g., *_chN_stackNNNN*.tif).",
        exists=True,      # Ensure the input directory exists
        file_okay=False,  # Ensure it's a directory, not a file
        dir_okay=True,
        readable=True,    # Ensure it's readable
        resolve_path=True,# Resolve to absolute path
    )] = Path("./Input_TIFFs"), # Default value

    output_folder: Annotated[Path, typer.Option(
        "--output", "-o", # Short option added
        help="Output folder for WebP volumes and manifest.json.",
        file_okay=False,
        dir_okay=True,
        writable=True,    # Ensure it's writable (or can be created)
        resolve_path=True,
    )] = Path("./volumes"), # Default value

    stretch_mode: Annotated[str, typer.Option(
        "--stretch", "-s",
        help="Contrast stretch method.",
        # Add validation for choices if desired
        # case_sensitive=False # Example option
    )] = "smart-late", # Default value

    global_contrast: Annotated[bool, typer.Option(
        "--global-contrast", "-g",
        help="Apply contrast range calculated globally across all timepoints."
    )] = False, # Default value (flags are False by default)

    threads: Annotated[int, typer.Option(
        "--threads", "-t",
        min=1, # Ensure at least 1 thread
        help="Number of worker threads for parallel processing."
    )] = 8, # Default value

    dry_run: Annotated[bool, typer.Option(
        "--dry-run",
        help="Simulate without reading/writing image files."
    )] = False,

    debug: Annotated[bool, typer.Option(
        "--debug",
        help="Enable detailed debug logging and save intermediate debug images."
    )] = False,

    # Typer automatically handles --version using the callback
    version: Annotated[Optional[bool], typer.Option(
        "--version", callback=version_callback, is_eager=True,
        help="Show the application's version and exit."
    )] = None,
):
    """
    Main command function executed by Typer.
    Sets up logging and calls the core processing function.
    """
    # --- Setup Logging ---
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    log.debug("Starting PixelPacker with arguments:")
    log.debug(f"  Input Folder: {input_folder}")
    log.debug(f"  Output Folder: {output_folder}")
    log.debug(f"  Stretch Mode: {stretch_mode}")
    log.debug(f"  Global Contrast: {global_contrast}")
    log.debug(f"  Threads: {threads}")
    log.debug(f"  Dry Run: {dry_run}")
    log.debug(f"  Debug: {debug}")

    # --- Prepare arguments dictionary for core function ---
    # core.run_preprocessing expects a dictionary similar to docopt's output
    # We construct it manually from the typed arguments.
    args_dict = {
        "--input": str(input_folder), # Pass paths as strings if core expects strings
        "--output": str(output_folder),
        "--stretch": stretch_mode,
        "--global-contrast": global_contrast,
        "--threads": str(threads), # Pass as string if core expects string
        "--dry-run": dry_run,
        "--debug": debug,
        # Add other keys core.run_preprocessing might expect, even if not CLI args
        # For example, if it expects --help or --version keys, add them as None/False
        "--help": False,
        "--version": False, # The callback handles the version flag itself
    }

    # --- Run Core Processing ---
    try:
        run_preprocessing(args_dict) # Pass the constructed dict
        log.info("✅ Preprocessing finished successfully.")
        # Typer handles exiting, no need for sys.exit(0) unless specific reasons
    except FileNotFoundError as e:
        # Typer's built-in validation (`exists=True`) should catch this earlier,
        # but good to keep for potential issues within core logic.
        log.error(f"❌ Error: {e}")
        raise typer.Exit(code=1) # Exit with error code using Typer's exit
    except ValueError as e:
         log.error(f"❌ Invalid configuration or data error: {e}")
         raise typer.Exit(code=1)
    except Exception as e:
        log.critical(f"❌ An unexpected critical error occurred: {e}", exc_info=True)
        raise typer.Exit(code=1)

# Standard Python entry point guard (optional with Typer app run style)
# If you want `python src/pixelpacker/cli.py` to work directly:
if __name__ == "__main__":
    app() # Run the Typer application

# However, the primary entry point will now be configured via pyproject.toml
# to call this `app` object.