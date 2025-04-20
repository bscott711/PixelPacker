# src/pixelpacker/cli.py

import enum
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, cast

import typer
import yaml
from typing_extensions import Annotated

# --- Core Imports ---
from .core import run_preprocessing
from .data_models import PreprocessingConfig
from . import __version__


log = logging.getLogger(__name__)
app = typer.Typer(
    name="pixelpacker",
    help="Processes multi-channel TIFFs into tiled WebP volumes.",
    add_completion=False,
)


# --- Enums ---
class ZCropMethod(str, enum.Enum):
    slope = "slope"
    threshold = "threshold"


class ExecutorChoice(str, enum.Enum):
    thread = "thread"
    process = "process"


# --- Config Loading Helper ---
def _load_config_from_file(config_path: Path) -> Dict[str, Any]:
    """Loads configuration from YAML or JSON file."""
    log.info(f"Loading configuration from file: {config_path}")
    try:
        with open(config_path, "r") as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                config_data = yaml.safe_load(f)
            elif config_path.suffix.lower() == ".json":
                config_data = json.load(f)
            else:
                raise ValueError(
                    "Unsupported config file extension:"
                    f" {config_path.suffix}. Use .yaml, .yml, or .json."
                )
            if config_data is None:
                log.warning(f"Configuration file {config_path} is empty.")
                return {}
            if not isinstance(config_data, dict):
                raise ValueError(
                    f"Configuration file {config_path} must contain"
                    " a dictionary (key-value pairs)."
                )
            log.debug(f"Config file content: {config_data}")
            return config_data
    except FileNotFoundError:
        log.error(f"Configuration file not found: {config_path}")
        raise
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        log.error(f"Error parsing configuration file {config_path}: {e}")
        raise ValueError(f"Invalid format in config file {config_path}.") from e
    except Exception as e:
        log.error(f"Unexpected error loading config file {config_path}: {e}")
        raise ValueError(f"Unexpected error loading config file {config_path}.") from e


def version_callback(value: bool):
    """Prints the version and exits."""
    if value:
        print(f"PixelPacker Version: {__version__}")
        raise typer.Exit()


@app.command()
def main(
    ctx: typer.Context,
    # --- Options MUST have defaults defined here accurately ---
    input_folder: Annotated[
        Optional[Path],
        typer.Option(
            "--input",
            "-i",
            help="Input folder containing TIFF files.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
        ),
    ] = None,  # Required, no default path makes sense
    output_folder: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output folder for WebP volumes and manifest.",
            file_okay=False,
            dir_okay=True,
            # Writable check happens later after potentially creating dir
            resolve_path=True,
        ),
    ] = None,  # Required, no default path makes sense
    config_file: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            "-c",
            help="Path to YAML or JSON configuration file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    input_pattern: Annotated[
        str, typer.Option("--input-pattern", help="Glob pattern for input TIFFs.")
    ] = "*_ch*_stack*.tif*",  # Default
    stretch_mode: Annotated[
        str,
        typer.Option(
            "--stretch",
            "-s",
            help="Contrast stretch mode (smart, max, imagej-auto, smart-late).",
        ),
    ] = "smart",  # Default
    z_crop_method: Annotated[
        ZCropMethod,
        typer.Option(
            "--z-crop-method",
            case_sensitive=False,
            help="Method for automatic Z-cropping.",
        ),
    ] = ZCropMethod.slope,  # Default
    z_crop_threshold: Annotated[
        int,
        typer.Option(
            "--z-crop-threshold",
            min=0,
            help="Intensity threshold for 'threshold' Z-crop method.",
        ),
    ] = 0,  # Default
    per_image_contrast: Annotated[
        bool,
        typer.Option(
            "--per-image-contrast/--global-contrast",
            help=(
                "Use per-image contrast limits (--per-image-contrast) or "
                "global limits across timepoints (--global-contrast)."
            ),
        ),
    ] = False,  # Default: --global-contrast is implicitly True
    executor: Annotated[
        ExecutorChoice,
        typer.Option(
            "--executor",
            case_sensitive=False,
            help="Concurrency execution model ('thread' or 'process').",
        ),
    ] = ExecutorChoice.process,  # Default
    threads: Annotated[
        int, typer.Option("--threads", "-t", min=1, help="Number of worker threads or processes.")
    ] = 8,  # Default
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Simulate processing without reading/writing image files.")
    ] = False,  # Default
    debug: Annotated[
        bool, typer.Option("--debug", help="Enable debug logging and save intermediate files.")
    ] = False,  # Default
    version: Annotated[
        Optional[bool],
        typer.Option("--version", callback=version_callback, is_eager=True, help="Show version and exit."),
    ] = None,
):
    """Main command function: Loads config, merges args, runs preprocessing."""
    # --- Initial Logging Setup ---
    initial_cli_debug_value = ctx.params.get("debug", False)
    initial_log_level = logging.DEBUG if initial_cli_debug_value else logging.INFO
    logging.basicConfig(
        level=initial_log_level,
        format="%(levelname)s: [%(name)s] %(message)s",
    )
    log.info("🚀 Starting PixelPacker Preprocessing Pipeline...")

    # --- Configuration Loading and Merging ---
    config_dict: Dict[str, Any] = {}  # Initialize empty dict
    try:
        # 1. Define Typer Option Defaults FIRST
        typer_option_defaults = {
            "input_folder": None,
            "output_folder": None,
            "input_pattern": "*_ch*_stack*.tif*",
            "stretch_mode": "smart",
            "z_crop_method": ZCropMethod.slope.value,
            "z_crop_threshold": 0,
            "per_image_contrast": False,
            "executor": ExecutorChoice.process.value,
            "threads": 8,
            "dry_run": False,
            "debug": False,
        }
        log.debug(f"Initial defaults from Typer options: {typer_option_defaults}")

        # Map CLI param names to PreprocessingConfig field names
        param_to_config_map = {
            "input_folder": "input_folder",
            "output_folder": "output_folder",
            "input_pattern": "input_pattern",
            "stretch_mode": "stretch_mode",
            "z_crop_method": "z_crop_method",
            "z_crop_threshold": "z_crop_threshold",
            "per_image_contrast": "use_global_contrast",
            "executor": "executor_type",
            "threads": "max_threads",
            "dry_run": "dry_run",
            "debug": "debug",
        }

        # 2. Start building config_dict using Typer defaults mapped to config keys
        for param_name, default_value in typer_option_defaults.items():
            config_key = param_to_config_map.get(param_name)
            if config_key:
                if param_name == "per_image_contrast":
                    config_dict["use_global_contrast"] = not default_value
                else:
                    config_dict[config_key] = default_value

        log.debug(f"Config dict after applying Typer defaults: {config_dict}")

        # 3. Load Config File -> Overrides Typer Defaults
        if config_file:
            file_values = _load_config_from_file(config_file)
            log.debug(f"Raw values loaded from file: {file_values}")
            for key, value in file_values.items():
                if key in PreprocessingConfig.__dataclass_fields__:
                    try:
                        if key in ["input_folder", "output_folder"] and isinstance(value, str):
                            config_dict[key] = Path(value)
                        elif key == "z_crop_threshold" and value is not None:
                            config_dict[key] = int(value)
                        elif key == "max_threads" and value is not None:
                            config_dict[key] = int(value)
                        elif key in ["use_global_contrast", "dry_run", "debug"] and value is not None:
                            config_dict[key] = bool(value)
                        elif key == "z_crop_method" and isinstance(value, str):
                            config_dict[key] = ZCropMethod(value).value
                        elif key == "executor_type" and isinstance(value, str):
                            config_dict["executor_type"] = ExecutorChoice(value).value
                        else:
                            config_dict[key] = value
                    except (ValueError, TypeError, KeyError) as e:
                        log.warning(
                            f"Could not convert config file value for '{key}': {value} - {e}. "
                            "Using default or previously set value."
                        )
                elif key == "executor" and "executor_type" not in config_dict:
                    try:
                        config_dict["executor_type"] = ExecutorChoice(value).value
                    except (ValueError, TypeError, KeyError) as e:
                        log.warning(
                            f"Could not convert config file value for deprecated key 'executor': {value} - {e}."
                        )
                else:
                    log.warning(f"Ignoring unknown or already handled key '{key}' from config file.")

            log.debug(f"Config dict after applying file values: {config_dict}")
        else:
            log.debug("No configuration file provided.")

        # 4. Apply CLI Arguments IF they differ from Typer defaults (as overrides)
        cli_overrides = {}
        for param_name, cli_arg_value in ctx.params.items():
            if param_name not in typer_option_defaults:
                continue

            typer_default = typer_option_defaults.get(param_name)

            # Process value before comparison (resolve Paths, get Enum value)
            processed_cli_value: Any
            if isinstance(cli_arg_value, Path):
                processed_cli_value = cli_arg_value.resolve()
                # If default is None, any path is different
                is_different = processed_cli_value is not None
            elif isinstance(cli_arg_value, enum.Enum):
                processed_cli_value = cli_arg_value.value
                is_different = processed_cli_value != typer_default
            else:
                processed_cli_value = cli_arg_value
                is_different = processed_cli_value != typer_default

            if is_different:
                config_key = param_to_config_map.get(param_name)
                if not config_key:
                    continue

                # Handle the boolean flag inversion
                if param_name == "per_image_contrast":
                    config_dict["use_global_contrast"] = not cli_arg_value
                    cli_overrides["use_global_contrast"] = not cli_arg_value
                    log.debug(
                        f"CLI override: {param_name}={cli_arg_value} -> "
                        f"use_global_contrast={config_dict['use_global_contrast']}"
                    )
                else:
                    # Apply type conversion based on dataclass annotation and store override
                    try:
                        final_typed_value: Any
                        field_type = PreprocessingConfig.__dataclass_fields__[config_key].type

                        # Handle Optional Paths and regular Paths
                        is_optional_path = str(field_type).startswith("typing.Optional[pathlib.Path") or \
                                           str(field_type).startswith("Optional[Path")
                        is_path = field_type is Path

                        if is_optional_path:
                            # Pass str or Path to Path()
                            if isinstance(processed_cli_value, (str, Path)):
                                final_typed_value = Path(processed_cli_value).resolve()
                            elif processed_cli_value is None:
                                final_typed_value = None
                            else:
                                raise TypeError(f"Expected str or Path for Optional[Path], got {type(processed_cli_value)}")
                        elif is_path:
                             # Pass str or Path to Path()
                            if isinstance(processed_cli_value, (str, Path)):
                                final_typed_value = Path(processed_cli_value).resolve()
                            else:
                                # Should be caught by required check later if None
                                raise TypeError(f"Expected str or Path for Path, got {type(processed_cli_value)}")
                        elif field_type is int:
                            # Pass int-convertible to int()
                            final_typed_value = int(processed_cli_value)
                        elif field_type is bool:
                             # Pass bool-convertible to bool()
                            final_typed_value = bool(processed_cli_value)
                        else:
                            # Assume other types (str, already converted Enum values) are directly assignable
                            final_typed_value = processed_cli_value

                        # Apply the override
                        config_dict[config_key] = final_typed_value
                        cli_overrides[config_key] = final_typed_value
                        log.debug(f"CLI override: {param_name}={cli_arg_value} -> {config_key}={final_typed_value}")

                    except (ValueError, TypeError) as e:
                        log.error(
                            f"Invalid value/type for explicitly set CLI argument {param_name}: "
                            f"'{processed_cli_value}' (from '{cli_arg_value}') - {e}"
                        )
                        raise ValueError(f"Invalid CLI value for {param_name}: {e}") from e

        if cli_overrides:
            log.info(f"Applying explicitly set CLI values (overriding file/defaults): {cli_overrides}")
        else:
            log.debug("No explicit CLI arguments provided to override defaults/file.")

        # --- Final Validation and Instantiation ---
        # Ensure required paths are set AFTER merging everything
        if not isinstance(config_dict.get("input_folder"), Path):
            if config_dict.get("input_folder") is None:
                raise ValueError("Input folder must be provided via --input or config file.")
            else:
                raise TypeError(f"Input folder must be a valid path, got: {config_dict.get('input_folder')}")

        if not isinstance(config_dict.get("output_folder"), Path):
            if config_dict.get("output_folder") is None:
                raise ValueError("Output folder must be provided via --output or config file.")
            else:
                raise TypeError(f"Output folder must be a valid path, got: {config_dict.get('output_folder')}")

        # Ensure output dir exists
        output_dir_path = cast(Path, config_dict["output_folder"])
        try:
            output_dir_path.mkdir(parents=True, exist_ok=True)
            log.info(f"Ensured output directory exists: {output_dir_path}")
        except OSError as e:
            log.error(f"Failed to create output directory {output_dir_path}: {e}")
            raise

        # 5. Create final PreprocessingConfig object
        final_config = PreprocessingConfig(**config_dict)
        log.info("Preprocessing configuration merged successfully.")
        log.debug(f"Final configuration: {final_config}")

        # --- Update Log Level based on Final Config ---
        final_log_level = logging.DEBUG if final_config.debug else logging.INFO
        current_log_level = logging.getLogger().getEffectiveLevel()
        if final_log_level != current_log_level:
            logging.getLogger().setLevel(final_log_level)
            log_format = "%(levelname)s: [%(name)s] %(message)s"
            date_format = None
            if final_config.debug:
                log_format = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
                date_format = "%H:%M:%S"
            # Use force=True to ensure handler updates
            logging.basicConfig(level=final_log_level, format=log_format, datefmt=date_format, force=True)

            log.info(
                f"Log level updated to {'DEBUG' if final_config.debug else 'INFO'} based on final config."
            )
            if final_config.debug:
                log.debug(f"Final configuration (re-logged for debug): {final_config}")

    except (ValueError, TypeError, FileNotFoundError, KeyError) as e:
        log.error(f"❌ Configuration Error: {e}")
        if config_dict.get("debug", initial_cli_debug_value):
            log.debug(f"Problematic config_dict state during error: {config_dict}")
        raise typer.Exit(code=1)
    except Exception as e:
        log.critical(f"❌ Unexpected Configuration Setup Error: {e}", exc_info=True)
        raise typer.Exit(code=1)

    # --- Run Core Processing ---
    try:
        run_preprocessing(config=final_config)
        log.info("✅ Preprocessing finished successfully.")
    except (ValueError, FileNotFoundError, OSError, TypeError, RuntimeError) as e:
        log.error(f"❌ Pipeline Error: {e}", exc_info=final_config.debug)
        raise typer.Exit(code=1)
    except Exception as e:
        log.critical(
            f"❌ An unexpected critical pipeline error occurred: {e}", exc_info=True
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()