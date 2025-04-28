# src/pixelpacker/cli.py

import enum
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import cProfile
import pstats
import io

import typer
import click
import yaml
from typing_extensions import Annotated

# --- Core Imports ---
from . import __version__
from .core import run_preprocessing
from .data_models import PreprocessingConfig  # Keep existing import

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
# _load_config_from_file remains the same


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
    except ValueError as e:
        log.error(f"Configuration value error in {config_path}: {e}")
        raise
    except Exception as e:
        log.error(
            f"Unexpected error loading config file {config_path}: {e}", exc_info=True
        )
        raise RuntimeError(
            f"Unexpected error loading config file {config_path}."
        ) from e


# --- Version Callback ---
def version_callback(value: bool):
    """Prints the version and exits."""
    if value:
        print(f"PixelPacker Version: {__version__}")
        raise typer.Exit()


# === Main CLI Command ===
@app.command()
def main(  # noqa: PLR0913
    ctx: typer.Context,
    # --- Input/Output ---
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
    ] = None,
    output_folder: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output folder for WebP volumes and manifest.",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
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
    # --- Processing Parameters ---
    input_pattern: Annotated[
        str, typer.Option("--input-pattern", help="Glob pattern for input TIFFs.")
    ] = "*_ch*_stack*.tif*",
    stretch_mode: Annotated[
        str,
        typer.Option(
            "--stretch",
            "-s",
            help="Contrast stretch mode (smart, max, imagej-auto, smart-late).",
        ),
    ] = "smart",
    # --- Z-Crop Options --- <<< MODIFIED SECTION START
    enable_z_crop: Annotated[
        bool,
        typer.Option(
            "--enable-z-crop/--disable-z-crop",
            help="Explicitly enable/disable Z-cropping (default is disabled).",
        ),
    ] = False,  # Default is False
    z_crop_method: Annotated[
        ZCropMethod,
        typer.Option(
            "--z-crop-method",
            case_sensitive=False,
            help="Method for automatic Z-cropping (only used if --enable-z-crop is set).",
        ),
    ] = ZCropMethod.slope,
    z_crop_threshold: Annotated[
        int,
        typer.Option(
            "--z-crop-threshold",
            min=0,
            help="Intensity threshold for 'threshold' Z-crop method (only used if --enable-z-crop is set).",
        ),
    ] = 0,
    # --- END Z-Crop Options --- <<< MODIFIED SECTION END
    per_image_contrast: Annotated[
        bool,
        typer.Option(
            "--per-image-contrast/--global-contrast",
            help="Use per-image contrast limits (--per-image-contrast) or global limits across timepoints (--global-contrast).",
        ),
    ] = False,
    # --- Execution Control ---
    executor: Annotated[
        ExecutorChoice,
        typer.Option(
            "--executor",
            case_sensitive=False,
            help="Concurrency execution model ('thread' or 'process').",
        ),
    ] = ExecutorChoice.process,
    threads: Annotated[
        int,
        typer.Option(
            "--threads", "-t", min=1, help="Number of worker threads or processes."
        ),
    ] = 8,
    # --- Flags ---
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", help="Simulate processing without reading/writing image files."
        ),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug", help="Enable debug logging and save intermediate files."
        ),
    ] = False,
    profile: Annotated[
        bool,
        typer.Option(
            "--profile",
            help="Enable cProfile for performance analysis (adds overhead).",
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
    """Main command function: Loads config, merges args, runs preprocessing."""
    # --- Initial Logging Setup ---
    # (remains the same)
    initial_cli_debug_value = ctx.params.get("debug", False)
    initial_log_level = logging.DEBUG if initial_cli_debug_value else logging.INFO
    if ctx.params.get("profile", False):
        initial_log_level = logging.INFO

    logging.basicConfig(
        level=initial_log_level,
        format="%(levelname)s: [%(name)s] %(message)s",
    )
    log.info("🚀 Starting PixelPacker Preprocessing Pipeline...")
    profile_flag = ctx.params.get("profile", False)  # Get profile flag status
    if profile_flag:
        log.info("⏳ Profiling enabled. This will add overhead.")

    # --- Configuration Loading and Merging ---
    config_dict: Dict[str, Any] = {}
    final_config: Optional[PreprocessingConfig] = None
    try:
        # 1. Initialize with Typer Defaults
        typer_param_defs = {p.name: p for p in ctx.command.params if p.name is not None}
        # --- ADD 'enable_z_crop' to mapping ---
        param_to_config_map = {
            "input_folder": "input_folder",
            "output_folder": "output_folder",
            "input_pattern": "input_pattern",
            "stretch_mode": "stretch_mode",
            "enable_z_crop": "enable_z_crop",  # <<< ADDED
            "z_crop_method": "z_crop_method",
            "z_crop_threshold": "z_crop_threshold",
            "per_image_contrast": "use_global_contrast",
            "executor": "executor_type",
            "threads": "max_threads",
            "dry_run": "dry_run",
            "debug": "debug",
        }

        log.debug("Step 1: Initializing config with Typer defaults...")
        for param_name, param_def in typer_param_defs.items():
            config_key = param_to_config_map.get(param_name)
            if not config_key:
                continue

            default_value = param_def.default
            processed_default: Any = default_value

            if isinstance(default_value, enum.Enum):
                processed_default = default_value.value
            # Removed Path processing here, handle later if needed

            # Special handling for boolean flags mapped differently
            if param_name == "per_image_contrast":
                # Default value is False, so use_global_contrast becomes True
                config_dict["use_global_contrast"] = not processed_default
            # --- ADDED handling for enable_z_crop ---
            elif param_name == "enable_z_crop":
                # Default value is False, matches config field directly
                config_dict["enable_z_crop"] = processed_default
            else:
                # Other parameters map directly
                config_dict[config_key] = processed_default
        log.debug(f"Config dict after Typer defaults: {config_dict}")

        # 2. Load Config File -> Overrides Defaults
        log.debug("Step 2: Loading and merging config file...")
        if config_file:
            file_values = _load_config_from_file(config_file)
            log.debug(f"Raw values loaded from file: {file_values}")
            for key, value in file_values.items():
                if key in PreprocessingConfig.__dataclass_fields__:
                    try:
                        # Type conversions (ensure bools are handled correctly)
                        if key in ["input_folder", "output_folder"] and isinstance(
                            value, str
                        ):
                            config_dict[key] = Path(value)  # Resolve later
                        elif key == "z_crop_threshold" and value is not None:
                            config_dict[key] = int(value)
                        elif key == "max_threads" and value is not None:
                            config_dict[key] = int(value)
                        elif (
                            key
                            in [
                                "use_global_contrast",
                                "dry_run",
                                "debug",
                                "enable_z_crop",
                            ]  # <<< ADDED enable_z_crop
                            and value is not None
                        ):
                            # Explicitly handle boolean conversion robustly
                            if isinstance(value, bool):
                                config_dict[key] = value
                            elif isinstance(value, str):
                                config_dict[key] = value.lower() in [
                                    "true",
                                    "1",
                                    "yes",
                                    "on",
                                ]
                            elif isinstance(value, int):
                                config_dict[key] = bool(value)
                            else:
                                log.warning(
                                    f"Could not parse boolean value for '{key}': {value}. Ignoring."
                                )
                        elif key == "z_crop_method" and isinstance(value, str):
                            config_dict[key] = ZCropMethod(value).value
                        elif key == "executor_type" and isinstance(value, str):
                            config_dict[key] = ExecutorChoice(value).value
                        elif (
                            value is not None
                        ):  # Assign other types directly if not None
                            config_dict[key] = value
                        else:
                            log.debug(
                                f"Ignoring null value for '{key}' in config file."
                            )
                    except (ValueError, TypeError, KeyError) as e:
                        log.warning(
                            f"Could not convert config file value for '{key}': {value} - {e}."
                        )
                # Handle deprecated 'executor' key (remains same)
                elif key == "executor" and "executor_type" not in file_values:
                    try:
                        if isinstance(value, str):
                            config_dict["executor_type"] = ExecutorChoice(value).value
                            log.warning(
                                "Using deprecated config key 'executor'. Mapped to 'executor_type'."
                            )
                        else:
                            raise TypeError("Value for 'executor' must be a string.")
                    except (ValueError, TypeError, KeyError) as e:
                        log.warning(
                            f"Could not convert deprecated 'executor': {value} - {e}."
                        )
                else:
                    log.warning(f"Ignoring unknown key '{key}' from config file.")

            log.debug(f"Config dict after applying file values: {config_dict}")
        else:
            log.debug("No configuration file provided.")

        # 3. Apply Explicit CLI Arguments -> Overrides File/Defaults
        log.debug("Step 3: Applying explicit CLI arguments...")
        cli_applied_values: Dict[str, Any] = {}
        for param_name, cli_arg_value in ctx.params.items():
            if param_name in ["config_file", "profile", "version"]:
                continue
            source = ctx.get_parameter_source(param_name)
            if source != click.core.ParameterSource.COMMANDLINE:
                log.debug(
                    f"Skipping '{param_name}': not provided via CLI (source={source})"
                )
                continue

            config_key = param_to_config_map.get(param_name)
            if config_key is None:
                log.warning(
                    f"CLI parameter '{param_name}' has no config mapping, skipping override."
                )
                continue

            target_config_key = config_key
            processed_cli_value = cli_arg_value

            # Special handling for boolean flags mapped differently
            if param_name == "per_image_contrast":
                processed_cli_value = not cli_arg_value
                target_config_key = "use_global_contrast"
            elif param_name == "enable_z_crop":
                # This flag maps directly, no special processing needed for value
                target_config_key = "enable_z_crop"

            # Attempt type conversion/validation (simplified)
            try:
                final_value = processed_cli_value
                if isinstance(processed_cli_value, enum.Enum):
                    final_value = processed_cli_value.value
                elif isinstance(processed_cli_value, Path):
                    final_value = processed_cli_value  # Resolve later

                if config_dict.get(target_config_key) != final_value:
                    config_dict[target_config_key] = final_value
                    cli_applied_values[target_config_key] = final_value
            except Exception as e:
                log.error(f"Invalid CLI value for {param_name}: {e}")
                raise

        if cli_applied_values:
            log.info("Applied CLI overrides: %s", cli_applied_values)

        # 4. Final Validation and Instantiation
        log.debug("Step 4: Final validation and instantiation...")
        input_val = config_dict.get("input_folder")
        output_val = config_dict.get("output_folder")

        if not input_val:
            raise ValueError("Input folder is required.")
        if not output_val:
            raise ValueError("Output folder is required.")

        # Ensure paths are resolved *before* creating the config object
        input_path = Path(input_val).resolve()
        output_path = Path(output_val).resolve()
        config_dict["input_folder"] = input_path
        config_dict["output_folder"] = output_path

        if not input_path.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {input_path}")

        # Check output dir writability
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            dummy_file = output_path / ".pixelpacker_write_test"
            dummy_file.touch()
            dummy_file.unlink()
            log.info(f"Ensured output directory exists and is writable: {output_path}")
        except OSError as e:
            log.error(
                f"Failed to create or write to output directory {output_path}: {e}"
            )
            raise PermissionError(
                f"Cannot write to output directory: {output_path}"
            ) from e

        # 5. Create final PreprocessingConfig object
        valid_keys = PreprocessingConfig.__dataclass_fields__.keys()
        final_config_data = {k: v for k, v in config_dict.items() if k in valid_keys}
        log.debug(f"Final data passed to PreprocessingConfig: {final_config_data}")
        try:
            final_config = PreprocessingConfig(**final_config_data)
        except TypeError as e:
            log.error(
                f"Configuration Error: Missing/incorrect arguments for PreprocessingConfig: {e}"
            )
            raise ValueError(
                f"Missing/incorrect configuration arguments. Details: {e}"
            ) from e

        log.info("Preprocessing configuration merged successfully.")
        log.debug(f"Final configuration object: {final_config}")

        # --- Update Log Level based on Final Config ---
        if not profile_flag:
            final_log_level = logging.DEBUG if final_config.debug else logging.INFO
            current_log_level = logging.getLogger().getEffectiveLevel()
            if final_log_level != current_log_level:
                # Reset logging config with potentially new level and format
                log_format = "%(levelname)s: [%(name)s] %(message)s"
                date_format = None
                if final_config.debug:
                    log_format = (
                        "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
                    )
                    date_format = "%H:%M:%S"

                # Use force=True to replace existing handlers
                logging.basicConfig(
                    level=final_log_level,
                    format=log_format,
                    datefmt=date_format,
                    force=True,
                )
                log.info(
                    f"Log level updated to {'DEBUG' if final_config.debug else 'INFO'}."
                )
                if final_config.debug:
                    log.debug(
                        f"Final configuration (re-logged for debug): {final_config}"
                    )

    except (ValueError, TypeError, FileNotFoundError, KeyError, PermissionError) as e:
        log.error("❌ Configuration Error: %s", e)
        if initial_cli_debug_value:
            log.debug(f"Problematic config_dict state during error: {config_dict}")
        raise typer.Exit(code=1)
    except Exception as e:
        log.critical(f"❌ Unexpected Configuration Setup Error: {e}", exc_info=True)
        raise typer.Exit(code=1)

    if final_config is None:
        log.critical("❌ Configuration object was not created successfully.")
        raise typer.Exit(code=1)

    # --- Run Core Processing (Potentially Profiled) ---
    # (profiler logic remains the same)
    profiler = None
    if profile_flag:  # Use stored flag value
        profiler = cProfile.Profile()
        log.debug("Starting profiler...")
        profiler.enable()

    try:
        run_preprocessing(config=final_config)
        log.info("✅ Preprocessing finished successfully.")
    except (ValueError, FileNotFoundError, OSError, TypeError, RuntimeError) as e:
        log.error(f"❌ Pipeline Error: {e}", exc_info=final_config.debug)
        if profiler:
            log.debug("Stopping profiler after pipeline error...")
            profiler.disable()
        raise typer.Exit(code=1)
    except Exception as e:
        log.critical(
            f"❌ An unexpected critical pipeline error occurred: {e}", exc_info=True
        )
        if profiler:
            log.debug("Stopping profiler after critical pipeline error...")
            profiler.disable()
        raise typer.Exit(code=1)
    finally:
        if profiler:
            log.debug("Stopping profiler...")
            profiler.disable()

            log.info("📊 Processing profiler results...")
            stats_stream = io.StringIO()
            ps = pstats.Stats(profiler, stream=stats_stream).sort_stats("cumulative")
            ps.print_stats(30)
            log.info(f"Profiler Stats (Top 30 Cumulative):\n{stats_stream.getvalue()}")

            profile_file = "pixelpacker_profile.prof"
            try:
                profiler.dump_stats(profile_file)
                log.info(f"Full profiler stats saved to: {profile_file}")
                log.info(f"Use 'snakeviz {profile_file}' to visualize.")
            except Exception as dump_e:
                log.error(f"Failed to dump profiler stats to {profile_file}: {dump_e}")


if __name__ == "__main__":
    app()
