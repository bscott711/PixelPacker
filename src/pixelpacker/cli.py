# src/pixelpacker/cli.py

import enum
import json
import logging
import typing
from pathlib import Path
from typing import Any, Dict, Optional, Union
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
from .data_models import PreprocessingConfig

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
    z_crop_method: Annotated[
        ZCropMethod,
        typer.Option(
            "--z-crop-method",
            case_sensitive=False,
            help="Method for automatic Z-cropping.",
        ),
    ] = ZCropMethod.slope,
    z_crop_threshold: Annotated[
        int,
        typer.Option(
            "--z-crop-threshold",
            min=0,
            help="Intensity threshold for 'threshold' Z-crop method.",
        ),
    ] = 0,
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

        log.debug("Step 1: Initializing config with Typer defaults...")
        for param_name, param_def in typer_param_defs.items():
            config_key = param_to_config_map.get(param_name)
            if not config_key:
                continue

            default_value = param_def.default
            processed_default: Any = default_value

            if isinstance(default_value, enum.Enum):
                processed_default = default_value.value
            elif isinstance(default_value, Path):
                processed_default = default_value.resolve()

            if param_name == "per_image_contrast":
                config_dict["use_global_contrast"] = not processed_default
            else:
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
                        if key in ["input_folder", "output_folder"] and isinstance(
                            value, str
                        ):
                            config_dict[key] = Path(value).resolve()
                        elif key == "z_crop_threshold" and value is not None:
                            config_dict[key] = int(value)
                        elif key == "max_threads" and value is not None:
                            config_dict[key] = int(value)
                        elif (
                            key in ["use_global_contrast", "dry_run", "debug"]
                            and value is not None
                        ):
                            config_dict[key] = bool(value)
                        elif key == "z_crop_method" and isinstance(value, str):
                            config_dict[key] = ZCropMethod(value).value
                        elif key == "executor_type" and isinstance(value, str):
                            config_dict[key] = ExecutorChoice(value).value
                        elif value is not None:
                            config_dict[key] = value
                        else:
                            log.debug(
                                f"Ignoring null value for '{key}' in config file."
                            )
                    except (ValueError, TypeError, KeyError) as e:
                        log.warning(
                            f"Could not convert config file value for '{key}': {value} - {e}."
                        )
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
            # Skip non-config params or params not explicitly set on CLI
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

            if param_name == "per_image_contrast":
                processed_cli_value = not cli_arg_value
                target_config_key = "use_global_contrast"

            try:
                field = PreprocessingConfig.__dataclass_fields__[target_config_key].type
                origin = typing.get_origin(field)
                args = typing.get_args(field)
                is_opt = origin is Union and type(None) in args
                is_path = field is Path or (is_opt and Path in args)

                final: Any  # Define final variable
                if isinstance(cli_arg_value, enum.Enum):
                    final = cli_arg_value.value
                elif is_path:
                    final = (
                        None if cli_arg_value is None else Path(cli_arg_value).resolve()
                    )
                elif field is int and cli_arg_value is not None:
                    final = int(cli_arg_value)
                elif field is bool and cli_arg_value is not None:
                    if param_name != "per_image_contrast":
                        final = bool(cli_arg_value)
                    else:
                        final = processed_cli_value
                elif is_opt and cli_arg_value is None:
                    final = None
                else:
                    final = cli_arg_value

                processed_cli_value = final

            except Exception as e:
                log.error(f"Invalid CLI value for {param_name}: {e}")
                raise

            if config_dict.get(target_config_key) != processed_cli_value:
                config_dict[target_config_key] = processed_cli_value
                cli_applied_values[target_config_key] = processed_cli_value

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

        if not isinstance(input_val, Path):
            raise TypeError(
                f"Input folder must be a valid path, got: {type(input_val)}"
            )
        if not isinstance(output_val, Path):
            raise TypeError(
                f"Output folder must be a valid path, got: {type(output_val)}"
            )

        output_dir_path = output_val
        try:
            output_dir_path.mkdir(parents=True, exist_ok=True)
            dummy_file = output_dir_path / ".pixelpacker_write_test"
            dummy_file.touch()
            dummy_file.unlink()
            log.info(
                f"Ensured output directory exists and is writable: {output_dir_path}"
            )
        except OSError as e:
            log.error(
                f"Failed to create or write to output directory {output_dir_path}: {e}"
            )
            raise PermissionError(
                f"Cannot write to output directory: {output_dir_path}"
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
        if not profile_flag:  # Check actual flag value from ctx.params
            final_log_level = logging.DEBUG if final_config.debug else logging.INFO
            current_log_level = logging.getLogger().getEffectiveLevel()
            if final_log_level != current_log_level:
                logging.getLogger().setLevel(final_log_level)
                log_format = "%(levelname)s: [%(name)s] %(message)s"
                date_format = None
                if final_config.debug:
                    log_format = (
                        "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
                    )
                    date_format = "%H:%M:%S"
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
        # Removed unused variable assignment
        if profiler:
            log.debug("Stopping profiler after pipeline error...")
            profiler.disable()
        raise typer.Exit(code=1)
    except Exception as e:
        log.critical(
            f"❌ An unexpected critical pipeline error occurred: {e}", exc_info=True
        )
        # Removed unused variable assignment
        if profiler:
            log.debug("Stopping profiler after critical pipeline error...")
            profiler.disable()
        raise typer.Exit(code=1)
    finally:
        # Stop profiler if it was started
        if profiler:  # Removed the .is_enabled() check
            log.debug("Stopping profiler...")  # Simplified log message
            profiler.disable()

            # Process and print profiler stats
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
