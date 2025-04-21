# src/pixelpacker/tests/test_cli.py

import yaml
import json
from pathlib import Path
import pytest
from typer.testing import CliRunner
import re

from pixelpacker.cli import app, _load_config_from_file
from pixelpacker import __version__


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(mix_stderr=False)


# --- Tests for basic CLI invocation ---
@pytest.mark.cli
def test_cli_version(runner: CliRunner):
    """Test the --version flag."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert f"PixelPacker Version: {__version__}" in result.stdout


# Regex to find ANSI escape codes
ANSI_ESCAPE_REGEX = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def strip_ansi(text: str) -> str:
    """Removes ANSI escape codes from a string."""
    return ANSI_ESCAPE_REGEX.sub("", text)


@pytest.mark.cli
def test_cli_help(runner: CliRunner):
    """Test the --help flag."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0

    # 1. Strip ANSI codes from the raw output
    output_no_ansi = strip_ansi(result.output)
    # 2. Strip leading/trailing whitespace
    output_clean = output_no_ansi.strip()

    # 3. Assert startswith on the cleaned string
    assert output_clean.startswith("Usage: main [OPTIONS]"), (
        f"Cleaned usage line did not start as expected.\nCleaned output:\n>>>\n{output_clean}\n<<<"
    )

    # 4. Check for key arguments in the CLEANED output
    assert "--input" in output_clean  # Use cleaned output
    assert "--output" in output_clean  # Use cleaned output
    assert "--help" in output_clean  # Use cleaned output


@pytest.mark.cli
def test_cli_missing_required_args(runner: CliRunner, tmp_path):
    """Test failure when required --input or --output are missing."""
    # Missing --output
    result = runner.invoke(app, ["--input", str(tmp_path)])
    assert result.exit_code != 0

    # Missing --input
    result = runner.invoke(app, ["--output", str(tmp_path)])
    assert result.exit_code != 0


@pytest.mark.cli
def test_cli_invalid_input_path(runner: CliRunner, tmp_path):
    """Test failure when --input path doesn't exist."""
    output_path = tmp_path / "output"
    output_path.mkdir()
    input_path = tmp_path / "nonexistent_input"
    result = runner.invoke(
        app, ["--input", str(input_path), "--output", str(output_path)]
    )
    assert result.exit_code != 0


# --- Tests for config file loading ---


@pytest.fixture
def sample_config_yaml(tmp_path: Path) -> Path:
    config_data = {
        "input_folder": "/path/from/yaml",
        "output_folder": str(tmp_path / "output_yaml"),  # Use tmp_path for validity
        "stretch_mode": "max",
        "max_threads": 4,
        "debug": True,
        "executor_type": "thread",
        "z_crop_method": "threshold",
        "z_crop_threshold": 15,
        "use_global_contrast": False,  # Corresponds to --per-image-contrast
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return config_path


@pytest.fixture
def sample_config_json(tmp_path: Path) -> Path:
    config_data = {
        "output_folder": str(tmp_path / "output_json"),
        "stretch_mode": "imagej-auto",
        "max_threads": 2,
        "dry_run": True,
    }
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f)
    return config_path


@pytest.mark.cli
def test_load_config_yaml(sample_config_yaml):
    """Test loading a valid YAML config file."""
    config_dict = _load_config_from_file(sample_config_yaml)
    assert config_dict["stretch_mode"] == "max"
    assert config_dict["max_threads"] == 4
    assert config_dict["debug"] is True
    assert config_dict["use_global_contrast"] is False


@pytest.mark.cli
def test_load_config_json(sample_config_json):
    """Test loading a valid JSON config file."""
    config_dict = _load_config_from_file(sample_config_json)
    assert config_dict["stretch_mode"] == "imagej-auto"
    assert config_dict["max_threads"] == 2
    assert config_dict["dry_run"] is True


@pytest.mark.cli
def test_load_config_not_found(tmp_path):
    """Test error handling when config file doesn't exist."""
    config_path = tmp_path / "not_a_config.yaml"
    with pytest.raises(FileNotFoundError):
        _load_config_from_file(config_path)


@pytest.mark.cli
def test_load_config_invalid_format(tmp_path):
    """Test error handling for invalid YAML/JSON."""
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("not: valid: yaml:")
    with pytest.raises(ValueError, match="Invalid format"):
        _load_config_from_file(config_path)

    config_path_json = tmp_path / "invalid.json"
    config_path_json.write_text("{invalid json")
    with pytest.raises(ValueError, match="Invalid format"):
        _load_config_from_file(config_path_json)


@pytest.mark.cli
def test_load_config_unsupported_extension(tmp_path):
    """Test error handling for unsupported config file extensions."""
    config_path = tmp_path / "config.txt"
    config_path.write_text("value=1")
    with pytest.raises(ValueError, match="Unsupported config file extension"):
        _load_config_from_file(config_path)


# --- Tests for CLI arguments overriding config files ---


@pytest.mark.cli
def test_cli_overrides_config(runner: CliRunner, sample_config_yaml, tmp_path, mocker):
    """Test that CLI arguments take precedence over config file settings."""
    input_dir = tmp_path / "cli_input"
    input_dir.mkdir()
    output_dir_cli = tmp_path / "cli_output"
    # Note: sample_config_yaml defines output as tmp_path / "output_yaml"

    # Mock the core processing function to just check the config
    mock_run = mocker.patch("pixelpacker.cli.run_preprocessing")

    result = runner.invoke(
        app,
        [
            "--config",
            str(sample_config_yaml),
            "--input",
            str(input_dir),  # Provide required input via CLI
            "--output",
            str(output_dir_cli),  # Override output from YAML
            "--stretch",
            "smart",  # Override stretch_mode from YAML ("max")
            "--threads",
            "6",  # Override max_threads from YAML (4)
            "--global-contrast",  # Override use_global_contrast from YAML (False)
        ],
    )

    assert result.exit_code == 0, f"CLI failed unexpectedly: {result.stdout}"
    mock_run.assert_called_once()

    # Get the config object passed to the mocked function
    call_args, call_kwargs = mock_run.call_args
    final_config = call_kwargs.get("config")
    assert final_config is not None

    # Verify overrides
    assert final_config.input_folder == input_dir.resolve()
    assert final_config.output_folder == output_dir_cli.resolve()
    assert final_config.stretch_mode == "smart"
    assert final_config.max_threads == 6
    assert final_config.use_global_contrast is True  # Set by --global-contrast flag
    assert final_config.debug is True  # This came from the YAML and wasn't overridden
    assert final_config.executor_type == "thread"  # From YAML, not overridden


@pytest.mark.cli
def test_cli_bool_flags(runner: CliRunner, tmp_path, mocker):
    """Test boolean flags (--debug, --dry-run, --per-image-contrast)."""
    input_dir = tmp_path / "input_flags"
    output_dir = tmp_path / "output_flags"
    input_dir.mkdir()
    output_dir.mkdir()  # Create dir

    mock_run = mocker.patch("pixelpacker.cli.run_preprocessing")

    # Test defaults (debug=False, dry_run=False, use_global_contrast=True)
    runner.invoke(app, ["--input", str(input_dir), "--output", str(output_dir)])
    mock_run.assert_called_once()
    config = mock_run.call_args.kwargs.get("config")
    assert config.debug is False
    assert config.dry_run is False
    assert config.use_global_contrast is True
    mock_run.reset_mock()

    # Test enabling flags
    runner.invoke(
        app,
        [
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--debug",
            "--dry-run",
            "--per-image-contrast",  # Sets use_global_contrast to False
        ],
    )
    mock_run.assert_called_once()
    config = mock_run.call_args.kwargs.get("config")
    assert config.debug is True
    assert config.dry_run is True
    assert config.use_global_contrast is False
