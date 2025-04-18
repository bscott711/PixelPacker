# src/pixelpacker/tests/test_pipeline.py

import json
from pathlib import Path
import pytest
from typer.testing import CliRunner

from pixelpacker.cli import app # Import the Typer app

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
INPUT_DATA_DIR = PROJECT_ROOT / "Input_TIFFS"

if not INPUT_DATA_DIR.is_dir() or not list(INPUT_DATA_DIR.glob("*.tif*")):
    pytest.skip("Test input data not found or git-lfs files not pulled.", allow_module_level=True)

@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()

# ADD MARKER HERE (and to all other test functions in this file)
@pytest.mark.integration
def test_pipeline_runs_successfully(runner: CliRunner, tmp_path: Path):
    """
    Test that the pipeline runs end-to-end on the sample data
    and produces expected output files using default settings.
    """
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()

    args = [
        "--input", str(INPUT_DATA_DIR),
        "--output", str(output_dir),
    ]
    result = runner.invoke(app, args)

    print("CLI Output:\n", result.stdout)
    assert result.exit_code == 0, f"CLI exited with code {result.exit_code}\nOutput:\n{result.stdout}"

    manifest_path = output_dir / "manifest.json"
    assert manifest_path.is_file(), "manifest.json was not created"

    webp_files = list(output_dir.glob("volume_*.webp"))
    assert len(webp_files) == 10, f"Expected 10 .webp files, found {len(webp_files)}"

    try:
        with open(manifest_path, 'r') as f:
            manifest_data = json.load(f)
        assert len(manifest_data["timepoints"]) == 10
        tp0 = manifest_data["timepoints"][0]
        assert tp0["time"] == "stack0000"
        assert "files" in tp0 and "c0" in tp0["files"]
        assert tp0["files"]["c0"]["file"] == "volume_stack0000_c0.webp"
        assert "p_low" in tp0["files"]["c0"] and "p_high" in tp0["files"]["c0"]
    except Exception as e:
        pytest.fail(f"Error checking manifest content: {e}")

@pytest.mark.integration
def test_pipeline_with_global_contrast_and_debug(runner: CliRunner, tmp_path: Path):
    """Test pipeline with global contrast and debug flags enabled."""
    output_dir = tmp_path / "test_output_global_debug"
    output_dir.mkdir()

    args = [
        "--input", str(INPUT_DATA_DIR),
        "--output", str(output_dir),
        "--global-contrast",
        "--debug",
        "--stretch=imagej-auto",
    ]
    result = runner.invoke(app, args)

    assert result.exit_code == 0, f"CLI exited with code {result.exit_code} (global/debug)\nOutput:\n{result.stdout}"

    assert (output_dir / "manifest.json").is_file()
    assert len(list(output_dir.glob("volume_*.webp"))) == 10
    hist_files = list(output_dir.glob("debug_hist_*.png"))
    preview_files = list(output_dir.glob("preview_*.png"))
    assert len(hist_files) == 10, "Expected 10 debug histogram files"
    assert len(preview_files) == 10, "Expected 10 debug preview files"

