# src/pixelpacker/tests/test_pipeline.py

import json
import logging
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pixelpacker.cli import app  # Import the Typer app

# --- Constants ---
# Define base filename used in synthetic tests or expected components
BASE_TEST_FILENAME = "image.tif"
EXPECTED_SYNTH_FILENAME = f"test_ch0_stack0000_{BASE_TEST_FILENAME}"

# Path to the full integration test dataset
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
INPUT_DATA_DIR = PROJECT_ROOT / "Input_TIFFS"

# --- Path to the NEW single real image input directory ---
SINGLE_REAL_INPUT_DIR = Path(__file__).parent / "data" / "single_real_input"
# --- Define the expected output based on YOUR chosen real file name ---
# --- !!! IMPORTANT: Update this based on your actual file !!! ---
# --- Example: If your file is 'real_sample_ch0_stack0000.tif' ---
EXPECTED_SINGLE_REAL_OUTPUT_WEBP = "volume_stack0000_c0.webp"

# Module-level skip if the main data directory is missing LFS files
if not INPUT_DATA_DIR.is_dir() or not list(INPUT_DATA_DIR.glob("*.tif*")):
    pytest.skip(
        f"Full test input data not found or git-lfs files not pulled: {INPUT_DATA_DIR}",
        allow_module_level=True,
    )

log = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def runner() -> CliRunner:
    return CliRunner()


@pytest.mark.integration
def test_pipeline_runs_successfully(runner: CliRunner, tmp_path: Path):
    """
    Test that the pipeline runs end-to-end on the sample data
    and produces expected output files using default settings.
    """
    output_dir = tmp_path / "test_output_full"
    # No need to mkdir, app should handle it

    args = [
        "--input",
        str(INPUT_DATA_DIR),
        "--output",
        str(output_dir),
        # Run faster for CI/testing - adjust threads as needed
        "--threads=4",
        "--executor=thread",
    ]
    log.info(f"Running full pipeline test: input={INPUT_DATA_DIR}, output={output_dir}")
    result = runner.invoke(app, args)

    print(f"CLI Output (Full Run):\n{result.output}")
    assert result.exit_code == 0, (
        f"CLI exited with code {result.exit_code}\nOutput:\n{result.output}"
    )

    manifest_path = output_dir / "manifest.json"
    assert manifest_path.is_file(), "manifest.json was not created"

    webp_files = list(output_dir.glob("volume_*.webp"))
    # Check against the known number of files in Input_TIFFS
    assert len(webp_files) == 10, f"Expected 10 .webp files, found {len(webp_files)}"

    try:
        with open(manifest_path, "r") as f:
            manifest_data = json.load(f)
        assert len(manifest_data["timepoints"]) == 10
        # Find the tp matching stack0000 as order isn't guaranteed
        tp0_data = next(
            (tp for tp in manifest_data["timepoints"] if tp["time"] == "stack0000"),
            None,
        )
        assert tp0_data is not None, "Timepoint stack0000 not found in manifest"
        assert "files" in tp0_data and "c0" in tp0_data["files"]
        assert tp0_data["files"]["c0"]["file"] == "volume_stack0000_c0.webp"
        assert (
            "p_low" in tp0_data["files"]["c0"] and "p_high" in tp0_data["files"]["c0"]
        )
    except Exception as e:
        pytest.fail(f"Error checking manifest content: {e}")


@pytest.mark.integration
def test_pipeline_with_global_contrast_and_debug(runner: CliRunner, tmp_path: Path):
    """Test pipeline with global contrast and debug flags enabled."""
    output_dir = tmp_path / "test_output_global_debug"
    # No need to mkdir

    args = [
        "--input",
        str(INPUT_DATA_DIR),
        "--output",
        str(output_dir),
        "--global-contrast",  # Default, but explicit here
        "--debug",
        "--stretch=imagej-auto",
        # Run faster for CI/testing
        "--threads=4",
        "--executor=thread",
    ]
    log.info(
        f"Running full pipeline debug test: input={INPUT_DATA_DIR}, output={output_dir}"
    )
    result = runner.invoke(app, args)

    print(f"CLI Output (Full Run Debug):\n{result.output}")
    assert result.exit_code == 0, (
        f"CLI exited with code {result.exit_code} (global/debug)\nOutput:\n{result.output}"
    )

    assert (output_dir / "manifest.json").is_file()
    assert len(list(output_dir.glob("volume_*.webp"))) == 10
    # Debug files depend on specific processing steps, check counts based on expected output
    hist_files = list(output_dir.glob("debug_hist_*.png"))
    preview_files = list(output_dir.glob("preview_*.png"))
    # Z-crop debug files depend on method=slope (default)
    z_crop_files = list(
        output_dir.glob("*_debug_*.*")
    )  # Glob for Z-crop specific files

    assert len(hist_files) >= 10, "Expected >= 10 debug histogram files"
    # Preview only saved for channel 0 by default
    assert len(preview_files) >= 10, "Expected >= 10 debug preview files"

    # --- CORRECTED ASSERTION ---
    # Slope method produces 3 debug plots per file for 10 files = 30
    assert len(z_crop_files) >= 30, (
        f"Expected >= 30 Z-crop slope debug files, found {len(z_crop_files)}"
    )
    # --- END CORRECTION ---


# --- Test function using the single REAL file ---
@pytest.mark.integration
@pytest.mark.quick  # Add marker for easily running only this faster test
def test_pipeline_runs_single_real_file(runner: CliRunner, tmp_path: Path):
    """
    Test the pipeline runs end-to-end on a single real sample file.
    """
    output_dir = tmp_path / "single_real_output"
    # No need to mkdir

    # Check if the input directory and file exist first
    if not SINGLE_REAL_INPUT_DIR.is_dir():
        pytest.skip(f"Single real input directory not found: {SINGLE_REAL_INPUT_DIR}")
    input_files = list(SINGLE_REAL_INPUT_DIR.glob("*.tif*"))
    if not input_files:
        pytest.skip(f"No TIFF files found in: {SINGLE_REAL_INPUT_DIR}")
    if len(input_files) > 1:
        log.warning(
            f"Found multiple files in {SINGLE_REAL_INPUT_DIR}, using first: {input_files[0]}"
        )
    # TODO: Add check here based on EXPECTED_SINGLE_REAL_OUTPUT_WEBP parsing if needed

    args = [
        "--input",
        str(SINGLE_REAL_INPUT_DIR),  # Use the committed data path
        "--output",
        str(output_dir),
        "--threads=1",  # Keep it simple and fast for this test
        "--executor=thread",
    ]
    log.info(
        f"Running single real file test: input={SINGLE_REAL_INPUT_DIR}, output={output_dir}"
    )
    result = runner.invoke(app, args)

    print(f"CLI Output (Single Real File Test):\n{result.output}")
    # This test will fail until the input filename is corrected
    assert result.exit_code == 0, (
        f"CLI exited with code {result.exit_code} (single real file)\nOutput:\n{result.output}"
    )

    # Check for expected output files
    manifest_path = output_dir / "manifest.json"
    # --- IMPORTANT: Ensure this matches the file you added ---
    webp_path = output_dir / EXPECTED_SINGLE_REAL_OUTPUT_WEBP

    assert manifest_path.is_file(), "Manifest file was not created"
    assert webp_path.is_file(), (
        f"WebP file '{EXPECTED_SINGLE_REAL_OUTPUT_WEBP}' was not created"
    )

    # Optional: Basic check of manifest content based on the known single file
    try:
        with open(manifest_path, "r") as f:
            manifest_data = json.load(f)
        assert len(manifest_data["timepoints"]) == 1, (
            "Manifest should have one timepoint"
        )
        # --- IMPORTANT: Update "stack0000" and "c0" if your file is different ---
        tp_key = "stack0000"
        ch_key = "c0"
        assert manifest_data["timepoints"][0]["time"] == tp_key, (
            "Manifest timepoint key mismatch"
        )
        assert "files" in manifest_data["timepoints"][0], (
            "Manifest timepoint missing 'files'"
        )
        assert ch_key in manifest_data["timepoints"][0]["files"], (
            f"Manifest files missing channel '{ch_key}'"
        )
        assert (
            manifest_data["timepoints"][0]["files"][ch_key]["file"]
            == EXPECTED_SINGLE_REAL_OUTPUT_WEBP
        )
        assert "global_intensity" in manifest_data
        assert (
            ch_key in manifest_data["global_intensity"]
        )  # Check global intensity info exists
    except Exception as e:
        pytest.fail(f"Failed to load or validate manifest content: {e}")
