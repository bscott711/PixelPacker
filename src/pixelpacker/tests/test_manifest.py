# src/pixelpacker/tests/test_manifest.py

import json
from pathlib import Path
from typing import List

import pytest

from pixelpacker.data_models import (
    PreprocessingConfig,
    ProcessingResult,
    VolumeLayout,
)
from pixelpacker.manifest import finalize_metadata, write_manifest


# --- Fixtures ---
@pytest.fixture
def sample_layout() -> VolumeLayout:
    return VolumeLayout(
        width=100, height=80, depth=12, cols=4, rows=3, tile_width=400, tile_height=240
    )


@pytest.fixture
def sample_results() -> List[ProcessingResult]:
    return [
        ProcessingResult(
            time_id="stack0000",
            channel=0,
            filename="volume_stack0000_c0.webp",
            p_low=10.0,
            p_high=500.0,
        ),
        ProcessingResult(
            time_id="stack0001",
            channel=0,
            filename="volume_stack0001_c0.webp",
            p_low=15.0,
            p_high=550.0,
        ),
        ProcessingResult(
            time_id="stack0000",
            channel=1,
            filename="volume_stack0000_c1.webp",
            p_low=200.0,
            p_high=1000.0,
        ),
    ]


@pytest.fixture
def sample_config(tmp_path: Path) -> PreprocessingConfig:
    # Minimal config needed for manifest writing
    return PreprocessingConfig(
        input_folder=tmp_path / "input",  # Dummy paths
        output_folder=tmp_path / "output",
        stretch_mode="smart",
        z_crop_method="slope",
        z_crop_threshold=0,
        dry_run=False,
        debug=False,
        max_threads=1,
        use_global_contrast=False,  # Default case
        executor_type="process",
        input_pattern="*.tif",
    )


# --- Tests for finalize_metadata ---


@pytest.mark.unit
def test_finalize_metadata_structure(sample_results, sample_layout):
    """Verify the basic structure and content of the finalized metadata."""
    global_z = (5, 16)  # depth = 16 - 5 + 1 = 12, matches layout
    metadata = finalize_metadata(sample_results, sample_layout, global_z, None)

    assert metadata["tile_layout"]["cols"] == sample_layout.cols
    assert metadata["tile_layout"]["rows"] == sample_layout.rows
    assert metadata["volume_size"]["width"] == sample_layout.width
    assert metadata["volume_size"]["height"] == sample_layout.height
    assert metadata["volume_size"]["depth"] == sample_layout.depth
    assert metadata["channels"] == 2  # 0 and 1
    assert metadata["global_z_crop_range"] == list(global_z)
    assert len(metadata["timepoints"]) == 2  # stack0000 and stack0001

    tp0 = next(tp for tp in metadata["timepoints"] if tp["time"] == "stack0000")
    tp1 = next(tp for tp in metadata["timepoints"] if tp["time"] == "stack0001")

    assert "c0" in tp0["files"]
    assert "c1" in tp0["files"]
    assert "c0" in tp1["files"]
    assert "c1" not in tp1["files"]  # Channel 1 only exists for time 0

    assert tp0["files"]["c0"]["file"] == "volume_stack0000_c0.webp"
    assert tp0["files"]["c0"]["p_low"] == 10.0
    assert tp0["files"]["c0"]["p_high"] == 500.0

    assert tp0["files"]["c1"]["file"] == "volume_stack0000_c1.webp"
    assert tp0["files"]["c1"]["p_low"] == 200.0
    assert tp0["files"]["c1"]["p_high"] == 1000.0

    assert tp1["files"]["c0"]["file"] == "volume_stack0001_c0.webp"
    assert tp1["files"]["c0"]["p_low"] == 15.0
    assert tp1["files"]["c0"]["p_high"] == 550.0


@pytest.mark.unit
def test_finalize_metadata_global_intensity_per_image(sample_results, sample_layout):
    """Test calculation of retrospective global intensity when per-image was used."""
    metadata = finalize_metadata(sample_results, sample_layout, (0, 11), None)

    assert "c0" in metadata["global_intensity"]
    assert "c1" in metadata["global_intensity"]

    # Should be min of lows and max of highs for each channel
    assert metadata["global_intensity"]["c0"]["p_low"] == 10.0  # min(10.0, 15.0)
    assert metadata["global_intensity"]["c0"]["p_high"] == 550.0  # max(500.0, 550.0)
    assert metadata["global_intensity"]["c1"]["p_low"] == 200.0  # Only one value
    assert metadata["global_intensity"]["c1"]["p_high"] == 1000.0  # Only one value


@pytest.mark.unit
def test_finalize_metadata_global_intensity_provided(sample_results, sample_layout):
    """Test population of global intensity when global ranges were provided."""
    global_ranges = {0: (12.0, 525.0), 1: (210.0, 990.0)}
    metadata = finalize_metadata(sample_results, sample_layout, (0, 11), global_ranges)

    assert "c0" in metadata["global_intensity"]
    assert "c1" in metadata["global_intensity"]

    # Should directly reflect the provided global ranges
    assert metadata["global_intensity"]["c0"]["p_low"] == 12.0
    assert metadata["global_intensity"]["c0"]["p_high"] == 525.0
    assert metadata["global_intensity"]["c1"]["p_low"] == 210.0
    assert metadata["global_intensity"]["c1"]["p_high"] == 990.0


@pytest.mark.unit
def test_finalize_metadata_empty_results(sample_layout):
    """Test behavior with no processing results."""
    metadata = finalize_metadata([], sample_layout, (0, 11), None)

    assert metadata["channels"] == 0
    assert len(metadata["timepoints"]) == 0
    assert len(metadata["global_intensity"]) == 0
    assert metadata["global_z_crop_range"] == [0, 11]  # Still populated


# --- Tests for write_manifest ---


@pytest.mark.unit
def test_write_manifest_creates_file(sample_config, sample_results, sample_layout):
    """Test that write_manifest actually creates and populates the JSON file."""
    metadata = finalize_metadata(sample_results, sample_layout, (0, 11), None)
    manifest_path = sample_config.output_folder / "manifest.json"

    # Ensure output dir exists (usually done by CLI part)
    sample_config.output_folder.mkdir(parents=True, exist_ok=True)

    assert not manifest_path.is_file()
    write_manifest(metadata, sample_config)
    assert manifest_path.is_file()

    # Verify content
    with open(manifest_path, "r") as f:
        written_data = json.load(f)
    assert written_data["volume_size"]["width"] == 100
    assert len(written_data["timepoints"]) == 2


@pytest.mark.unit
def test_write_manifest_dry_run(sample_config, sample_results, sample_layout):
    """Test that write_manifest respects the dry_run flag."""
    sample_config.dry_run = True
    metadata = finalize_metadata(sample_results, sample_layout, (0, 11), None)
    manifest_path = sample_config.output_folder / "manifest.json"

    # Output dir might not exist in dry run
    if not sample_config.output_folder.exists():
        sample_config.output_folder.mkdir(parents=True, exist_ok=True)

    assert not manifest_path.is_file()
    write_manifest(metadata, sample_config)
    assert not manifest_path.is_file()


@pytest.mark.unit
def test_write_manifest_no_timepoints(sample_config, sample_layout):
    """Test that write_manifest skips writing if metadata has no timepoints."""
    metadata = finalize_metadata([], sample_layout, (0, 11), None)  # Empty results
    manifest_path = sample_config.output_folder / "manifest.json"

    sample_config.output_folder.mkdir(parents=True, exist_ok=True)

    assert not manifest_path.is_file()
    write_manifest(metadata, sample_config)
    assert not manifest_path.is_file()  # Should not write file


@pytest.mark.unit
def test_write_manifest_os_error(
    mocker, sample_config, sample_results, sample_layout, mock_path_mkdir
):
    """Test handling of OSError during file writing (e.g., permissions)."""
    metadata = finalize_metadata(sample_results, sample_layout, (0, 11), None)
    manifest_path = sample_config.output_folder / "manifest.json"

    # Ensure output dir exists initially
    sample_config.output_folder.mkdir(parents=True, exist_ok=True)

    # Mock open() to raise an OSError
    mock_open = mocker.patch("builtins.open", side_effect=OSError("Permission denied"))

    # write_manifest should log an error but not raise it
    write_manifest(metadata, sample_config)

    mock_open.assert_called_once_with(manifest_path, "w")
    # Check logs (if capture configured) or just ensure no exception propagated
