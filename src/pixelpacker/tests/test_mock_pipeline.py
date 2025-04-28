# src/pixelpacker/tests/test_mock_pipeline.py
import pytest
from typer.testing import CliRunner  # Keep for mkdir test
import logging

# Import the core function and config dataclass directly
from pixelpacker.cli import app  # Keep for mkdir test
from pixelpacker.core import run_preprocessing
from pixelpacker.data_models import PreprocessingConfig

BASE_TEST_FILENAME = "image.tif"
# EXPECTED_TEST_FILENAME = f"test_ch0_stack0000_{BASE_TEST_FILENAME}" # Unused


@pytest.mark.mock
def test_pipeline_handles_output_write_error(mocker, tmp_path, synthetic_tiff_factory):
    """Test pipeline failure when saving WebP raises OSError."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    # Ensure output dir exists for the run_preprocessing call
    output_dir.mkdir(parents=True, exist_ok=True)

    tiff_path = synthetic_tiff_factory(
        input_dir, BASE_TEST_FILENAME, shape=(10, 20, 20)
    )
    assert tiff_path.exists()

    # Mock PIL save function
    mock_save = mocker.patch("PIL.Image.Image.save", autospec=True)
    mock_save.side_effect = OSError("Disk full")

    # Create config manually
    config = PreprocessingConfig(
        input_folder=input_dir,
        output_folder=output_dir,
        stretch_mode="smart",
        enable_z_crop=False,
        z_crop_method="slope",
        z_crop_threshold=0,
        dry_run=False,
        debug=False,
        max_threads=1,
        use_global_contrast=True,
        executor_type="thread",
        input_pattern="*_ch*_stack*.tif*",
    )

    # Expect run_preprocessing to raise RuntimeError due to error propagation in Pass 2
    # Note: Output errors often manifest as RuntimeErrors from the processing stage
    with pytest.raises(RuntimeError) as excinfo:
        run_preprocessing(config=config)

    # Assert mock was called (Pass 2 is reached before PIL save error)
    try:
        mock_save.assert_called()
    except AssertionError as e:
        pytest.fail(f"PIL.Image.Image.save mock was not called! {e}")

    # Assert the exception message contains info about the processing error
    # The specific error might be wrapped, check for common indicators
    assert "Disk full" in str(excinfo.value) or "processing error" in str(excinfo.value)


@pytest.mark.mock
def test_pipeline_handles_input_read_error(mocker, tmp_path, synthetic_tiff_factory):
    """Test pipeline failure when reading TIFF (metadata) raises an error."""
    # Mock the *metadata reading function* which is called first in Pass 0
    mock_get_dims = mocker.patch("pixelpacker.crop.get_dimensions_from_metadata")
    mock_get_dims.side_effect = ValueError("Invalid TIFF data (metadata read fail)")

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a dummy file path for the metadata reader to attempt
    tiff_path = synthetic_tiff_factory(
        input_dir, BASE_TEST_FILENAME, shape=(10, 20, 20)
    )
    assert tiff_path.exists()

    # Create config manually (default enable_z_crop=False)
    config = PreprocessingConfig(
        input_folder=input_dir,
        output_folder=output_dir,
        stretch_mode="smart",
        z_crop_method="slope",
        z_crop_threshold=0,
        dry_run=False,
        debug=False,
        max_threads=1,
        use_global_contrast=True,
        executor_type="thread",
        input_pattern="*_ch*_stack*.tif*",
    )

    # Expect run_preprocessing to raise an error stemming from the metadata read failure
    # The original ValueError gets caught in core.py and a new ValueError is raised
    with pytest.raises(ValueError) as excinfo:  # Expecting ValueError specifically now
        run_preprocessing(config=config)

    # Assert the new mock target was called
    try:
        assert mock_get_dims.call_count > 0, (
            "get_dimensions_from_metadata mock was not called!"
        )
    except AssertionError as e:
        pytest.fail(str(e))

    # --- MODIFICATION START ---
    # Assert the exception message *is* the one raised by core.py
    expected_message = "Failed during task prep/Z-crop/layout phase."
    assert expected_message in str(excinfo.value), (
        f"Expected message '{expected_message}' not found in exception: {str(excinfo.value)}"
    )
    # --- MODIFICATION END ---


@pytest.mark.mock
def test_pipeline_handles_output_mkdir_error(
    mocker, tmp_path, synthetic_tiff_factory, caplog
):
    """Test pipeline failure when creating output directory raises PermissionError."""
    runner = CliRunner()
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "protected_output"

    tiff_path = synthetic_tiff_factory(
        input_dir, BASE_TEST_FILENAME, shape=(10, 20, 20)
    )
    assert tiff_path.exists()

    # Mock mkdir where it's called in cli.py during setup
    mock_mkdir = mocker.patch("pixelpacker.cli.Path.mkdir", autospec=True)
    mock_mkdir.side_effect = PermissionError("Cannot create directory")

    with caplog.at_level(logging.ERROR):
        result = runner.invoke(
            app,
            [
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
                "--threads=1",
            ],
        )

    assert result.exit_code != 0, "CLI should fail (Exit Code)."
    assert any(
        "Failed to create or write to output directory" in record.message
        and record.levelname == "ERROR"
        and "pixelpacker.cli" in record.name
        for record in caplog.records
    ), "Expected error message about creating output directory not found in ERROR logs."
    # Assert mkdir was attempted
    mock_mkdir.assert_called()
