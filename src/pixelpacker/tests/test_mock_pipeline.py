# src/pixelpacker/tests/test_mock_pipeline.py
import pytest
from typer.testing import CliRunner  # Keep for mkdir test
import logging

# Import the core function and config dataclass directly
from pixelpacker.cli import app  # Keep for mkdir test
from pixelpacker.core import run_preprocessing
from pixelpacker.data_models import PreprocessingConfig

BASE_TEST_FILENAME = "image.tif"
EXPECTED_TEST_FILENAME = f"test_ch0_stack0000_{BASE_TEST_FILENAME}"


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
        stretch_mode="smart",  # Use relevant defaults or test values
        z_crop_method="slope",
        z_crop_threshold=0,
        dry_run=False,
        debug=False,  # Set True if needed for log inspection
        max_threads=1,
        use_global_contrast=True,
        executor_type="thread",  # Force thread executor
        input_pattern="*_ch*_stack*.tif*",  # Ensure pattern matches
    )

    # Expect run_preprocessing to raise RuntimeError due to error propagation
    with pytest.raises(RuntimeError) as excinfo:
        run_preprocessing(config=config)

    # Assert mock was called (hopefully before exception fully unwound)
    try:
        mock_save.assert_called()
    except AssertionError as e:
        pytest.fail(f"PIL.Image.Image.save mock was not called! {e}")

    # Assert the exception message contains info about the processing error
    assert "processing error" in str(excinfo.value)


@pytest.mark.mock
def test_pipeline_handles_input_read_error(mocker, tmp_path, synthetic_tiff_factory):
    """Test pipeline failure when reading TIFF raises an error."""
    # Apply mock BEFORE runner initialization
    # Change the patch target to where it's imported/used in the first pass
    mock_extract = mocker.patch(
        "pixelpacker.crop.extract_original_volume"
    )  # <<< CHANGED TARGET
    mock_extract.side_effect = ValueError("Invalid TIFF data")

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    tiff_path = synthetic_tiff_factory(
        input_dir, BASE_TEST_FILENAME, shape=(10, 20, 20)
    )
    assert tiff_path.exists()

    # Create config manually
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
        executor_type="thread",  # Force thread executor
        input_pattern="*_ch*_stack*.tif*",
    )

    # Expect run_preprocessing to raise an error
    with pytest.raises((RuntimeError, ValueError)) as excinfo:
        run_preprocessing(config=config)

    # Assert mock was called (it should be hit during Pass 0 now)
    try:
        assert mock_extract.call_count > 0, (
            "extract_original_volume mock was not called!"
        )
    except AssertionError as e:
        pytest.fail(str(e))

    # Assert the exception message contains info about the error
    assert (
        "Invalid TIFF data" in str(excinfo.value)
        or "processing error" in str(excinfo.value)
        or "Failed during task prep" in str(excinfo.value)
        or "No contrast limits" in str(excinfo.value)
    )


# Keep the passing test using CliRunner
@pytest.mark.mock
def test_pipeline_handles_output_mkdir_error(
    mocker, tmp_path, synthetic_tiff_factory, caplog
):
    """Test pipeline failure when creating output directory raises PermissionError."""
    runner = CliRunner()  # Keep runner for this test
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "protected_output"

    tiff_path = synthetic_tiff_factory(
        input_dir, BASE_TEST_FILENAME, shape=(10, 20, 20)
    )
    assert tiff_path.exists()

    mock_mkdir = mocker.patch("pathlib.Path.mkdir", autospec=True)
    mock_mkdir.side_effect = PermissionError("Cannot create directory")

    with caplog.at_level(logging.ERROR):
        result = runner.invoke(
            app,  # Still invoke CLI here as error is in CLI setup
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
    mock_mkdir.assert_called()
