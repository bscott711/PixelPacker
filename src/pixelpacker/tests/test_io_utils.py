# src/pixelpacker/tests/test_io_utils.py
import numpy as np
import pytest
from PIL import Image

from pixelpacker.io_utils import (
    extract_original_volume,
    find_z_crop_range,
    save_preview_slice,
    _save_debug_array_as_image,
)
from pixelpacker.stretch import ContrastLimits


# --- Fixtures for volumes ---
@pytest.fixture
def volume_all_zeros() -> np.ndarray:
    """Volume containing only zeros."""
    return np.zeros((20, 50, 50), dtype=np.uint16)


@pytest.fixture
def volume_constant_non_zero() -> np.ndarray:
    """Volume with a constant non-zero value."""
    return np.full((20, 50, 50), 100, dtype=np.uint16)


@pytest.fixture
def volume_with_empty_ends() -> np.ndarray:
    """Volume with data only in the middle Z slices."""
    vol = np.zeros((30, 50, 50), dtype=np.uint16)
    vol[10:20, :, :] = np.random.randint(50, 200, size=(10, 50, 50), dtype=np.uint16)
    return vol


@pytest.fixture
def volume_gradient_z() -> np.ndarray:
    """Volume with a gradient along the Z axis."""
    z, y, x = 30, 50, 50
    grad = np.linspace(0, 255, z, dtype=np.float32).reshape((z, 1, 1))
    vol = np.broadcast_to(grad, (z, y, x)).astype(np.uint16)
    return vol


# --- Tests for extract_original_volume ---


@pytest.mark.unit
def test_extract_volume_basic_3d(tmp_path, synthetic_tiff_factory):
    """Test extracting a standard 3D TIFF."""
    shape = (15, 30, 40)
    # Add tmp_path as the first argument
    filepath = synthetic_tiff_factory(tmp_path, "basic_3d.tif", shape=shape)  # MODIFIED
    vol = extract_original_volume(filepath)
    assert vol is not None
    assert vol.shape == shape
    assert vol.dtype == np.uint16


@pytest.mark.unit
def test_extract_volume_squeezes_dims(tmp_path, synthetic_tiff_factory):
    """Test that singleton dimensions are squeezed correctly."""
    # Simulate a 5D TIFF (T=1, C=1)
    shape_5d = (1, 15, 1, 30, 40)
    # Add tmp_path as the first argument
    filepath = synthetic_tiff_factory(
        tmp_path, "squeezable_5d.tif", shape=shape_5d
    )  # MODIFIED
    vol = extract_original_volume(filepath)
    assert vol is not None
    assert vol.shape == (15, 30, 40)

    # Simulate a 4D TIFF (C=1)
    shape_4d = (1, 15, 30, 40)
    # Add tmp_path as the first argument
    filepath = synthetic_tiff_factory(
        tmp_path, "squeezable_4d.tif", shape=shape_4d
    )  # MODIFIED
    vol = extract_original_volume(filepath)
    assert vol is not None
    assert vol.shape == (15, 30, 40)

    # Simulate a 2D TIFF
    shape_2d = (30, 40)
    # Add tmp_path as the first argument
    filepath = synthetic_tiff_factory(
        tmp_path, "basic_2d.tif", shape=shape_2d
    )  # MODIFIED
    vol = extract_original_volume(filepath)
    assert vol is not None
    assert vol.shape == (1, 30, 40)


@pytest.mark.unit
def test_extract_volume_file_not_found(tmp_path):
    """Test behavior when the TIFF file does not exist."""
    filepath = tmp_path / "non_existent.tif"
    vol = extract_original_volume(filepath)
    assert vol is None


@pytest.mark.unit
def test_extract_volume_corrupt_file(tmp_path, mocker):  # Use mocker directly if needed
    """Test behavior when tifffile fails to read the file."""
    filepath = tmp_path / "corrupt.tif"
    filepath.touch()  # Create empty file

    # Option 1: Mock TiffFile.__init__ or TiffFile.__enter__ if needed,
    # but the function already handles the exception from tifffile.
    # Option 2: Accept that asarray isn't called and just check the result.

    # Mocking tifffile.TiffFile itself to raise error on init/open
    mock_tiff_file = mocker.patch("pixelpacker.io_utils.tifffile.TiffFile")
    # Simulate error during context management or initial read attempt
    mock_tiff_file.side_effect = ValueError("Cannot read TIFF (mocked)")

    vol = extract_original_volume(filepath)
    assert vol is None
    # Remove the assert_called_once for asarray, as it's not reached.
    # mock_tifffile_imread.assert_called_once() # REMOVED
    # Instead, assert that our TiffFile mock was attempted
    mock_tiff_file.assert_called_once_with(str(filepath))


# --- Tests for Z-Cropping ---


@pytest.mark.parametrize("method", ["slope", "threshold"])
@pytest.mark.unit
def test_zcrop_empty_volume(method):
    """Test Z-cropping on an empty or zero-depth volume."""
    empty_vol = np.zeros((0, 10, 10), dtype=np.uint16)
    start, end = find_z_crop_range(empty_vol, method=method, threshold=10)
    assert start == 0
    assert end == 0


@pytest.mark.parametrize("method", ["slope", "threshold"])
@pytest.mark.unit
def test_zcrop_all_zeros(volume_all_zeros, method):
    """Test Z-cropping on a volume of all zeros."""
    depth = volume_all_zeros.shape[0]
    start, end = find_z_crop_range(volume_all_zeros, method=method, threshold=10)
    # Slope method might have issues, threshold should return full range
    if method == "threshold":
        assert start == 0
        assert end == depth - 1
    else:  # Slope
        # Slope of zero profile is zero, might default to full range
        assert start == 0
        assert end == depth - 1


@pytest.mark.unit
def test_zcrop_threshold_method(volume_with_empty_ends):
    """Test the threshold Z-cropping method specifically."""
    depth = volume_with_empty_ends.shape[0]
    # Threshold below the actual data
    start, end = find_z_crop_range(
        volume_with_empty_ends, method="threshold", threshold=10
    )
    assert start == 10
    assert end == 19

    # Threshold above the actual data
    start, end = find_z_crop_range(
        volume_with_empty_ends, method="threshold", threshold=300
    )
    assert start == 0  # Should default to full range if nothing found
    assert end == depth - 1


@pytest.mark.unit
def test_zcrop_slope_method(volume_with_empty_ends, tmp_path):
    """Test the slope-based Z-cropping method."""
    # Should identify the sharp increase/decrease around the data block
    # Exact values depend on smoothing and thresholds, aim for approximate range
    start, end = find_z_crop_range(
        volume_with_empty_ends,
        method="slope",
        threshold=0,  # Threshold not used by slope
        debug=True,  # Test debug plot generation
        output_folder=tmp_path,
        filename_prefix="zcrop_slope_test",
    )
    assert 5 <= start <= 11  # Allow some buffer around the actual start (10)
    assert 18 <= end <= 24  # Allow some buffer around the actual end (19)

    # Check that debug files were created
    assert (tmp_path / "zcrop_slope_test_debug_mip_yz.png").is_file()
    assert (tmp_path / "zcrop_slope_test_debug_mip_xz.png").is_file()
    assert (tmp_path / "zcrop_slope_test_debug_z_profile_slope.png").is_file()


@pytest.mark.unit
def test_zcrop_slope_method_gradient(volume_gradient_z):
    """Test slope method on a smooth gradient (should keep most slices)."""
    depth = volume_gradient_z.shape[0]
    start, end = find_z_crop_range(volume_gradient_z, method="slope", threshold=0)
    # Slope should be relatively constant, might trim very first/last slices
    assert start <= 5
    assert end >= depth - 6


@pytest.mark.unit
def test_zcrop_unknown_method(volume_with_empty_ends):
    """Test fallback to slope method if unknown method is provided."""
    start, end = find_z_crop_range(
        volume_with_empty_ends, method="unknown", threshold=0
    )
    # Should behave like the slope test
    assert 5 <= start <= 11
    assert 18 <= end <= 24


# --- Tests for Saving Utilities ---


@pytest.mark.unit
def test_save_preview_slice(tmp_path):
    """Test saving a preview PNG slice."""
    vol_8bit = np.random.randint(0, 256, size=(10, 50, 60), dtype=np.uint8)
    preview_path = tmp_path / "preview.png"
    save_preview_slice(vol_8bit, preview_path)
    assert preview_path.is_file()
    try:
        img = Image.open(preview_path)
        assert img.size == (60, 50)  # W, H
        assert img.mode == "L"  # Grayscale
    except Exception as e:
        pytest.fail(f"Failed to open saved preview image: {e}")


@pytest.mark.unit
def test_save_preview_slice_invalid_input(tmp_path):
    """Test save_preview_slice handles invalid input shapes."""
    vol_2d = np.random.randint(0, 256, size=(50, 60), dtype=np.uint8)
    vol_empty = np.zeros((0, 50, 60), dtype=np.uint8)
    preview_path = tmp_path / "preview_invalid.png"

    save_preview_slice(vol_2d, preview_path)
    assert not preview_path.exists()  # Should log warning and not save

    save_preview_slice(vol_empty, preview_path)
    assert not preview_path.exists()  # Should log warning and not save


@pytest.mark.unit
def test_save_debug_array_as_image(tmp_path):
    """Test saving a 2D debug array as PNG."""
    arr = np.linspace(0, 500, 100 * 80).reshape((100, 80)).astype(np.uint16)
    debug_path = tmp_path / "debug_arr.png"

    # Test basic min-max scaling
    _save_debug_array_as_image(arr, debug_path, limits=None)
    assert debug_path.is_file()
    img = Image.open(debug_path)
    assert img.size == (80, 100)
    assert img.mode == "L"

    # Test scaling with ContrastLimits
    limits = ContrastLimits(p_low=50.0, p_high=450.0)
    debug_path_scaled = tmp_path / "debug_arr_scaled.png"
    _save_debug_array_as_image(arr, debug_path_scaled, limits=limits)
    assert debug_path_scaled.is_file()
    # Could potentially check pixel values if needed, but basic save is main goal


@pytest.mark.unit
def test_save_debug_array_invalid_input(tmp_path):
    """Test _save_debug_array_as_image handles invalid inputs."""
    arr_3d = np.zeros((2, 10, 10), dtype=np.uint16)
    arr_empty = np.zeros((0, 10), dtype=np.uint16)
    debug_path = tmp_path / "debug_invalid.png"

    _save_debug_array_as_image(arr_3d, debug_path)
    assert not debug_path.exists()

    _save_debug_array_as_image(arr_empty, debug_path)
    assert not debug_path.exists()
