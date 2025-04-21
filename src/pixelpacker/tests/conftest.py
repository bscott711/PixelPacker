# conftest.py
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import numpy.typing as npt
import pytest
import tifffile

log = logging.getLogger(__name__)
# Disable excessive logging from tifffile/PIL during tests
logging.getLogger("tifffile").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.WARNING)


@pytest.fixture(scope="function")
def synthetic_tiff_factory(): # No longer needs tmp_path directly injected
    """
    Factory fixture to create synthetic TIFF files in a specified directory.
    Returns a function that takes the target directory as the first argument.
    """

    # Corrected type hint for dtype
    def _create_tiff(
        target_dir: Path,               # <<< Added: Target directory parameter
        base_filename: str,             # <<< Renamed: Base name for clarity
        shape: tuple,
        dtype: npt.DTypeLike = np.uint16,
        content_type: str = "constant",
        value: int = 100,
        channel: int = 0,
        timepoint: int = 0,
        prefix: str = "test",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Creates a synthetic TIFF file in target_dir.

        Args:
            target_dir: The directory where the file will be created.
            base_filename: Base filename (channel/timepoint info added automatically).
            shape: The shape of the data array (Z, Y, X).
            dtype: Numpy data type (default: uint16).
            content_type: 'constant', 'gradient_x', 'gradient_z', 'ramp', 'noise'.
            value: The constant value for 'constant' type.
            channel: Channel number (for filename).
            timepoint: Timepoint number (for filename).
            prefix: Prefix for the filename.
            metadata: Optional metadata dictionary for tifffile.

        Returns:
            The Path object of the created TIFF file.
        """
        if not base_filename.endswith((".tif", ".tiff")):
            base_filename += ".tif"

        # Construct filename with channel/stack info
        structured_filename = (
            f"{prefix}_ch{channel}_stack{timepoint:04d}_{base_filename}"
        )
        # Use the provided target_dir
        filepath = target_dir / structured_filename # <<< Uses target_dir

        # Ensure target directory exists (good practice within factory)
        target_dir.mkdir(parents=True, exist_ok=True)

        # --- Data Generation Logic (Copied from your original code) ---
        if len(shape) == 2:
            shape = (1,) + shape
        elif len(shape) != 3:
            raise ValueError("Shape must be 3D (Z, Y, X) or 2D (Y, X)")
        z, y, x = shape
        data = np.zeros(shape, dtype=dtype) # Initialize data array
        if content_type == "constant":
            data[:] = value
        elif content_type == "gradient_x":
            grad = np.linspace(0, 255, x, dtype=np.float32)
            data = np.broadcast_to(grad, (z, y, x)).astype(dtype)
        elif content_type == "gradient_z":
            grad = np.linspace(0, 255, z, dtype=np.float32).reshape((z, 1, 1))
            data = np.broadcast_to(grad, (z, y, x)).astype(dtype)
        elif content_type == "ramp":
            max_val = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1000
            data_float = np.linspace(0, min(max_val, 1000), z * y * x)
            data = data_float.reshape(shape).astype(dtype)
        elif content_type == "noise":
            max_val = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 255
            random_data = np.random.randint(0, max_val // 2, size=shape)
            data = random_data.astype(dtype)
        elif content_type == "zeros":
            pass  # Already zeros
        else:
            raise ValueError(f"Unknown content_type: {content_type}")
        # --- End Data Generation Logic ---

        log.debug(f"Creating synthetic TIFF at: {filepath}") # Use logger
        tifffile.imwrite(filepath, data, metadata=metadata)
        return filepath

    return _create_tiff # Return the inner function


# --- Other fixtures remain the same ---
@pytest.fixture
def mock_tifffile_imread(mocker):
    # Assuming TiffFile.asarray is the correct target for your code
    return mocker.patch("tifffile.TiffFile.asarray", autospec=True)

@pytest.fixture
def mock_path_mkdir(mocker):
    return mocker.patch("pathlib.Path.mkdir", autospec=True)