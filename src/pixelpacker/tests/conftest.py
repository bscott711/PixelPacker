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
logging.getLogger("matplotlib").setLevel(logging.INFO)


@pytest.fixture(scope="function")
def synthetic_tiff_factory():
    """
    Factory fixture to create synthetic TIFF files in a specified directory.
    Returns a function that takes the target directory as the first argument.
    """

    def _create_tiff(
        target_dir: Path,
        base_filename: str,
        shape: tuple,  # Original requested shape
        dtype: npt.DTypeLike = np.uint16,
        content_type: str = "constant",
        value: int = 100,
        channel: int = 0,
        timepoint: int = 0,
        prefix: str = "test",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Creates a synthetic TIFF file in target_dir."""
        if not base_filename.endswith((".tif", ".tiff")):
            base_filename += ".tif"

        structured_filename = (
            f"{prefix}_ch{channel}_stack{timepoint:04d}_{base_filename}"
        )
        filepath = target_dir / structured_filename
        target_dir.mkdir(parents=True, exist_ok=True)

        # --- Data Generation Logic ---

        # 1. Determine the effective 3D shape (ZYX) for generating content
        generation_shape = shape
        original_ndim = len(shape)  # Store original ndim
        if original_ndim == 5 and shape[0] == 1 and shape[2] == 1:
            generation_shape = shape[1:2] + shape[3:]  # (Z, Y, X)
        elif original_ndim == 4 and shape[0] == 1:
            generation_shape = shape[1:]  # (Z, Y, X)
        elif original_ndim == 2:
            generation_shape = (1,) + shape  # (1, Y, X)
        elif original_ndim == 3:
            pass  # generation_shape is already correct
        else:
            raise ValueError(
                f"Factory cannot generate data for unsupported input shape: {shape}"
            )

        if len(generation_shape) != 3:
            raise ValueError(
                f"Internal factory error: generation_shape is not 3D: {generation_shape}"
            )
        z, y, x = generation_shape

        # 2. Generate the 3D content data based on generation_shape
        # Initialize with a type annotation for clarity
        content_data_3d: np.ndarray
        if content_type == "constant":
            content_data_3d = np.full(generation_shape, value, dtype=dtype)
        elif content_type == "gradient_x":
            grad = np.linspace(0, 255, x, dtype=np.float32)
            content_data_3d = np.broadcast_to(grad, generation_shape).astype(dtype)
        elif content_type == "gradient_z":
            grad = np.linspace(0, 255, z, dtype=np.float32).reshape((z, 1, 1))
            content_data_3d = np.broadcast_to(grad, generation_shape).astype(dtype)
        elif content_type == "ramp":
            max_val = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1000
            data_float = np.linspace(0, min(max_val, 1000), z * y * x)
            content_data_3d = data_float.reshape(generation_shape).astype(dtype)
        elif content_type == "noise":
            max_val = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 255
            random_data = np.random.randint(0, max_val // 2, size=generation_shape)
            content_data_3d = random_data.astype(dtype)
        elif content_type == "zeros":
            content_data_3d = np.zeros(generation_shape, dtype=dtype)
        else:
            raise ValueError(f"Unknown content_type: {content_type}")

        # 3. Create the final data array with the original requested shape
        #    and place the generated 3D content appropriately.
        #    Initialize final_data outside the conditional block.
        final_data: np.ndarray
        if original_ndim <= 3:
            # If original request was 2D or 3D, save as 3D (1, Y, X) or (Z, Y, X)
            final_data = content_data_3d
        elif original_ndim == 4:
            # If original was 4D (1, Z, Y, X), create 4D and assign
            final_data = np.zeros(shape, dtype=dtype)
            final_data[0, :, :, :] = content_data_3d
        elif original_ndim == 5:
            # If original was 5D (1, Z, 1, Y, X), create 5D and assign
            final_data = np.zeros(shape, dtype=dtype)
            final_data[0, :, 0, :, :] = content_data_3d
        else:
            # This path should not be reached given the checks in step 1
            raise ValueError(
                f"Cannot handle final assignment for original shape {shape}"
            )

        # --- End Data Generation Logic ---

        log.debug(
            f"Creating synthetic TIFF at: {filepath} with final data shape {final_data.shape}"
        )
        tifffile.imwrite(filepath, final_data, metadata=metadata)
        return filepath

    return _create_tiff


# --- Other fixtures remain the same ---
@pytest.fixture
def mock_tifffile_imread(mocker):
    # Assuming TiffFile.asarray is the correct target for your code
    return mocker.patch("tifffile.TiffFile.asarray", autospec=True)


@pytest.fixture
def mock_path_mkdir(mocker):
    return mocker.patch("pathlib.Path.mkdir", autospec=True)
