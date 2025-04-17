# tiff_preprocessor/io_utils.py

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import matplotlib
import numpy as np
import tifffile
from PIL import Image

# Assuming stretch functions are in the same package
from .stretch import apply_autocontrast_8bit, compute_dynamic_cutoffs


# Ensure Matplotlib uses a non-interactive backend BEFORE importing pyplot
# Good practice for scripts/servers where GUIs aren't available/desired.
try:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None # type: ignore
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not found or backend 'Agg' failed. Debug histogram saving disabled.")



log = logging.getLogger(__name__)

# Constants for WebP saving
WEBP_QUALITY = 87
WEBP_METHOD = 6 # 0 (fastest) to 6 (slowest, best compression)
WEBP_LOSSLESS = False


def extract_volume(tif_path: Path) -> np.ndarray:
    """
    Extracts image data from a TIFF file and reshapes it into a 3D (Z, Y, X) volume.

    Attempts to handle common multi-dimensional TIFF formats by squeezing
    singleton dimensions and reshaping.

    Args:
        tif_path: Path object pointing to the input TIFF file.

    Returns:
        A 3D NumPy array representing the image volume (Z, Y, X).

    Raises:
        FileNotFoundError: If the tif_path does not exist.
        ValueError: If the TIFF shape after squeezing is not 2D, 3D, 4D (with singleton),
                    or 5D (with singleton T and C).
        Exception: Other potential errors during file reading or processing via tifffile.
    """
    log.debug(f"Extracting volume from: {tif_path}")
    if not tif_path.is_file():
        raise FileNotFoundError(f"TIFF file not found: {tif_path}")

    with tifffile.TiffFile(str(tif_path)) as tif:
        vol: np.ndarray = tif.asarray()

    log.debug(f"Original TIFF shape: {vol.shape}")
    # Remove singleton dimensions
    vol = np.squeeze(vol)
    log.debug(f"Squeezed TIFF shape: {vol.shape}")

    # Reshape based on remaining dimensions to get (Z, Y, X)
    if vol.ndim == 5:
        # Assuming shape might be (T, Z, C, Y, X) with T=1, C=1
        if vol.shape[0] == 1 and vol.shape[2] == 1:
            _, z, _, y, x = vol.shape
            vol = vol.reshape((z, y, x))
            log.debug(f"Reshaped 5D -> 3D: {vol.shape}")
        else:
             raise ValueError(f"Unsupported 5D TIFF shape after squeeze: {vol.shape}. Expected singleton T and C.")
    elif vol.ndim == 4:
        # Assuming one dimension is singleton (e.g., C or T)
        if 1 in vol.shape:
            vol = vol.reshape([s for s in vol.shape if s != 1])
            log.debug(f"Reshaped 4D -> 3D: {vol.shape}")
            if vol.ndim != 3: # Double check result is 3D
                 raise ValueError(f"Could not unambiguously reshape 4D TIFF to 3D: Original {tif.series[0].shape}, Squeezed {vol.shape}")
        else:
            raise ValueError(f"Unsupported 4D TIFF shape after squeeze: {vol.shape}. Expected one singleton dim.")
    elif vol.ndim == 3:
        # Already in (Z, Y, X) format, assuming Z is the first dim
        log.debug("Shape is 3D, no reshape needed.")
        pass
    elif vol.ndim == 2:
        # Assume (Y, X), add a singleton Z dimension
        log.debug("Shape is 2D, adding singleton Z dimension.")
        vol = vol[np.newaxis, :, :]
    else:
        raise ValueError(f"Unsupported TIFF shape after squeeze: {vol.shape} (ndim={vol.ndim})")

    if vol.ndim != 3:
         # This should ideally not be reached if logic above is correct
         raise ValueError(f"Final volume shape is not 3D after processing: {vol.shape}")

    log.debug(f"Final extracted volume shape: {vol.shape}")
    return vol


def save_preview_slice(vol: np.ndarray, path: Path, stretch_mode: str):
    """
    Saves a contrast-stretched middle Z-slice of the volume as a PNG image.

    Args:
        vol: The 3D (Z, Y, X) NumPy array volume.
        path: Path object for the output PNG file.
        stretch_mode: The contrast stretch mode to apply (passed to apply_autocontrast_8bit).
    """
    if vol.ndim != 3 or vol.shape[0] == 0:
        log.warning(f"Cannot save preview slice from volume with shape {vol.shape}")
        return

    mid_z = vol.shape[0] // 2
    slice_2d = vol[mid_z]

    try:
        # Apply contrast stretch to the 2D slice
        stretched_slice, _, _ = apply_autocontrast_8bit(slice_2d, stretch_mode)

        # Save using Pillow
        img_pil = Image.fromarray(stretched_slice)
        log.debug(f"Saving preview slice to: {path}")
        img_pil.save(str(path), format="PNG") # Specify format explicitly
    except Exception as e:
        log.error(f"Failed to save preview slice to {path}: {e}", exc_info=True)

def save_histogram_debug(img: np.ndarray, out_path: Path, stretch_mode: str):
    """
    Generates and saves a debug histogram plot of image intensities.

    Includes markers for various percentile and dynamic cutoff values.

    Args:
        img: The original (pre-stretch) NumPy array image data (can be 2D or 3D).
        out_path: Path object for the output PNG plot file.
        stretch_mode: String identifier for the stretch mode used (for title).
    """
    # Check if Matplotlib is usable early
    if not MATPLOTLIB_AVAILABLE or plt is None: # Added 'plt is None' check
        log.warning("Matplotlib unavailable, skipping histogram saving.")
        return

    # --- Fix 1: Assert plt is not None after the check ---
    assert plt is not None, "Matplotlib's pyplot (plt) should be loaded if MATPLOTLIB_AVAILABLE is True"

    # --- Fix 2: Initialize fig to None ---
    fig = None
    try:
        # Ensure img is numpy array
        img = np.asarray(img)
        # Consider only non-zero pixels for histogram stats
        nonzero_pixels = img[img > 0].flatten()

        if nonzero_pixels.size == 0:
            log.warning(f"Image for histogram {out_path.name} contains no non-zero pixels. Skipping plot.")
            return

        # --- Calculate various intensity markers ---
        p1 = float(np.percentile(nonzero_pixels, 1.0))
        p035_np, p9965_np = np.percentile(nonzero_pixels, (0.35, 99.65))
        p035 = float(p035_np)
        p9965 = float(p9965_np)
        smart_early, smart_late = compute_dynamic_cutoffs(nonzero_pixels)
        vmin = float(nonzero_pixels.min())
        vmax = float(nonzero_pixels.max())

        # --- Create Plot ---
        # This call is now safe due to the assert above
        fig, ax = plt.subplots(figsize=(12, 6)) # Line 170 in your error message

        ax.hist(nonzero_pixels, bins=256, color="gray", alpha=0.7, log=True, label="Log Histogram")

        plot_markers = {
            f"1% ({p1:.1f})": (p1, "blue"),
            f"ImageJ Min ({p035:.1f})": (p035, "orange"),
            f"Smart Early ({smart_early:.1f})": (smart_early, "purple"),
            f"Smart Late ({smart_late:.1f})": (smart_late, "green"),
            f"ImageJ Max ({p9965:.1f})": (p9965, "cyan"),
            f"Actual Max ({vmax:.1f})": (vmax, "red")
        }

        for label, (value, color) in plot_markers.items():
             if np.isfinite(value) and value >= vmin and value <= vmax :
                  ax.axvline(value, color=color, linestyle="--", label=label)

        ax.set_title(f"Intensity Histogram Debug – Stretch Mode: '{stretch_mode}'")
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Log Frequency (Count)")
        ax.legend(fontsize='small')
        ax.grid(True, axis='y', linestyle=':', alpha=0.5)
        fig.tight_layout()

        log.debug(f"Saving histogram debug plot to: {out_path}")
        fig.savefig(str(out_path), dpi=100)

    except Exception as e:
        log.error(f"Failed to generate or save histogram plot to {out_path}: {e}", exc_info=True)
    finally:
        # --- Fix 2: Check fig is not None before closing ---
        # This addresses the error on line 206
        if fig is not None:
            assert plt is not None # Safety assertion
            plt.close(fig) # Ensure figure is closed even on error

def process_channel(
    time_id: str,
    ch_id: int,
    tiff_path: str, # Keep as str if core.py passes str
    tile_w: int,
    tile_h: int,
    w: int,
    h: int,
    d: int,
    cols: int,
    rows: int,
    stretch_mode: str,
    dry_run: bool = False,
    debug: bool = False,
    output_folder: str = ".", # Keep as str if core.py passes str
    global_limits: Optional[Tuple[float, float]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Processes a single channel of a timepoint: extracts volume, stretches, tiles, and saves.

    Args:
        time_id: Identifier for the timepoint (e.g., 'stack0001').
        ch_id: Channel identifier (integer).
        tiff_path: Path to the input TIFF file for this channel/timepoint.
        tile_w: Total width of the final tiled image.
        tile_h: Total height of the final tiled image.
        w: Width of a single Z-slice.
        h: Height of a single Z-slice.
        d: Depth (number of Z-slices) of the volume.
        cols: Number of columns in the tile grid.
        rows: Number of rows in the tile grid.
        stretch_mode: Contrast stretch mode to apply.
        dry_run: If True, performs checks but avoids saving files.
        debug: If True, enables saving of debug histogram and preview slice.
        output_folder: Path to the folder where outputs will be saved.
        global_limits: Optional tuple (min, max) for global contrast stretching.

    Returns:
        A dictionary containing processing results ('channel', 'filename', 'intensity_range')
        if successful, otherwise None.
    """
    # Convert string paths to Path objects internally
    tiff_path_obj = Path(tiff_path)
    output_folder_obj = Path(output_folder)
    log.info(f"Processing T:{time_id} C:{ch_id} - File: {tiff_path_obj.name}")

    result_data: Optional[Dict[str, Any]] = None

    try:
        # --- Extract Volume ---
        vol = extract_volume(tiff_path_obj)

        # --- Validate Shape ---
        expected_shape = (d, h, w)
        if vol.shape != expected_shape:
            log.warning(
                f"Shape mismatch for T:{time_id} C:{ch_id}. "
                f"Expected {expected_shape}, got {vol.shape}. Skipping channel."
            )
            return None # Skip this channel

        # --- Apply Contrast Stretching ---
        vol_8bit, p_low, p_high = apply_autocontrast_8bit(vol, stretch_mode, global_limits)
        log.debug(f"T:{time_id} C:{ch_id} - Stretched Range: ({p_low:.2f}, {p_high:.2f})")


        # --- Save Debug Histogram (if enabled) ---
        if debug and not dry_run:
            hist_filename = f"debug_hist_T{time_id}_C{ch_id}.png"
            hist_path = output_folder_obj / hist_filename
            save_histogram_debug(vol, hist_path, stretch_mode) # Pass original volume

        # --- Create Tiled Image (if not dry run) ---
        # Define output filename regardless of dry run for metadata
        out_file = f"volume_{time_id}_c{ch_id}.webp"

        if not dry_run:
            tiled_img = Image.new("L", (tile_w, tile_h), color=0) # Initialize with black bg
            log.debug(f"T:{time_id} C:{ch_id} - Creating {tile_w}x{tile_h} tile image...")
            for i in range(d): # Iterate through depth (Z-slices)
                slice_img = Image.fromarray(vol_8bit[i])
                paste_x = (i % cols) * w
                paste_y = (i // cols) * h
                tiled_img.paste(slice_img, (paste_x, paste_y))

            # --- Save Tiled WebP Image ---
            out_path = output_folder_obj / out_file
            log.debug(f"Saving tiled WebP image to: {out_path}")
            try:
                tiled_img.save(
                    str(out_path),
                    format="WEBP",
                    quality=WEBP_QUALITY,
                    method=WEBP_METHOD,
                    lossless=WEBP_LOSSLESS
                )
            except Exception as e:
                 log.error(f"Failed to save WebP image {out_path}: {e}", exc_info=True)
                 # Decide if failure to save should prevent metadata return
                 # For now, we'll continue to return metadata but log error
                 # return None # Alternatively, uncomment to signify complete failure


        # --- Save Debug Preview Slice (if enabled, only for first channel) ---
        # Save preview based on original volume for better intensity comparison
        if debug and not dry_run and ch_id == 0:
            preview_filename = f"preview_T{time_id}_C{ch_id}.png" # Include C just in case
            preview_path = output_folder_obj / preview_filename
            save_preview_slice(vol, preview_path, stretch_mode)

        # --- Prepare Return Data ---
        result_data = {
            "time_id": time_id, # Include time_id for robustness
            "channel": ch_id,
            "filename": out_file,
            "intensity_range": {"p_low": p_low, "p_high": p_high}
        }
        log.info(f"Successfully processed T:{time_id} C:{ch_id}")

    except FileNotFoundError as e:
         log.error(f"❌ Error processing T:{time_id} C:{ch_id} - Input file not found: {e}")
         result_data = None # Ensure None is returned on file error
    except ValueError as e:
         log.error(f"❌ Error processing T:{time_id} C:{ch_id} - Data/Shape error: {e}")
         result_data = None
    except Exception as e:
        log.error(f"❌ Unexpected Error processing T:{time_id} C:{ch_id}: {e}", exc_info=True)
        result_data = None

    return result_data