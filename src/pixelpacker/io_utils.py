# src/pixelpacker/io_utils.py

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib
import numpy as np
import tifffile
from PIL import Image

from .stretch import apply_autocontrast_8bit, ContrastLimits
from .data_models import VolumeLayout

log = logging.getLogger(__name__)
try:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not found or backend 'Agg' failed. Debug histogram saving disabled.")

# Constants
WEBP_QUALITY = 87
WEBP_METHOD = 6
WEBP_LOSSLESS = False

# --- Z-Cropping Function ---
def find_z_crop_range(volume: np.ndarray, threshold: float) -> Tuple[int, int]:
    """
    Finds the start and end Z-slice indices to keep based on max intensity.

    Args:
        volume: The 3D numpy array (Z, Y, X).
        threshold: The maximum intensity value below which a slice is considered empty.

    Returns:
        A tuple (z_start, z_end) representing the inclusive slice range to keep.
        Returns (0, depth-1) if no cropping is needed or possible.
    """
    if volume.ndim != 3 or volume.shape[0] == 0:
        return (0, 0) # Cannot crop non-3D or empty volume

    depth = volume.shape[0]
    if depth <= 1:
        return (0, depth - 1) # Cannot crop single slice

    # Calculate max intensity per Z-slice
    # Use try-except for potential memory errors on huge slices? For now, assume it fits.
    try:
        max_per_slice = np.max(volume, axis=(1, 2))
    except MemoryError:
         log.error("MemoryError calculating max per slice for Z-cropping. Skipping crop.")
         return (0, depth - 1)

    # Find indices where max intensity is > threshold
    valid_indices = np.where(max_per_slice > threshold)[0]

    if valid_indices.size == 0:
        # No slices are above the threshold - volume is effectively empty or below threshold
        log.warning(f"No Z-slices found with max intensity > {threshold}. Cannot crop.")
        # Return a single slice range (e.g., the middle one) or the original?
        # Returning original might be safer, but could lead to large empty outputs.
        # Let's return the original range for now.
        return (0, depth - 1)

    z_start = int(valid_indices.min())
    z_end = int(valid_indices.max())

    # Ensure start <= end (should always be true if valid_indices is not empty)
    if z_start > z_end:
         log.error(f"Z-crop calculation resulted in start > end ({z_start} > {z_end}). This should not happen.")
         return (0, depth - 1) # Fallback to original range

    log.debug(f"Z-crop range determined: Keep slices {z_start} to {z_end} (inclusive).")
    return z_start, z_end


# --- Modified extract_volume ---
def extract_volume(tif_path: Path, z_crop_threshold: int) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
    """
    Extracts, reshapes, and optionally crops the volume from a TIFF file.

    Args:
        tif_path: Path to the input TIFF file.
        z_crop_threshold: Max intensity threshold for Z-cropping.

    Returns:
        A tuple containing:
          - The processed (and potentially cropped) 3D numpy array, or None if extraction fails.
          - A tuple (z_start, z_end) of the kept slice range, or None if extraction fails.
    """
    log.debug(f"Extracting volume from: {tif_path}")
    if not tif_path.is_file():
        log.error(f"TIFF file not found: {tif_path}")
        return None, None # Return None if file not found

    try:
        with tifffile.TiffFile(str(tif_path)) as tif:
            vol: np.ndarray = tif.asarray()

        log.debug(f"Original TIFF shape: {vol.shape}")

        # --- Squeeze/Reshape Logic (same as before) ---
        vol = np.squeeze(vol)
        log.debug(f"Squeezed TIFF shape: {vol.shape}")
        if vol.ndim == 5:
            if vol.shape[0] == 1 and vol.shape[2] == 1:
                 _, z, _, y, x = vol.shape
                 vol = vol.reshape((z, y, x))
                 log.debug(f"Reshaped 5D (assumed T=1, C=1) -> 3D: {vol.shape}")
            else:
                raise ValueError(f"Unsupported 5D TIFF shape after squeeze: {vol.shape}.")
        elif vol.ndim == 4:
            if 1 in vol.shape:
                original_4d_shape = vol.shape
                vol = vol.reshape([s for s in vol.shape if s != 1])
                log.debug(f"Reshaped 4D {original_4d_shape} -> {vol.ndim}D: {vol.shape}")
                if vol.ndim != 3:
                     raise ValueError("Could not unambiguously reshape 4D TIFF to 3D.")
            else:
                raise ValueError(f"Unsupported 4D TIFF shape after squeeze: {vol.shape}.")
        elif vol.ndim == 3:
             log.debug("Shape is 3D, no reshape needed.")
        elif vol.ndim == 2:
             log.debug("Shape is 2D, adding singleton Z dimension.")
             vol = vol[np.newaxis, :, :]
        else:
             raise ValueError(f"Unsupported TIFF shape after squeeze: {vol.shape} (ndim={vol.ndim})")

        if vol.ndim != 3:
             raise ValueError(f"Volume shape is not 3D after reshaping: {vol.shape}")

        original_depth = vol.shape[0]
        log.debug(f"Volume shape before Z-crop: {vol.shape}")

        # --- Apply Z-Cropping ---
        z_start, z_end = find_z_crop_range(vol, z_crop_threshold)
        cropped_vol = vol[z_start : z_end + 1, :, :]
        cropped_depth = cropped_vol.shape[0]
        z_range = (z_start, z_end)

        if original_depth != cropped_depth:
            log.info(f"Applied Z-crop (threshold={z_crop_threshold}): Original depth {original_depth}, Cropped depth {cropped_depth} (slices {z_start}-{z_end})")
        else:
            log.debug("No Z-cropping applied (or threshold too high).")

        log.debug(f"Final extracted volume shape: {cropped_vol.shape}")
        return cropped_vol, z_range

    except FileNotFoundError: # Already handled above, but keep for safety
        log.error(f"File not found during extraction: {tif_path}")
        return None, None
    except ValueError as e:
        log.error(f"Shape/Value error extracting {tif_path.name}: {e}")
        return None, None
    except Exception as e:
        log.error(f"Unexpected error extracting volume from {tif_path.name}: {e}", exc_info=True)
        return None, None


# --- save_preview_slice, save_histogram_debug (remain unchanged) ---
# ... (code for these functions) ...
def save_preview_slice(vol_8bit: np.ndarray, path: Path):
    # (Implementation unchanged)
    if vol_8bit.ndim != 3 or vol_8bit.shape[0] == 0:
        log.warning(f"Cannot save preview slice from volume with shape {vol_8bit.shape}")
        return
    if vol_8bit.dtype != np.uint8:
        log.warning(f"Preview slice input is not uint8 (dtype: {vol_8bit.dtype}). Result may be unexpected.")
    mid_z = vol_8bit.shape[0] // 2
    slice_2d = vol_8bit[mid_z]
    try:
        img_pil = Image.fromarray(slice_2d)
        log.debug(f"Saving preview slice to: {path}")
        img_pil.save(str(path), format="PNG")
    except Exception as e:
        log.error(f"Failed to save preview slice to {path}: {e}", exc_info=True)

def save_histogram_debug(img: np.ndarray, limits: ContrastLimits, out_path: Path, stretch_mode: str):
    # (Implementation unchanged)
    if not MATPLOTLIB_AVAILABLE or plt is None:
        log.warning("Matplotlib unavailable, skipping histogram saving.")
        return
    assert plt is not None, "plt should be loaded if MATPLOTLIB_AVAILABLE is True"
    fig = None
    try:
        img = np.asarray(img)
        nonzero_pixels = img[img > 0].flatten()
        if nonzero_pixels.size == 0:
            log.warning(f"Image for histogram {out_path.name} contains no non-zero pixels. Skipping plot.")
            return
        vmin = limits.actual_min if limits.actual_min is not None else 0.0
        vmax = limits.actual_max if limits.actual_max is not None else 0.0
        # Avoid plotting if range is invalid or zero
        if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmax <= vmin:
             log.warning(f"Invalid or zero range [{vmin}, {vmax}] for histogram {out_path.name}. Skipping plot.")
             return

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(nonzero_pixels, bins=256, range=(vmin, vmax), color="gray", alpha=0.7, log=True, label="Log Histogram (Non-zero Pixels)")
        plot_markers = {
            f"Stretch Low ({limits.p_low:.1f})": (limits.p_low, "red", "-"),
            f"Stretch High ({limits.p_high:.1f})": (limits.p_high, "red", "-"),
            f"1% ({limits.p1:.1f})" if limits.p1 is not None else None: (limits.p1, "blue", "--"),
            f"ImageJ Min ({limits.p035:.1f})" if limits.p035 is not None else None: (limits.p035, "orange", "--"),
            f"Smart Early ({limits.smart_early:.1f})" if limits.smart_early is not None else None: (limits.smart_early, "purple", "--"),
            f"Smart Late ({limits.smart_late:.1f})" if limits.smart_late is not None else None: (limits.smart_late, "green", "--"),
            f"ImageJ Max ({limits.p9965:.1f})" if limits.p9965 is not None else None: (limits.p9965, "cyan", "--"),
            f"Actual Max ({limits.actual_max:.1f})" if limits.actual_max is not None else None: (limits.actual_max, "magenta", ":"),
            f"Actual Min ({limits.actual_min:.1f})" if limits.actual_min is not None else None: (limits.actual_min, "yellow", ":"),
        }
        added_labels = set()
        # Get current Y-axis limits to position lines correctly
        y_min, y_max = ax.get_ylim()

        for label, (value, color, style) in plot_markers.items():
            # Ensure value is valid and within plot range before drawing line
            if label is not None and value is not None and np.isfinite(value) and value >= vmin and value <= vmax:
                if label not in added_labels:
                     # Use ax.vlines for better control than axvline with log scale
                     ax.vlines(value, y_min, y_max, color=color, linestyle=style, label=label)
                     added_labels.add(label)

        ax.set_title(f"Intensity Histogram Debug – Stretch Mode: '{stretch_mode}'\n(Bounds Used: [{limits.p_low:.1f} - {limits.p_high:.1f}])")
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Log Frequency (Count)")
        # Reset Y limits after adding vlines if necessary, or adjust vlines ymax
        ax.set_ylim(y_min, y_max)
        ax.legend(fontsize="small")
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        ax.set_xlim(vmin, vmax)
        log.debug(f"Saving histogram debug plot to: {out_path}")
        fig.savefig(str(out_path), dpi=100)
    except Exception as e:
        log.error(f"Failed to generate or save histogram plot to {out_path}: {e}", exc_info=True)
    finally:
        if fig is not None:
            plt.close(fig) # Ensure figure is closed

# --- Modified process_channel ---
def process_channel(
    time_id: str,
    ch_id: int,
    # --- Pass cropped volume and layout based on cropped depth ---
    cropped_vol: np.ndarray,
    layout: VolumeLayout, # Layout should now be based on cropped_vol.shape
    # --- Pass limits and original z_range for metadata ---
    limits: ContrastLimits,
    z_range: Tuple[int, int],
    # --- Other args remain ---
    stretch_mode: str, # Keep for logging/debug purposes maybe?
    dry_run: bool = False,
    debug: bool = False,
    output_folder: str = ".",
) -> Optional[Dict[str, Any]]:
    """Processes a single channel: tiles stretched volume and saves."""
    output_folder_obj = Path(output_folder)
    log.info(f"Processing T:{time_id} C:{ch_id} - Cropped Shape: {cropped_vol.shape}")
    result_data: Optional[Dict[str, Any]] = None

    try:
        # --- Volume is already extracted, cropped, and stretched ---
        # --- Contrast limits are already calculated ---
        vol_8bit = apply_autocontrast_8bit(cropped_vol, stretch_mode="max", global_limits_tuple=(limits.p_low, limits.p_high))[0]
        # We re-apply contrast here using the pre-calculated limits.
        # Using "max" mode with global_limits ensures it just applies the given range.

        log.debug(f"T:{time_id} C:{ch_id} - Applied Stretched Range: ({limits.p_low:.2f}, {limits.p_high:.2f})")

        if debug and not dry_run:
            # Save histogram based on the original limits calculation pass
            hist_filename = f"debug_hist_T{time_id}_C{ch_id}.png"
            hist_path = output_folder_obj / hist_filename
            # Need original volume data for histogram? Or cropped is fine?
            # Let's use cropped volume for histogram consistency with processing.
            save_histogram_debug(cropped_vol, limits, hist_path, stretch_mode) # Pass original mode for title

        out_file = f"volume_{time_id}_c{ch_id}.webp"
        if not dry_run:
            # --- Tiling logic uses layout based on cropped depth ---
            tiled_img = Image.new("L", (layout.tile_width, layout.tile_height), color=0)
            log.debug(f"T:{time_id} C:{ch_id} - Creating {layout.tile_width}x{layout.tile_height} tile image...")
            # Iterate using the depth from the layout (which is based on cropped_vol)
            for i in range(layout.depth):
                slice_img = Image.fromarray(vol_8bit[i])
                paste_x = (i % layout.cols) * layout.width
                paste_y = (i // layout.cols) * layout.height
                tiled_img.paste(slice_img, (paste_x, paste_y))

            out_path = output_folder_obj / out_file
            log.debug(f"Saving tiled WebP image to: {out_path}")
            try:
                tiled_img.save(
                    str(out_path), format="WEBP", quality=WEBP_QUALITY,
                    method=WEBP_METHOD, lossless=WEBP_LOSSLESS,
                )
            except Exception as e:
                log.error(f"Failed to save WebP image {out_path}: {e}", exc_info=True)
                # Continue to return metadata even if save fails? Or return None? Let's return None.
                return None

        if debug and not dry_run and ch_id == 0:
            # Save preview slice from the processed 8-bit volume
            preview_filename = f"preview_T{time_id}_C{ch_id}.png"
            preview_path = output_folder_obj / preview_filename
            save_preview_slice(vol_8bit, preview_path)

        # --- Metadata includes the applied contrast limits and z_range ---
        result_data = {
            "time_id": time_id,
            "channel": ch_id,
            "filename": out_file,
            "intensity_range": {"p_low": limits.p_low, "p_high": limits.p_high},
            "z_crop_range": list(z_range), # Store as list [start, end]
        }
        log.info(f"Successfully processed T:{time_id} C:{ch_id}")

    except Exception as e:
        log.error(f"❌ Unexpected Error processing T:{time_id} C:{ch_id}: {e}", exc_info=True)
        result_data = None # Ensure failure returns None

    return result_data

