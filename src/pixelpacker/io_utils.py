# src/pixelpacker/io_utils.py

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib
import numpy as np
import tifffile
from PIL import Image
from scipy.ndimage import gaussian_filter1d

# --- Added scikit-image import ---
try:
    from skimage.util import montage as skimage_montage

    SKIMAGE_AVAILABLE = True
except ImportError:
    skimage_montage = None
    SKIMAGE_AVAILABLE = False
    logging.warning("scikit-image not found. Tiling will use NumPy loop fallback.")
# --- End Added import ---


# Import ContrastLimits for scaling debug MIPs
from .data_models import VolumeLayout
from .stretch import ContrastLimits, apply_autocontrast_8bit, calculate_limits_only

log = logging.getLogger(__name__)
try:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False
    logging.warning(
        "Matplotlib not found or backend 'Agg' failed."
        " Debug histogram/plot saving disabled."
    )

# Constants
WEBP_QUALITY = 87
WEBP_METHOD = 6
WEBP_LOSSLESS = False
SMOOTHING_SIGMA_Z = 2.0
SLOPE_WINDOW_Z = 2
SLOPE_THRESH_Z_POS = 20
MIN_SEARCH_OFFSET_Z = 3


# --- Helper to save debug NumPy arrays as images ---
# _save_debug_array_as_image remains the same
def _save_debug_array_as_image(
    arr: np.ndarray, filename: Path, limits: Optional[ContrastLimits] = None
):
    """Scales a 2D numpy array and saves as PNG."""
    if arr.ndim != 2:
        log.warning("Cannot save debug array %s: Not 2D.", filename.name)
        return
    if arr.size == 0:
        log.warning("Cannot save empty debug array: %s", filename.name)
        return
    try:
        arr_8bit: np.ndarray
        if limits:
            log.debug(
                "Scaling debug image %s using limits: [%.1f-%.1f]",
                filename.name,
                limits.p_low,
                limits.p_high,
            )
            p_low = limits.p_low
            p_high = limits.p_high
            if p_high <= p_low:
                arr_8bit = np.where(arr <= p_low, 0, 255).astype(np.uint8)
            else:
                arr_float = arr.astype(np.float32)
                denominator = p_high - p_low
                # Check denominator again just in case
                if denominator == 0:
                    denominator = 1e-6
                scaled_float = (arr_float - p_low) / denominator
                scaled_float = np.clip(scaled_float, 0.0, 1.0)
                arr_8bit = (scaled_float * 255.0).astype(np.uint8)
        else:
            log.debug("Scaling debug image %s using min-max.", filename.name)
            arr_min = np.min(arr)
            arr_max = np.max(arr)
            if arr_max > arr_min:
                arr_norm = (arr.astype(np.float32) - arr_min) / (arr_max - arr_min)
                arr_8bit = (np.clip(arr_norm, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                # Flat image
                arr_8bit = np.zeros_like(arr, dtype=np.uint8)

        img_pil = Image.fromarray(arr_8bit)
        log.debug("Saving debug image: %s", filename)
        img_pil.save(str(filename), format="PNG")
    except Exception as e:
        log.error("Failed to save debug array image %s: %s", filename, e, exc_info=True)


# --- Z-Cropping Method 1: Simple Threshold on Max Profile ---
# _find_z_crop_range_threshold remains the same
def _find_z_crop_range_threshold(volume: np.ndarray, threshold: int) -> Tuple[int, int]:
    """
    Finds Z-crop range based on thresholding the max intensity profile
    derived from YZ and XZ MIPs. (Original projection method).
    """
    if volume.ndim != 3 or volume.shape[0] == 0:
        return (0, 0)
    depth = volume.shape[0]
    if depth <= 1:
        return (0, depth - 1 if depth > 0 else 0)

    try:
        log.debug("Calculating MIPs for threshold-based Z-cropping...")
        mip_yz = np.max(volume, axis=2)
        mip_xz = np.max(volume, axis=1)
        max_per_z_yz = np.max(mip_yz, axis=1)
        max_per_z_xz = np.max(mip_xz, axis=1)
        max_z_profile = np.maximum(max_per_z_yz, max_per_z_xz)
        log.debug(
            "Max Z profile range: [%.1f, %.1f]",
            np.min(max_z_profile),
            np.max(max_z_profile),
        )

        valid_indices = np.where(max_z_profile > threshold)[0]

        if valid_indices.size == 0:
            log.warning(
                "Threshold method: No Z-slices found with max projection intensity > %d. Cannot crop.",
                threshold,
            )
            return (0, depth - 1)

        z_start = int(valid_indices.min())
        z_end = int(valid_indices.max())

        if z_start > z_end:
            log.error(
                "Threshold method: Z-crop calc start > end (%d > %d).", z_start, z_end
            )
            return (0, depth - 1)

        log.debug(
            "Threshold Z-crop range determined: Keep slices %d to %d.", z_start, z_end
        )
        return z_start, z_end

    except MemoryError:
        log.error(
            "MemoryError calculating MIPs/profile for threshold crop. Skipping crop."
        )
        return (0, depth - 1)
    except Exception as e:
        log.error(
            "Unexpected error during threshold Z-crop calculation: %s",
            e,
            exc_info=True,
        )
        return (0, depth - 1)


# --- Z-Cropping Method 2: Slope Analysis (Default) ---
# _find_z_crop_range_slope remains the same
def _find_z_crop_range_slope(
    volume: np.ndarray,
    debug: bool = False,
    output_folder: Optional[Path] = None,
    filename_prefix: str = "debug",
) -> Tuple[int, int]:
    """
    Finds Z-crop range based on the slope of the smoothed max intensity profile
    derived from YZ and XZ MIPs.
    """
    if volume.ndim != 3 or volume.shape[0] == 0:
        log.warning(
            "Slope method: Cannot crop non-3D/empty volume shape %s", volume.shape
        )
        return (0, 0)

    depth = volume.shape[0]
    if depth <= (2 * MIN_SEARCH_OFFSET_Z + SLOPE_WINDOW_Z):
        log.warning("Slope method: Volume depth %d too small. Skipping crop.", depth)
        return (0, depth - 1 if depth > 0 else 0)

    z_start_found = 0
    z_end_found = depth - 1
    max_z_profile = np.array([0.0])
    smoothed_profile = np.array([0.0])
    slope = np.array([0.0])
    # Define mip variables outside try for finally block check
    mip_yz = None
    mip_xz = None

    try:
        log.debug("Calculating MIPs for slope-based Z-cropping...")
        mip_yz = np.max(volume, axis=2)
        mip_xz = np.max(volume, axis=1)
        max_per_z_yz = np.max(mip_yz, axis=1)
        max_per_z_xz = np.max(mip_xz, axis=1)
        max_z_profile = np.maximum(max_per_z_yz, max_per_z_xz)
        log.debug(
            "Raw Max Z profile range: [%.1f, %.1f]",
            np.min(max_z_profile),
            np.max(max_z_profile),
        )

        log.debug(
            "Applying Gaussian smoothing (sigma=%.1f) to Z profile...",
            SMOOTHING_SIGMA_Z,
        )
        smoothed_profile = gaussian_filter1d(
            max_z_profile, sigma=SMOOTHING_SIGMA_Z, mode="reflect"
        )
        log.debug(
            "Smoothed Max Z profile range: [%.1f, %.1f]",
            np.min(smoothed_profile),
            np.max(smoothed_profile),
        )

        slope = np.gradient(smoothed_profile)
        log.debug("Slope range: [%.2f, %.2f]", np.min(slope), np.max(slope))

        # Find Start Slice
        found_start = False
        start_search_end = depth - SLOPE_WINDOW_Z
        if MIN_SEARCH_OFFSET_Z < start_search_end:
            for i in range(MIN_SEARCH_OFFSET_Z, start_search_end):
                window = slope[i : i + SLOPE_WINDOW_Z]
                if np.all(window > SLOPE_THRESH_Z_POS):
                    z_start_found = i
                    found_start = True
                    log.debug(
                        "Found potential Z start at index %d (slope > %d)",
                        i,
                        SLOPE_THRESH_Z_POS,
                    )
                    break

        # Find End Slice
        found_end = False
        end_search_start = depth - MIN_SEARCH_OFFSET_Z - 1
        end_search_end = SLOPE_WINDOW_Z - 1
        if end_search_start > end_search_end:
            for i in range(end_search_start, end_search_end, -1):
                # Ensure window start index is not negative
                window_start_idx = i - SLOPE_WINDOW_Z + 1
                if window_start_idx < 0:
                    continue
                window = slope[window_start_idx : i + 1]
                if np.all(window < -SLOPE_THRESH_Z_POS):
                    z_end_found = i + 1  # Use i+1 to make range inclusive of drop-off
                    found_end = True
                    log.debug(
                        "Found potential Z end at index %d (slope < %d)",
                        i + 1,
                        -SLOPE_THRESH_Z_POS,
                    )
                    break

        # Apply Fallbacks if needed
        if not found_start:
            log.warning("Could not reliably determine Z start via slope. Using Z=0.")
            z_start_found = 0
        if not found_end:
            log.warning(
                "Could not reliably determine Z end via slope. Using Z=%d.", depth - 1
            )
            z_end_found = depth - 1

        # Final sanity check
        if z_start_found >= z_end_found:
            log.warning(
                "Z-crop slope analysis resulted in start >= end (%d >= %d). Using original range.",
                z_start_found,
                z_end_found,
            )
            z_start_found = 0
            z_end_found = depth - 1

        log.info(
            "Slope-based Z-crop range determined: Keep slices %d to %d.",
            z_start_found,
            z_end_found,
        )

    except MemoryError:
        log.error("MemoryError during Z-crop calculation. Skipping crop.")
        z_start_found = 0
        z_end_found = depth - 1  # Use defaults on error
    except Exception as e:
        log.error(
            "Unexpected error during slope-based Z-crop calculation: %s",
            e,
            exc_info=True,
        )
        z_start_found = 0
        z_end_found = depth - 1  # Use defaults on error

    # --- Save Debug Images/Plots if requested ---
    if debug and output_folder is not None and MATPLOTLIB_AVAILABLE and plt is not None:
        assert plt is not None
        log.debug("Saving Z-crop debug info for prefix: %s", filename_prefix)
        output_folder.mkdir(parents=True, exist_ok=True)

        # Save MIPs (Check if they were calculated before potential error)
        if mip_yz is not None and mip_xz is not None:
            mip_limits = calculate_limits_only(volume, stretch_mode="max")
            mip_yz_path = output_folder / f"{filename_prefix}_debug_mip_yz.png"
            _save_debug_array_as_image(mip_yz, mip_yz_path, mip_limits)
            mip_xz_path = output_folder / f"{filename_prefix}_debug_mip_xz.png"
            _save_debug_array_as_image(mip_xz, mip_xz_path, mip_limits)
        else:
            log.warning("MIP arrays not available for debug saving.")

        # Save Z-Profile Plot with Slope
        z_indices = np.arange(depth)
        profile_plot_path = (
            output_folder / f"{filename_prefix}_debug_z_profile_slope.png"
        )
        fig_prof = None
        try:
            fig_prof, ax1 = plt.subplots(figsize=(10, 6))
            color1 = "tab:blue"
            ax1.set_xlabel("Z Slice Index")
            ax1.set_ylabel("Smoothed Max Intensity", color=color1)
            ax1.plot(
                z_indices,
                smoothed_profile,
                color=color1,
                label=f"Smoothed Profile (sigma={SMOOTHING_SIGMA_Z:.1f})",
            )
            ax1.tick_params(axis="y", labelcolor=color1)
            ax1.grid(True, linestyle=":", alpha=0.6)
            ax1.plot(
                z_indices,
                max_z_profile,
                color="lightblue",
                alpha=0.4,
                label="Raw Max Profile",
            )

            ax2 = ax1.twinx()
            color2 = "tab:red"
            ax2.set_ylabel("Slope", color=color2)
            ax2.plot(
                z_indices,
                slope,
                color=color2,
                linestyle="--",
                alpha=0.8,
                label="Slope",
            )
            ax2.tick_params(axis="y", labelcolor=color2)
            ax2.axhline(0, color="gray", linestyle=":", linewidth=0.5)
            ax2.axhline(
                SLOPE_THRESH_Z_POS,
                color="pink",
                linestyle=":",
                linewidth=0.8,
                label=f"Pos Slope Thresh ({SLOPE_THRESH_Z_POS})",
            )
            ax2.axhline(
                -SLOPE_THRESH_Z_POS,
                color="pink",
                linestyle=":",
                linewidth=0.8,
                label=f"Neg Slope Thresh ({-SLOPE_THRESH_Z_POS})",
            )

            ax1.axvline(
                z_start_found,
                color="green",
                linestyle=":",
                label=f"Z Start ({z_start_found})",
            )
            # Use z_end_found - 1 for visual line if range is inclusive end
            ax1.axvline(
                z_end_found,
                color="lime",
                linestyle=":",
                label=f"Z End ({z_end_found})",
            )

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(
                lines1 + lines2,
                labels1 + labels2,
                loc="upper right",
                fontsize="small",
            )

            ax1.set_title(f"Z-Crop Slope Analysis - {filename_prefix}")
            fig_prof.tight_layout()
            log.debug("Saving Z-profile slope plot: %s", profile_plot_path)
            fig_prof.savefig(str(profile_plot_path), dpi=100)
        except Exception as plot_e:
            log.error(
                "Failed to generate or save Z-profile slope plot: %s",
                plot_e,
                exc_info=True,
            )
        finally:
            if fig_prof is not None:
                plt.close(fig_prof)
    # --- End Debug Saving ---

    return z_start_found, z_end_found


# --- Main Z-Crop Wrapper Function ---
# find_z_crop_range remains the same
def find_z_crop_range(
    volume: np.ndarray,
    method: str,
    threshold: int,
    debug: bool = False,
    output_folder: Optional[Path] = None,
    filename_prefix: str = "debug",
) -> Tuple[int, int]:
    """Wrapper function to select Z-cropping method."""
    if method == "slope":
        log.info("Using slope analysis for Z-cropping.")
        # Pass necessary debug args to slope function
        return _find_z_crop_range_slope(volume, debug, output_folder, filename_prefix)
    elif method == "threshold":
        log.info("Using simple threshold (%d) for Z-cropping.", threshold)
        return _find_z_crop_range_threshold(volume, threshold)
    else:
        log.warning("Unknown z_crop_method '%s'. Defaulting to 'slope'.", method)
        return _find_z_crop_range_slope(volume, debug, output_folder, filename_prefix)


# --- TIFF Volume Extraction (Loads FULL data) --- <<< MODIFIED DOCSTRING
def extract_original_volume(tif_path: Path) -> Optional[np.ndarray]:
    """
    Extracts and reshapes the volume data from a TIFF file to 3D (Z, Y, X).
    Note: This function loads the full pixel data.
    """
    log.debug("Extracting original volume from: %s", tif_path)
    if not tif_path.is_file():
        log.error("TIFF file not found: %s", tif_path)
        return None
    try:
        with tifffile.TiffFile(str(tif_path)) as tif:
            vol: np.ndarray = tif.asarray()  # <<< LOADS PIXEL DATA
        log.debug("Original TIFF shape: %s", vol.shape)

        squeezed_vol = np.squeeze(vol)
        log.debug("Squeezed TIFF shape: %s", squeezed_vol.shape)

        final_vol: np.ndarray
        if squeezed_vol.ndim == 3:
            final_vol = squeezed_vol
        elif squeezed_vol.ndim == 2:
            log.debug("Shape is 2D, adding Z dim.")
            final_vol = squeezed_vol[np.newaxis, :, :]
        else:
            log.warning(
                "Original shape %s was not 2D or 3D after squeezing. Assuming last 3 dimensions are ZYX.",
                vol.shape,
            )
            if squeezed_vol.ndim >= 3:
                # Ensure we handle potential negative strides correctly if needed
                final_vol = squeezed_vol[..., -3:, :, :]
                if final_vol.ndim != 3:  # Check result
                    raise ValueError(f"Could not resolve to 3D shape from {vol.shape}")
            else:
                raise ValueError(
                    f"Unsupported shape after squeeze: {squeezed_vol.shape}"
                )

        if final_vol.ndim != 3:
            raise ValueError(
                f"Volume shape is not 3D after processing: {final_vol.shape}"
            )

        log.debug("Final extracted original volume shape: %s", final_vol.shape)
        return final_vol
    except Exception as e:
        log.error("Error extracting %s: %s", tif_path.name, e, exc_info=True)
        return None


# --- NEW: TIFF Dimension Extraction from Metadata ---
def get_dimensions_from_metadata(tif_path: Path) -> Optional[Tuple[int, int, int]]:
    """
    Extracts dimensions (Depth, Height, Width) from TIFF metadata without
    loading pixel data. Assumes ZYX order if ndim > 3.
    """
    log.debug("Extracting dimensions from metadata: %s", tif_path)
    if not tif_path.is_file():
        log.error("TIFF file not found for metadata read: %s", tif_path)
        return None
    try:
        with tifffile.TiffFile(str(tif_path)) as tif:
            # Prefer series shape if available (handles multi-series files better)
            if tif.series and len(tif.series) > 0:
                shape = tif.series[0].shape
                log.debug("Using series[0] shape from metadata: %s", shape)
            else:
                # Fallback to page shape (might be less reliable for complex TIFFs)
                if len(tif.pages) > 0:
                    shape = tif.pages[0].shape
                    log.warning(
                        "Using page[0] shape from metadata (series not found): %s",
                        shape,
                    )
                else:
                    log.error(
                        "Could not find series or pages in TIFF metadata: %s",
                        tif_path.name,
                    )
                    return None

        # Process shape to get Z, Y, X
        squeezed_shape = tuple(dim for dim in shape if dim != 1)
        log.debug("Squeezed shape from metadata: %s", squeezed_shape)

        if len(squeezed_shape) == 3:  # Z, Y, X
            d, h, w = squeezed_shape
        elif len(squeezed_shape) == 2:  # Y, X
            log.debug("Metadata shape is 2D, setting depth=1.")
            h, w = squeezed_shape
            d = 1
        elif len(squeezed_shape) > 3:  # Assume last 3 are ZYX
            log.warning(
                "Metadata shape has >3 non-singleton dims (%s). Assuming last 3 are ZYX.",
                squeezed_shape,
            )
            d, h, w = squeezed_shape[-3:]
        else:
            log.error("Unsupported shape derived from metadata: %s", squeezed_shape)
            return None

        log.debug("Derived dimensions from metadata: D=%d, H=%d, W=%d", d, h, w)
        return d, h, w

    except Exception as e:
        log.error("Error reading metadata from %s: %s", tif_path.name, e, exc_info=True)
        return None


# --- Preview Slice Saving ---
# save_preview_slice remains the same
def save_preview_slice(vol_8bit: np.ndarray, path: Path):
    """Saves the middle Z-slice of an 8-bit volume as PNG."""
    if vol_8bit.ndim != 3 or vol_8bit.shape[0] == 0:
        log.warning("Cannot save preview: invalid shape %s", vol_8bit.shape)
        return
    if vol_8bit.dtype != np.uint8:
        log.warning(
            "Preview input should be uint8, but got: %s. Attempting conversion.",
            vol_8bit.dtype,
        )
        if np.can_cast(vol_8bit, np.uint8):
            try:
                max_val = np.max(vol_8bit)
                if max_val > 0:
                    vol_8bit = (vol_8bit.astype(float) / max_val * 255).astype(np.uint8)
                else:
                    vol_8bit = vol_8bit.astype(np.uint8)
            except Exception:
                log.error(
                    "Failed to auto-convert preview input dtype %s to uint8.",
                    vol_8bit.dtype,
                )
                return
        else:
            log.error(
                "Cannot safely cast preview input dtype %s to uint8.", vol_8bit.dtype
            )
            return

    mid_z = vol_8bit.shape[0] // 2
    slice_2d = vol_8bit[mid_z]
    try:
        img_pil = Image.fromarray(slice_2d)
        log.debug("Saving preview: %s", path)
        img_pil.save(str(path), format="PNG")
    except Exception as e:
        log.error("Failed save preview %s: %s", path, e, exc_info=True)


# --- Debug Histogram Saving ---
# save_histogram_debug remains the same
def save_histogram_debug(
    img: np.ndarray, limits: ContrastLimits, out_path: Path, stretch_mode: str
):
    """Saves a debug histogram plot showing intensity distribution and limits."""
    if not MATPLOTLIB_AVAILABLE or plt is None:
        log.warning("Matplotlib unavailable, skipping histogram.")
        return
    assert plt is not None
    fig = None
    try:
        img = np.asarray(img)
        finite_pixels = img[np.isfinite(img)]
        if finite_pixels.size == 0:
            log.warning("No finite pixels for hist %s. Skipping.", out_path.name)
            return

        nonzero_pixels = finite_pixels[finite_pixels > 0].flatten()
        if nonzero_pixels.size == 0:
            log.warning(
                "No positive pixel values for hist %s. Skipping.", out_path.name
            )
            return

        vmin = (
            limits.actual_min
            if limits.actual_min is not None and np.isfinite(limits.actual_min)
            else float(np.min(finite_pixels))
        )
        vmax = (
            limits.actual_max
            if limits.actual_max is not None and np.isfinite(limits.actual_max)
            else float(np.max(finite_pixels))
        )

        if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmax <= vmin:
            log.warning(
                "Invalid range [%s, %s] for hist %s. Skipping.",
                vmin,
                vmax,
                out_path.name,
            )
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(
            nonzero_pixels,
            bins=256,
            range=(vmin, vmax),
            color="gray",
            alpha=0.7,
            log=True,
            label="Log Histogram (Non-zero Pixels)",
        )

        plot_markers = {
            f"Stretch Low ({limits.p_low:.1f})": (limits.p_low, "red", "-"),
            f"Stretch High ({limits.p_high:.1f})": (limits.p_high, "red", "-"),
            (f"1% ({limits.p1:.1f})" if limits.p1 is not None else None): (
                limits.p1,
                "blue",
                "--",
            ),
            (f"ImageJ Min ({limits.p035:.1f})" if limits.p035 is not None else None): (
                limits.p035,
                "orange",
                "--",
            ),
            (
                f"Smart Early ({limits.smart_early:.1f})"
                if limits.smart_early is not None
                else None
            ): (limits.smart_early, "purple", "--"),
            (
                f"Smart Late ({limits.smart_late:.1f})"
                if limits.smart_late is not None
                else None
            ): (limits.smart_late, "green", "--"),
            (
                f"ImageJ Max ({limits.p9965:.1f})" if limits.p9965 is not None else None
            ): (limits.p9965, "cyan", "--"),
            (
                f"Actual Max ({limits.actual_max:.1f})"
                if limits.actual_max is not None
                else None
            ): (limits.actual_max, "magenta", ":"),
            (
                f"Actual Min ({limits.actual_min:.1f})"
                if limits.actual_min is not None
                else None
            ): (limits.actual_min, "yellow", ":"),
        }

        added_labels = set()
        y_min, y_max = ax.get_ylim()
        if y_min <= 0:
            y_min = min(1.0, y_max * 0.1) if y_max > 0 else 1e-1

        for label, (value, color, style) in plot_markers.items():
            if (
                label is not None
                and value is not None
                and np.isfinite(value)
                and value >= vmin
                and value <= vmax
            ):
                if label not in added_labels:
                    ax.vlines(
                        value, y_min, y_max, color=color, linestyle=style, label=label
                    )
                    added_labels.add(label)

        ax.set_title(
            f"Intensity Histogram Debug – Stretch Mode: '{stretch_mode}'\n"
            f"(Bounds Used: [{limits.p_low:.1f} - {limits.p_high:.1f}])"
        )
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Log Frequency (Count)")
        ax.set_ylim(bottom=y_min)
        ax.legend(fontsize="small")
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        ax.set_xlim(vmin, vmax)
        log.debug("Saving histogram: %s", out_path)
        fig.savefig(str(out_path), dpi=100)

    except Exception as e:
        log.error("Failed histogram %s: %s", out_path, e, exc_info=True)
    finally:
        if fig is not None:
            plt.close(fig)


# --- Main Channel Processing Function ---
# process_channel remains the same
def process_channel(
    time_id: str,
    ch_id: int,
    globally_cropped_vol: np.ndarray,
    layout: VolumeLayout,
    limits: ContrastLimits,
    stretch_mode: str,
    dry_run: bool = False,
    debug: bool = False,
    output_folder: str = ".",
) -> Optional[Dict[str, Any]]:
    """
    Processes a single channel: applies contrast, tiles using scikit-image montage
    (or NumPy fallback), and saves.

    Args:
        time_id: Identifier for the timepoint.
        ch_id: Channel identifier.
        globally_cropped_vol: The 3D NumPy array (Z, Y, X) already cropped
                              to the global Z range.
        layout: The VolumeLayout defining the tiling grid.
        limits: The ContrastLimits to apply for stretching.
        stretch_mode: The stretch mode name (used for histogram saving).
        dry_run: If True, skip saving output files.
        debug: If True, save debug histogram/preview.
        output_folder: The base directory for saving output files.

    Returns:
        A dictionary with result metadata if successful, otherwise None.
    """
    output_folder_obj = Path(output_folder)
    log.info(
        "Processing T:%s C:%d - Received Globally Cropped Shape: %s",
        time_id,
        ch_id,
        globally_cropped_vol.shape,
    )
    result_data: Optional[Dict[str, Any]] = None
    try:
        # 1. Apply Contrast Stretching
        vol_8bit, applied_limits = apply_autocontrast_8bit(
            globally_cropped_vol,
            stretch_mode="max",  # Use pre-calculated limits directly
            global_limits_tuple=(limits.p_low, limits.p_high),
        )
        final_p_low = applied_limits.p_low
        final_p_high = applied_limits.p_high
        log.debug(
            "T:%s C:%d - Applied Stretched Range: (%.2f, %.2f)",
            time_id,
            ch_id,
            final_p_low,
            final_p_high,
        )

        # 2. Save Debug Histogram (if needed)
        if debug and not dry_run:
            hist_filename = f"debug_hist_T{time_id}_C{ch_id}.png"
            hist_path = output_folder_obj / hist_filename
            debug_limits_for_hist = ContrastLimits(
                p_low=final_p_low, p_high=final_p_high
            )
            debug_limits_for_hist.actual_min = limits.actual_min
            debug_limits_for_hist.actual_max = limits.actual_max
            save_histogram_debug(
                globally_cropped_vol, debug_limits_for_hist, hist_path, stretch_mode
            )

        out_file = f"volume_{time_id}_c{ch_id}.webp"

        # 3. Perform Tiling using skimage.util.montage or NumPy fallback
        if not dry_run:
            tiled_array: Optional[np.ndarray] = None
            use_skimage = SKIMAGE_AVAILABLE and skimage_montage is not None

            if use_skimage:
                log.debug(
                    "T:%s C:%d - Creating tile array using skimage.util.montage...",
                    time_id,
                    ch_id,
                )
                try:
                    assert skimage_montage is not None
                    tiled_array = skimage_montage(
                        arr_in=vol_8bit,
                        grid_shape=(layout.rows, layout.cols),
                        fill=0,
                        padding_width=0,
                        rescale_intensity=False,
                    )
                    if tiled_array.dtype != np.uint8:
                        log.warning(
                            "skimage.montage returned dtype %s, converting back to uint8.",
                            tiled_array.dtype,
                        )
                        tiled_array = tiled_array.astype(np.uint8)

                    expected_shape = (layout.tile_height, layout.tile_width)
                    if tiled_array.shape != expected_shape:
                        log.error(
                            "Shape mismatch from skimage.montage: Got %s, Expected %s. Falling back to NumPy loop.",
                            tiled_array.shape,
                            expected_shape,
                        )
                        tiled_array = None  # Force fallback

                except Exception as montage_e:
                    log.error(
                        "skimage.util.montage failed: %s. Falling back to NumPy loop.",
                        montage_e,
                        exc_info=debug,
                    )
                    tiled_array = None  # Force fallback

            # --- NumPy Fallback / Default Logic ---
            if tiled_array is None:
                if use_skimage:  # Log only if fallback occurred
                    log.debug(
                        "T:%s C:%d - Using NumPy loop for tiling (fallback).",
                        time_id,
                        ch_id,
                    )
                else:
                    log.debug(
                        "T:%s C:%d - Using NumPy loop for tiling.", time_id, ch_id
                    )

                tiled_array = np.zeros(
                    (layout.tile_height, layout.tile_width), dtype=np.uint8
                )
                num_slices_in_volume = vol_8bit.shape[0]
                for i in range(layout.depth):  # Iterate up to expected layout depth
                    if i >= num_slices_in_volume:
                        # Pad with zeros if volume is shallower than layout depth
                        log.debug(
                            "Padding tile grid for slice index %d (volume depth %d)",
                            i,
                            num_slices_in_volume,
                        )
                        continue  # Skip pasting, leave as zeros

                    paste_col = i % layout.cols
                    paste_row = i // layout.cols
                    y_start = paste_row * layout.height
                    y_end = y_start + layout.height
                    x_start = paste_col * layout.width
                    x_end = x_start + layout.width

                    # Bounds check for safety
                    if y_end > layout.tile_height or x_end > layout.tile_width:
                        log.error(
                            "NumPy Tiling: Calculated paste coords OOB [%d:%d, %d:%d] for slice %d. Skipping.",
                            y_start,
                            y_end,
                            x_start,
                            x_end,
                            i,
                        )
                        continue
                    try:
                        tiled_array[y_start:y_end, x_start:x_end] = vol_8bit[i, :, :]
                    except ValueError as e:
                        log.error(
                            "NumPy Tiling: Error assigning slice %d (Shape %s into [%d:%d, %d:%d]). Error: %s",
                            i,
                            vol_8bit[i, :, :].shape,
                            y_start,
                            y_end,
                            x_start,
                            x_end,
                            e,
                        )
                        continue
            # --- End Tiling Logic ---

            # 4. Convert final NumPy array to PIL Image
            log.debug(
                "T:%s C:%d - Converting tiled NumPy array to PIL Image.", time_id, ch_id
            )
            try:
                tiled_img = Image.fromarray(tiled_array)
            except Exception as img_e:
                log.error(
                    "Failed to create PIL Image from tiled NumPy array: %s",
                    img_e,
                    exc_info=True,
                )
                return None

            # 5. Save the Tiled Image
            out_path = output_folder_obj / out_file
            log.debug("Attempting to save tiled WebP image to: %s", out_path)
            try:
                output_folder_obj.mkdir(parents=True, exist_ok=True)
            except Exception as mkdir_e:
                log.error(
                    "Failed to create output directory %s before saving %s: %s",
                    output_folder_obj,
                    out_file,
                    mkdir_e,
                    exc_info=True,
                )
                return None
            try:
                tiled_img.save(
                    str(out_path),
                    format="WEBP",
                    quality=WEBP_QUALITY,
                    method=WEBP_METHOD,
                    lossless=WEBP_LOSSLESS,
                )
            except Exception as e:
                log.error("Failed save WebP %s: %s", out_path, e, exc_info=True)
                return None

        # 6. Save Debug Preview (if needed)
        if debug and not dry_run and ch_id == 0:
            preview_filename = f"preview_T{time_id}_C{ch_id}.png"
            preview_path = output_folder_obj / preview_filename
            save_preview_slice(vol_8bit, preview_path)

        # 7. Prepare Result Metadata
        result_data = {
            "time_id": time_id,
            "channel": ch_id,
            "filename": out_file,
            "intensity_range": {
                "p_low": final_p_low,
                "p_high": final_p_high,
            },
        }
        log.info("Successfully processed T:%s C:%d", time_id, ch_id)

    except IndexError as e:
        log.error(
            "❌ IndexError T:%s C:%d: %s. Vol shape: %s, Layout depth: %d",
            time_id,
            ch_id,
            e,
            globally_cropped_vol.shape,
            layout.depth,
            exc_info=True,
        )
        result_data = None
    except Exception as e:
        log.error("❌ Unexpected Error T:%s C:%d: %s", time_id, ch_id, e, exc_info=True)
        result_data = None

    return result_data
