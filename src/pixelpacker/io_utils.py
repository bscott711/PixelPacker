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
def _save_debug_array_as_image(
    arr: np.ndarray, filename: Path, limits: Optional[ContrastLimits] = None
):
    """Scales a 2D numpy array and saves as PNG."""
    if arr.ndim != 2:
        log.warning(f"Cannot save debug array {filename.name}: Not 2D.")
        return
    if arr.size == 0:
        log.warning(f"Cannot save empty debug array: {filename.name}")
        return
    try:
        arr_8bit: np.ndarray
        if limits:
            log.debug(
                f"Scaling debug image {filename.name} using limits:"
                f" [{limits.p_low:.1f}-{limits.p_high:.1f}]"
            )
            p_low = limits.p_low
            p_high = limits.p_high
            if p_high <= p_low:
                arr_8bit = np.where(arr <= p_low, 0, 255).astype(np.uint8)
            else:
                arr_float = arr.astype(np.float32)
                denominator = p_high - p_low
                scaled_float = (arr_float - p_low) / denominator
                scaled_float = np.clip(scaled_float, 0.0, 1.0)
                arr_8bit = (scaled_float * 255.0).astype(np.uint8)
        else:
            log.debug(f"Scaling debug image {filename.name} using min-max.")
            arr_min = np.min(arr)
            arr_max = np.max(arr)
            if arr_max > arr_min:
                arr_norm = (arr.astype(np.float32) - arr_min) / (arr_max - arr_min)
                arr_8bit = (np.clip(arr_norm, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                # Flat image
                arr_8bit = np.zeros_like(arr, dtype=np.uint8)

        img_pil = Image.fromarray(arr_8bit)
        log.debug(f"Saving debug image: {filename}")
        img_pil.save(str(filename), format="PNG")
    except Exception as e:
        log.error(f"Failed to save debug array image {filename}: {e}", exc_info=True)


# --- Z-Cropping Method 1: Simple Threshold on Max Profile ---
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
            f"Max Z profile range: [{np.min(max_z_profile)}, {np.max(max_z_profile)}]"
        )

        valid_indices = np.where(max_z_profile > threshold)[0]

        if valid_indices.size == 0:
            log.warning(
                f"Threshold method: No Z-slices found with max projection"
                f" intensity > {threshold}. Cannot crop."
            )
            return (0, depth - 1)

        z_start = int(valid_indices.min())
        z_end = int(valid_indices.max())

        if z_start > z_end:
            log.error(
                f"Threshold method: Z-crop calc start > end ({z_start} > {z_end})."
            )
            return (0, depth - 1)

        log.debug(
            f"Threshold Z-crop range determined: Keep slices {z_start} to {z_end}."
        )
        return z_start, z_end

    except MemoryError:
        log.error(
            "MemoryError calculating MIPs/profile for threshold crop. Skipping crop."
        )
        return (0, depth - 1)
    except Exception as e:
        log.error(
            f"Unexpected error during threshold Z-crop calculation: {e}",
            exc_info=True,
        )
        return (0, depth - 1)


# --- Z-Cropping Method 2: Slope Analysis (Default) ---
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
            f"Slope method: Cannot crop non-3D/empty volume shape {volume.shape}"
        )
        return (0, 0)

    depth = volume.shape[0]
    if depth <= (2 * MIN_SEARCH_OFFSET_Z + SLOPE_WINDOW_Z):
        log.warning(f"Slope method: Volume depth {depth} too small. Skipping crop.")
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
            f"Raw Max Z profile range:"
            f" [{np.min(max_z_profile):.1f}, {np.max(max_z_profile):.1f}]"
        )

        log.debug(
            f"Applying Gaussian smoothing (sigma={SMOOTHING_SIGMA_Z}) to Z profile..."
        )
        smoothed_profile = gaussian_filter1d(
            max_z_profile, sigma=SMOOTHING_SIGMA_Z, mode="reflect"
        )
        log.debug(
            f"Smoothed Max Z profile range:"
            f" [{np.min(smoothed_profile):.1f}, {np.max(smoothed_profile):.1f}]"
        )

        slope = np.gradient(smoothed_profile)
        log.debug(f"Slope range: [{np.min(slope):.2f}, {np.max(slope):.2f}]")

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
                        f"Found potential Z start at index {i}"
                        f" (slope > {SLOPE_THRESH_Z_POS})"
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
                    z_end_found = i + 1
                    found_end = True
                    log.debug(
                        f"Found potential Z end at index {i + 1}"
                        f" (slope < {-SLOPE_THRESH_Z_POS})"
                    )
                    break

        # Apply Fallbacks if needed
        if not found_start:
            log.warning("Could not reliably determine Z start via slope. Using Z=0.")
            z_start_found = 0
        if not found_end:
            log.warning(
                f"Could not reliably determine Z end via slope. Using Z={depth - 1}."
            )
            z_end_found = depth - 1

        # Final sanity check
        if z_start_found >= z_end_found:
            log.warning(
                f"Z-crop slope analysis resulted in start >= end"
                f" ({z_start_found} >= {z_end_found}). Using original range."
            )
            z_start_found = 0
            z_end_found = depth - 1

        log.info(
            f"Slope-based Z-crop range determined: Keep slices"
            f" {z_start_found} to {z_end_found}."
        )

    except MemoryError:
        log.error("MemoryError during Z-crop calculation. Skipping crop.")
        z_start_found = 0
        z_end_found = depth - 1  # Use defaults on error
    except Exception as e:
        log.error(
            f"Unexpected error during slope-based Z-crop calculation: {e}",
            exc_info=True,
        )
        z_start_found = 0
        z_end_found = depth - 1  # Use defaults on error

    # --- Save Debug Images/Plots if requested ---
    if debug and output_folder is not None and MATPLOTLIB_AVAILABLE and plt is not None:
        assert plt is not None
        log.debug(f"Saving Z-crop debug info for prefix: {filename_prefix}")
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
                label=f"Smoothed Profile (sigma={SMOOTHING_SIGMA_Z})",
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
            log.debug(f"Saving Z-profile slope plot: {profile_plot_path}")
            fig_prof.savefig(str(profile_plot_path), dpi=100)
        except Exception as plot_e:
            log.error(
                f"Failed to generate or save Z-profile slope plot: {plot_e}",
                exc_info=True,
            )
        finally:
            if fig_prof is not None:
                plt.close(fig_prof)
    # --- End Debug Saving ---

    return z_start_found, z_end_found


# --- Main Z-Crop Wrapper Function ---
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
        log.info(f"Using simple threshold ({threshold}) for Z-cropping.")
        return _find_z_crop_range_threshold(volume, threshold)
    else:
        log.warning(f"Unknown z_crop_method '{method}'. Defaulting to 'slope'.")
        return _find_z_crop_range_slope(volume, debug, output_folder, filename_prefix)


# --- TIFF Volume Extraction ---
def extract_original_volume(tif_path: Path) -> Optional[np.ndarray]:
    """Extracts and reshapes the volume from a TIFF file to 3D (Z, Y, X)."""
    log.debug(f"Extracting original volume from: {tif_path}")
    if not tif_path.is_file():
        log.error(f"TIFF file not found: {tif_path}")
        return None
    try:
        with tifffile.TiffFile(str(tif_path)) as tif:
            vol: np.ndarray = tif.asarray()
        log.debug(f"Original TIFF shape: {vol.shape}")

        squeezed_vol = np.squeeze(vol)
        log.debug(f"Squeezed TIFF shape: {squeezed_vol.shape}")

        final_vol: np.ndarray
        if squeezed_vol.ndim == 3:
            final_vol = squeezed_vol
        elif squeezed_vol.ndim == 2:
            log.debug("Shape is 2D, adding Z dim.")
            final_vol = squeezed_vol[np.newaxis, :, :]
        else:
            # Attempt to handle common 4D/5D cases by keeping last 3 dims
            # Warn the user as this is making assumptions
            log.warning(
                f"Original shape {vol.shape} was not 2D or 3D after squeezing."
                " Assuming last 3 dimensions are ZYX."
            )
            if squeezed_vol.ndim >= 3:
                final_vol = squeezed_vol[..., -3:, :, :]
                # Check if the result is actually 3D
                if final_vol.ndim != 3:
                    raise ValueError(f"Could not resolve to 3D shape from {vol.shape}")
            else:
                raise ValueError(
                    f"Unsupported shape after squeeze: {squeezed_vol.shape}"
                )

        if final_vol.ndim != 3:
            # This should ideally not be reached if logic above is correct
            raise ValueError(
                f"Volume shape is not 3D after processing: {final_vol.shape}"
            )

        log.debug(f"Final extracted original volume shape: {final_vol.shape}")
        return final_vol
    except Exception as e:
        log.error(f"Error extracting {tif_path.name}: {e}", exc_info=True)
        return None


# --- Preview Slice Saving ---
def save_preview_slice(vol_8bit: np.ndarray, path: Path):
    """Saves the middle Z-slice of an 8-bit volume as PNG."""
    if vol_8bit.ndim != 3 or vol_8bit.shape[0] == 0:
        log.warning(f"Cannot save preview: invalid shape {vol_8bit.shape}")
        return
    if vol_8bit.dtype != np.uint8:
        log.warning(
            f"Preview input should be uint8, but got: {vol_8bit.dtype}."
            " Attempting conversion."
        )
        # Attempt conversion if possible, e.g., from boolean or low-value int
        if np.can_cast(vol_8bit, np.uint8):
            try:
                # Scale assuming max value might be > 0
                max_val = np.max(vol_8bit)
                if max_val > 0:
                    vol_8bit = (vol_8bit.astype(float) / max_val * 255).astype(np.uint8)
                else:
                    vol_8bit = vol_8bit.astype(np.uint8)  # Already zeros
            except Exception:
                log.error(
                    f"Failed to auto-convert preview input dtype {vol_8bit.dtype} to uint8."
                )
                return
        else:
            log.error(
                f"Cannot safely cast preview input dtype {vol_8bit.dtype} to uint8."
            )
            return

    mid_z = vol_8bit.shape[0] // 2
    slice_2d = vol_8bit[mid_z]
    try:
        img_pil = Image.fromarray(slice_2d)
        log.debug(f"Saving preview: {path}")
        img_pil.save(str(path), format="PNG")
    except Exception as e:
        log.error(f"Failed save preview {path}: {e}", exc_info=True)


# --- Debug Histogram Saving ---
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
        # Filter out non-finite values before calculating histogram
        finite_pixels = img[np.isfinite(img)]
        if finite_pixels.size == 0:
            log.warning(f"No finite pixels for hist {out_path.name}. Skipping.")
            return

        nonzero_pixels = finite_pixels[finite_pixels > 0].flatten()
        if nonzero_pixels.size == 0:
            log.warning(f"No positive pixel values for hist {out_path.name}. Skipping.")
            return

        # Use actual min/max from limits if available and finite, else calculate
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
                f"Invalid range [{vmin}, {vmax}] for hist {out_path.name}. Skipping."
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
        if y_min <= 0:  # Ensure log scale doesn't start at or below zero
            y_min = min(1.0, y_max * 0.1)  # Adjust y_min if necessary

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
        ax.set_ylim(bottom=y_min)  # Set adjusted bottom ylim
        ax.legend(fontsize="small")
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        ax.set_xlim(vmin, vmax)
        log.debug(f"Saving histogram: {out_path}")
        fig.savefig(str(out_path), dpi=100)

    except Exception as e:
        log.error(f"Failed histogram {out_path}: {e}", exc_info=True)
    finally:
        if fig is not None:
            plt.close(fig)


# --- Main Channel Processing Function ---
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
        f"Processing T:{time_id} C:{ch_id} - Received Globally Cropped Shape:"
        f" {globally_cropped_vol.shape}"
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
            f"T:{time_id} C:{ch_id} - Applied Stretched Range:"
            f" ({final_p_low:.2f}, {final_p_high:.2f})"
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
                    f"T:{time_id} C:{ch_id} - Creating tile array using"
                    " skimage.util.montage..."
                )
                try:
                    # skimage.util.montage expects (num_images, height, width)
                    # Our vol_8bit is (Z, Y, X), which matches (N=Z, H=Y, W=X)
                    assert skimage_montage is not None  # Help type checker
                    tiled_array = skimage_montage(
                        arr_in=vol_8bit,
                        grid_shape=(layout.rows, layout.cols),
                        fill=0,
                        padding_width=0,
                        rescale_intensity=False,
                    )
                    if tiled_array.dtype != np.uint8:
                        log.warning(
                            f"skimage.montage returned dtype {tiled_array.dtype},"
                            " converting back to uint8."
                        )
                        tiled_array = tiled_array.astype(np.uint8)

                    expected_shape = (layout.tile_height, layout.tile_width)
                    if tiled_array.shape != expected_shape:
                        log.error(
                            f"Shape mismatch from skimage.montage:"
                            f" Got {tiled_array.shape}, Expected {expected_shape}."
                            " Falling back to NumPy loop."
                        )
                        tiled_array = None  # Force fallback

                except Exception as montage_e:
                    log.error(
                        f"skimage.util.montage failed: {montage_e}."
                        " Falling back to NumPy loop.",
                        exc_info=debug,
                    )
                    tiled_array = None  # Force fallback

            # --- NumPy Fallback / Default Logic ---
            if tiled_array is None:
                # Log only if skimage was attempted but failed/shape mismatch
                if use_skimage:
                    log.debug(
                        f"T:{time_id} C:{ch_id} - Using NumPy loop for tiling"
                        " (fallback)."
                    )
                else:  # Log if skimage wasn't imported
                    log.debug(f"T:{time_id} C:{ch_id} - Using NumPy loop for tiling.")

                tiled_array = np.zeros(
                    (layout.tile_height, layout.tile_width), dtype=np.uint8
                )
                for i in range(layout.depth):
                    if i >= vol_8bit.shape[0]:
                        log.error(
                            f"NumPy Tiling: Slice index {i} OOB for depth"
                            f" {vol_8bit.shape[0]}. Skipping."
                        )
                        continue
                    paste_col = i % layout.cols
                    paste_row = i // layout.cols
                    y_start = paste_row * layout.height
                    y_end = y_start + layout.height
                    x_start = paste_col * layout.width
                    x_end = x_start + layout.width
                    if y_end > layout.tile_height or x_end > layout.tile_width:
                        log.error(
                            f"NumPy Tiling: Calculated paste coords OOB"
                            f" for slice {i}. Skipping."
                        )
                        continue
                    try:
                        tiled_array[y_start:y_end, x_start:x_end] = vol_8bit[i, :, :]
                    except ValueError as e:
                        log.error(
                            f"NumPy Tiling: Error assigning slice {i}. Error: {e}"
                        )
                        continue
            # --- End Tiling Logic ---

            # 4. Convert final NumPy array to PIL Image
            log.debug(
                f"T:{time_id} C:{ch_id} - Converting tiled NumPy array to PIL Image."
            )
            try:
                tiled_img = Image.fromarray(tiled_array)
            except Exception as img_e:
                log.error(
                    f"Failed to create PIL Image from tiled NumPy array: {img_e}",
                    exc_info=True,
                )
                return None

            # 5. Save the Tiled Image
            out_path = output_folder_obj / out_file
            log.debug(f"Attempting to save tiled WebP image to: {out_path}")
            try:
                output_folder_obj.mkdir(parents=True, exist_ok=True)
            except Exception as mkdir_e:
                log.error(
                    f"Failed to create output directory {output_folder_obj}"
                    f" before saving {out_file}: {mkdir_e}",
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
                log.error(f"Failed save WebP {out_path}: {e}", exc_info=True)
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
        log.info(f"Successfully processed T:{time_id} C:{ch_id}")

    except IndexError as e:
        log.error(
            f"❌ IndexError T:{time_id} C:{ch_id}: {e}. Vol shape:"
            f" {globally_cropped_vol.shape}, Layout depth: {layout.depth}",
            exc_info=True,
        )
        result_data = None
    except Exception as e:
        log.error(f"❌ Unexpected Error T:{time_id} C:{ch_id}: {e}", exc_info=True)
        result_data = None

    return result_data
