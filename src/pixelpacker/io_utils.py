# src/pixelpacker/io_utils.py

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib
import numpy as np
import tifffile
from PIL import Image
from scipy.ndimage import gaussian_filter1d

# Import ContrastLimits for scaling debug MIPs
from .stretch import ContrastLimits, calculate_limits_only, apply_autocontrast_8bit
from .data_models import VolumeLayout

log = logging.getLogger(__name__)
try:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False
    logging.warning(
        "Matplotlib not found or backend 'Agg' failed. Debug histogram/plot saving disabled."
    )

# Constants
WEBP_QUALITY = 87
WEBP_METHOD = 6
WEBP_LOSSLESS = False
SMOOTHING_SIGMA_Z = 2.0  # Sigma for Gaussian smoothing of Z profile.
SLOPE_WINDOW_Z = 2  # Number of consecutive slices to check slope sign
SLOPE_THRESH_Z_POS = (
    20  # Threshold for detecting significant positive slope (tune if needed)
)
MIN_SEARCH_OFFSET_Z = 3  # Start search slightly away from the edges


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
        if limits:
            log.debug(
                f"Scaling debug image {filename.name} using limits: [{limits.p_low:.1f}-{limits.p_high:.1f}]"
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
                f"Threshold method: No Z-slices found with max projection intensity > {threshold}. Cannot crop."
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
            f"Unexpected error during threshold Z-crop calculation: {e}", exc_info=True
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
            f"Raw Max Z profile range: [{np.min(max_z_profile):.1f}, {np.max(max_z_profile):.1f}]"
        )

        log.debug(
            f"Applying Gaussian smoothing (sigma={SMOOTHING_SIGMA_Z}) to Z profile..."
        )
        smoothed_profile = gaussian_filter1d(
            max_z_profile, sigma=SMOOTHING_SIGMA_Z, mode="reflect"
        )
        log.debug(
            f"Smoothed Max Z profile range: [{np.min(smoothed_profile):.1f}, {np.max(smoothed_profile):.1f}]"
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
                        f"Found potential Z start at index {i} (slope > {SLOPE_THRESH_Z_POS})"
                    )
                    break

        # Find End Slice
        found_end = False
        end_search_start = depth - MIN_SEARCH_OFFSET_Z - 1
        end_search_end = SLOPE_WINDOW_Z - 1
        if end_search_start > end_search_end:
            for i in range(end_search_start, end_search_end, -1):
                if i - SLOPE_WINDOW_Z + 1 < 0:
                    continue
                window = slope[i - SLOPE_WINDOW_Z + 1 : i + 1]
                if np.all(window < -SLOPE_THRESH_Z_POS):
                    z_end_found = i + 1
                    found_end = True
                    log.debug(
                        f"Found potential Z end at index {i + 1} (slope < {-SLOPE_THRESH_Z_POS})"
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
                f"Z-crop slope analysis resulted in start >= end ({z_start_found} >= {z_end_found}). Using original range."
            )
            z_start_found = 0
            z_end_found = depth - 1

        log.info(
            f"Slope-based Z-crop range determined: Keep slices {z_start_found} to {z_end_found}."
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
        # --- FIX: Check if MIPs exist before using ---
        if mip_yz is not None and mip_xz is not None:
            mip_limits = calculate_limits_only(volume, stretch_mode="max")
            mip_yz_path = output_folder / f"{filename_prefix}_debug_mip_yz.png"
            _save_debug_array_as_image(mip_yz, mip_yz_path, mip_limits)
            mip_xz_path = output_folder / f"{filename_prefix}_debug_mip_xz.png"
            _save_debug_array_as_image(mip_xz, mip_xz_path, mip_limits)
        # --- End FIX ---
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
                z_indices, slope, color=color2, linestyle="--", alpha=0.8, label="Slope"
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
                z_end_found, color="lime", linestyle=":", label=f"Z End ({z_end_found})"
            )

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(
                lines1 + lines2, labels1 + labels2, loc="upper right", fontsize="small"
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
    threshold: int,  # Keep threshold arg for 'threshold' method
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
        # Threshold function doesn't currently save debug plots, but could be added
        return _find_z_crop_range_threshold(volume, threshold)
    else:
        log.warning(f"Unknown z_crop_method '{method}'. Defaulting to 'slope'.")
        return _find_z_crop_range_slope(volume, debug, output_folder, filename_prefix)


# --- extract_original_volume (Unchanged) ---
def extract_original_volume(tif_path: Path) -> Optional[np.ndarray]:
    """Extracts and reshapes the volume from a TIFF file to 3D (Z, Y, X)."""
    # (Implementation unchanged)
    log.debug(f"Extracting original volume from: {tif_path}")
    if not tif_path.is_file():
        log.error(f"TIFF file not found: {tif_path}")
        return None
    try:
        with tifffile.TiffFile(str(tif_path)) as tif:
            vol: np.ndarray = tif.asarray()
        log.debug(f"Original TIFF shape: {vol.shape}")
        vol = np.squeeze(vol)
        log.debug(f"Squeezed TIFF shape: {vol.shape}")
        if vol.ndim == 5:
            if vol.shape[0] == 1 and vol.shape[2] == 1:
                _, z, _, y, x = vol.shape
                vol = vol.reshape((z, y, x))
                log.debug(f"Reshaped 5D -> 3D: {vol.shape}")
            else:
                raise ValueError(f"Unsupported 5D shape: {vol.shape}.")
        elif vol.ndim == 4:
            if 1 in vol.shape:
                original_4d_shape = vol.shape
                vol = vol.reshape([s for s in vol.shape if s != 1])
                log.debug(
                    f"Reshaped 4D {original_4d_shape} -> {vol.ndim}D: {vol.shape}"
                )
                assert vol.ndim == 3
            else:
                raise ValueError(f"Unsupported 4D shape: {vol.shape}.")
        elif vol.ndim == 3:
            log.debug("Shape is 3D.")
        elif vol.ndim == 2:
            log.debug("Shape is 2D, adding Z dim.")
            vol = vol[np.newaxis, :, :]
        else:
            raise ValueError(f"Unsupported shape: {vol.shape} (ndim={vol.ndim})")
        if vol.ndim != 3:
            raise ValueError(f"Not 3D after reshape: {vol.shape}")
        log.debug(f"Final extracted original volume shape: {vol.shape}")
        return vol
    except Exception as e:
        log.error(f"Error extracting {tif_path.name}: {e}", exc_info=True)
        return None


# --- save_preview_slice, save_histogram_debug (Unchanged) ---
def save_preview_slice(vol_8bit: np.ndarray, path: Path):
    if vol_8bit.ndim != 3 or vol_8bit.shape[0] == 0:
        log.warning(f"Cannot save preview: shape {vol_8bit.shape}")
        return
    if vol_8bit.dtype != np.uint8:
        log.warning(f"Preview input not uint8: {vol_8bit.dtype}")
    mid_z = vol_8bit.shape[0] // 2
    slice_2d = vol_8bit[mid_z]
    try:
        img_pil = Image.fromarray(slice_2d)
        log.debug(f"Saving preview: {path}")
        img_pil.save(str(path), format="PNG")
    except Exception as e:
        log.error(f"Failed save preview {path}: {e}", exc_info=True)


def save_histogram_debug(
    img: np.ndarray, limits: ContrastLimits, out_path: Path, stretch_mode: str
):
    if not MATPLOTLIB_AVAILABLE or plt is None:
        log.warning("Matplotlib unavailable, skip histogram.")
        return
    assert plt is not None
    fig = None
    try:
        img = np.asarray(img)
        nonzero_pixels = img[img > 0].flatten()
        if nonzero_pixels.size == 0:
            log.warning(f"No non-zero pixels for hist {out_path.name}. Skip.")
            return
        vmin = limits.actual_min if limits.actual_min is not None else 0.0
        vmax = limits.actual_max if limits.actual_max is not None else 0.0
        if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmax <= vmin:
            log.warning(
                f"Invalid range [{vmin}, {vmax}] for hist {out_path.name}. Skip."
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
            f"Intensity Histogram Debug – Stretch Mode: '{stretch_mode}'\n(Bounds Used: [{limits.p_low:.1f} - {limits.p_high:.1f}])"
        )
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Log Frequency (Count)")
        ax.set_ylim(y_min, y_max)
        ax.legend(fontsize="small")
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        ax.set_xlim(vmin, vmax)
        log.debug(f"Saving histogram: {out_path}")
        fig.savefig(str(out_path), dpi=100)
    except Exception as e:
        # --- Ruff Fix: F541 ---
        # Removed f-string as no placeholders were used
        log.error(f"Failed histogram {out_path}: {e}", exc_info=True)

    finally:
        if fig is not None:
            plt.close(fig)


# --- process_channel (Unchanged) ---
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
    """Processes a single channel: tiles the already cropped/stretched volume and saves."""
    # (Implementation unchanged)
    output_folder_obj = Path(output_folder)
    log.info(
        f"Processing T:{time_id} C:{ch_id} - Received Globally Cropped Shape: {globally_cropped_vol.shape}"
    )
    result_data: Optional[Dict[str, Any]] = None
    try:
        vol_8bit, _ = apply_autocontrast_8bit(
            globally_cropped_vol,
            stretch_mode="max",
            global_limits_tuple=(limits.p_low, limits.p_high),
        )
        log.debug(
            f"T:{time_id} C:{ch_id} - Applied Stretched Range: ({limits.p_low:.2f}, {limits.p_high:.2f})"
        )
        if debug and not dry_run:
            hist_filename = f"debug_hist_T{time_id}_C{ch_id}.png"
            hist_path = output_folder_obj / hist_filename
            save_histogram_debug(globally_cropped_vol, limits, hist_path, stretch_mode)

        out_file = f"volume_{time_id}_c{ch_id}.webp"
        if not dry_run:
            tiled_img = Image.new("L", (layout.tile_width, layout.tile_height), color=0)
            log.debug(
                f"T:{time_id} C:{ch_id} - Creating {layout.tile_width}x{layout.tile_height} tile image..."
            )
            for i in range(layout.depth):
                if i >= vol_8bit.shape[0]:
                    log.error(
                        f"Tiling error: Index {i} OOB for depth {vol_8bit.shape[0]}."
                    )
                    continue
                slice_img = Image.fromarray(vol_8bit[i])

                paste_col = i % layout.cols
                paste_row = i // layout.cols

                paste_x = paste_col * layout.width
                paste_y = paste_row * layout.height

                if (
                    paste_x + layout.width <= layout.tile_width
                    and paste_y + layout.height <= layout.tile_height
                ):
                    tiled_img.paste(slice_img, (paste_x, paste_y))
                else:
                    log.error(f"Paste coords OOB. Skipping slice {i}.")

            out_path = output_folder_obj / out_file
            log.debug(f"Saving tiled WebP image to: {out_path}")

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
        if debug and not dry_run and ch_id == 0:
            preview_filename = f"preview_T{time_id}_C{ch_id}.png"
            preview_path = output_folder_obj / preview_filename
            save_preview_slice(vol_8bit, preview_path)

        result_data = {
            "time_id": time_id,
            "channel": ch_id,
            "filename": out_file,
            "intensity_range": {"p_low": limits.p_low, "p_high": limits.p_high},
        }
        log.info(f"Successfully processed T:{time_id} C:{ch_id}")
    except IndexError as e:
        log.error(
            f"❌ IndexError T:{time_id} C:{ch_id}: {e}. Vol shape: {globally_cropped_vol.shape}, Layout depth: {layout.depth}",
            exc_info=True,
        )
        result_data = None

    except Exception as e:
        log.error(f"❌ Unexpected Error T:{time_id} C:{ch_id}: {e}", exc_info=True)
        result_data = None

    return result_data
