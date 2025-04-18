# src/pixelpacker/io_utils.py

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib
import numpy as np
import tifffile
from PIL import Image
# --- Add import for smoothing ---
from scipy.ndimage import gaussian_filter1d
# --- End Add import ---

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
    logging.warning("Matplotlib not found or backend 'Agg' failed. Debug histogram/plot saving disabled.")

# Constants
WEBP_QUALITY = 87
WEBP_METHOD = 6
WEBP_LOSSLESS = False
SMOOTHING_SIGMA_Z = 2.0

# --- Helper to save debug NumPy arrays as images ---
def _save_debug_array_as_image(arr: np.ndarray, filename: Path, limits: Optional[ContrastLimits] = None):
    """Scales a 2D numpy array and saves as PNG."""
    # --- Ruff Fix: E701 at line 40 ---
    if arr.ndim != 2:
        log.warning(f"Cannot save debug array {filename.name}: Not 2D.")
        return
    # --- End Ruff Fix ---
    # --- Ruff Fix: E701 at line 41 ---
    if arr.size == 0:
        log.warning(f"Cannot save empty debug array: {filename.name}")
        return
    # --- End Ruff Fix ---
    try:
        if limits:
            log.debug(f"Scaling debug image {filename.name} using limits: [{limits.p_low:.1f}-{limits.p_high:.1f}]")
            p_low, p_high = limits.p_low, limits.p_high
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
            # --- Ruff Fix: E702 at line 48 ---
            arr_min = np.min(arr)
            arr_max = np.max(arr)
            # --- End Ruff Fix ---
            if arr_max > arr_min:
                arr_norm = (arr.astype(np.float32) - arr_min) / (arr_max - arr_min)
                arr_8bit = (np.clip(arr_norm, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                arr_8bit = np.zeros_like(arr, dtype=np.uint8)

        # --- Ruff Fix: E702 at line 54 ---
        img_pil = Image.fromarray(arr_8bit)
        log.debug(f"Saving debug image: {filename}")
        img_pil.save(str(filename), format="PNG")
        # --- End Ruff Fix ---
    except Exception as e:
        # --- Ruff Fix: E701 at line 57 ---
        log.error(f"Failed to save debug array image {filename}: {e}", exc_info=True)
        # --- End Ruff Fix ---


# --- Modified Z-Cropping Function (Smoothed Projection Based + Debug Output) ---
def find_z_crop_range_projection(
    volume: np.ndarray,
    threshold: int,
    debug: bool = False,
    output_folder: Optional[Path] = None,
    filename_prefix: str = "debug"
) -> Tuple[int, int]:
    """
    Finds Z-crop range based on a SMOOTHED max intensity profile derived
    from YZ and XZ MIPs. Optionally saves debug outputs.
    """
    if volume.ndim != 3 or volume.shape[0] == 0:
        log.warning(f"Cannot calculate projection crop for non-3D/empty volume shape {volume.shape}")
        return (0, 0)

    depth = volume.shape[0]
    # --- Ruff Fix: E701 at line 69 ---
    if depth <= 1:
        return (0, depth - 1 if depth > 0 else 0)
    # --- End Ruff Fix ---

    # --- Ruff Fix: E702 at line 71 ---
    z_start = 0
    z_end = depth - 1 # Default to original range
    # --- End Ruff Fix ---

    try:
        # Calculate Maximum Intensity Projections
        log.debug("Calculating MIPs for Z-cropping...")
        # --- Ruff Fix: E702 at line 75 ---
        mip_yz = np.max(volume, axis=2) # Shape (Z, Y)
        mip_xz = np.max(volume, axis=1) # Shape (Z, X)
        # --- End Ruff Fix ---

        # Calculate max profile along Z
        max_per_z_yz = np.max(mip_yz, axis=1) # Shape (Z,)
        max_per_z_xz = np.max(mip_xz, axis=1) # Shape (Z,)
        max_z_profile = np.maximum(max_per_z_yz, max_per_z_xz) # Shape (Z,)
        log.debug(f"Raw Max Z profile range: [{np.min(max_z_profile):.1f}, {np.max(max_z_profile):.1f}]")

        log.debug(f"Applying Gaussian smoothing (sigma={SMOOTHING_SIGMA_Z}) to Z profile...")
        smoothed_profile = gaussian_filter1d(max_z_profile, sigma=SMOOTHING_SIGMA_Z, mode="reflect")
        log.debug(f"Smoothed Max Z profile range: [{np.min(smoothed_profile):.1f}, {np.max(smoothed_profile):.1f}]")

        # Threshold the SMOOTHED profile
        valid_indices = np.where(smoothed_profile > threshold)[0]

        if valid_indices.size == 0:
            log.warning(f"No Z-slices found with SMOOTHED max projection intensity > {threshold}. Cannot crop.")
            # Keep default z_start=0, z_end=depth-1
        else:
            # --- Ruff Fix: E702 at line 96 ---
            z_start = int(valid_indices.min())
            z_end = int(valid_indices.max())
            # --- End Ruff Fix ---
            # --- Ruff Fix: E701 at line 97 ---
            if z_start > z_end: # Sanity check
                 log.error(f"Z-crop projection calc resulted in start > end ({z_start} > {z_end}). Using original range.")
                 # --- Ruff Fix: E702 at line 97 ---
                 z_start = 0
                 z_end = depth - 1 # Fallback
                 # --- End Ruff Fix ---
            # --- End Ruff Fix ---
            else:
                 log.debug(f"Projection Z-crop range determined (from smoothed): Keep slices {z_start} to {z_end}.")

        # --- Save Debug Images/Plots if requested ---
        # --- Ruff Fix: E701 at line 101 ---
        if debug and output_folder is not None and MATPLOTLIB_AVAILABLE and plt is not None:
            assert plt is not None
            log.debug(f"Saving Z-crop debug info for prefix: {filename_prefix}")
            # --- Ruff Fix: E702 at line 103 ---
            output_folder.mkdir(parents=True, exist_ok=True) # Ensure folder exists

            # Calculate simple limits for scaling MIPs for visualization
            mip_limits = calculate_limits_only(volume, stretch_mode="max")
            # --- End Ruff Fix ---

            # Save MIPs
            # --- Ruff Fix: E702 at line 107 ---
            mip_yz_path = output_folder / f"{filename_prefix}_debug_mip_yz.png"
            _save_debug_array_as_image(mip_yz, mip_yz_path, mip_limits)
            mip_xz_path = output_folder / f"{filename_prefix}_debug_mip_xz.png"
            _save_debug_array_as_image(mip_xz, mip_xz_path, mip_limits)
            # --- End Ruff Fix ---

            # Save Z-Profile Plot
            z_indices = np.arange(depth)
            profile_plot_path = output_folder / f"{filename_prefix}_debug_z_profile.png"
            fig_prof = None
            try:
                fig_prof, ax_prof = plt.subplots(figsize=(10, 5))
                ax_prof.plot(z_indices, max_z_profile, label="Raw Max Profile (from MIPs)", color='lightblue', alpha=0.7)
                ax_prof.plot(z_indices, smoothed_profile, label=f"Smoothed Profile (sigma={SMOOTHING_SIGMA_Z})", color='blue', linewidth=1.5)
                ax_prof.axhline(threshold, color='red', linestyle='--', label=f"Threshold ({threshold})")
                ax_prof.axvline(z_start, color='green', linestyle=':', label=f"Z Start ({z_start})")
                ax_prof.axvline(z_end, color='lime', linestyle=':', label=f"Z End ({z_end})")
                ax_prof.set_xlabel("Z Slice Index")
                ax_prof.set_ylabel("Max Intensity in XY Plane")
                ax_prof.set_title(f"Z-Crop Profile Analysis - {filename_prefix}")
                ax_prof.legend()
                # --- Ruff Fix: E702 at line 128 ---
                ax_prof.grid(True, linestyle=':', alpha=0.6)
                # Optionally set y-axis to log scale if needed: ax_prof.set_yscale('log')
                log.debug(f"Saving Z-profile plot: {profile_plot_path}")
                # --- End Ruff Fix ---
                fig_prof.savefig(str(profile_plot_path), dpi=100)
            except Exception as plot_e:
                log.error(f"Failed to generate or save Z-profile plot: {plot_e}", exc_info=True)
            finally:
                # --- Ruff Fix: E701 at line 135 ---
                if fig_prof is not None:
                    plt.close(fig_prof)
                # --- End Ruff Fix ---
        # --- End Debug Saving ---

        return z_start, z_end

    except MemoryError:
         # --- Ruff Fix: E701 at line 140 ---
         log.error("MemoryError calculating MIPs or Z profile for cropping. Skipping crop.")
         return (0, depth - 1)
         # --- End Ruff Fix ---
    except Exception as e:
        # --- Ruff Fix: E701 at line 143 ---
        log.error(f"Unexpected error during projection Z-crop calculation: {e}", exc_info=True)
        return (0, depth - 1) # Fallback on any error
        # --- End Ruff Fix ---


# --- extract_original_volume ---
def extract_original_volume(tif_path: Path) -> Optional[np.ndarray]:
    """Extracts and reshapes the volume from a TIFF file to 3D (Z, Y, X)."""
    log.debug(f"Extracting original volume from: {tif_path}")
    # --- Ruff Fix: E701 at line 151 ---
    if not tif_path.is_file():
        log.error(f"TIFF file not found: {tif_path}")
        return None
    # --- End Ruff Fix ---
    try:
        with tifffile.TiffFile(str(tif_path)) as tif:
            vol: np.ndarray = tif.asarray()
        log.debug(f"Original TIFF shape: {vol.shape}")
        # --- Ruff Fix: E702 at line 158 ---
        vol = np.squeeze(vol)
        log.debug(f"Squeezed TIFF shape: {vol.shape}")
        # --- End Ruff Fix ---
        if vol.ndim == 5:
            # --- Ruff Fix: E701 at line 160 ---
            if vol.shape[0] == 1 and vol.shape[2] == 1:
                _, z, _, y, x = vol.shape
                vol = vol.reshape((z, y, x))
                log.debug(f"Reshaped 5D -> 3D: {vol.shape}")
            # --- End Ruff Fix ---
            else:
                raise ValueError(f"Unsupported 5D shape: {vol.shape}.")
        elif vol.ndim == 4:
            if 1 in vol.shape:
                # --- Ruff Fix: E702 ---
                original_4d_shape = vol.shape
                vol = vol.reshape([s for s in vol.shape if s != 1])
                log.debug(f"Reshaped 4D {original_4d_shape} -> {vol.ndim}D: {vol.shape}")
                assert vol.ndim == 3
                # --- End Ruff Fix ---
            else:
                raise ValueError(f"Unsupported 4D shape: {vol.shape}.")
        elif vol.ndim == 3:
            log.debug("Shape is 3D.")
        elif vol.ndim == 2:
            # --- Ruff Fix: E701 ---
            log.debug("Shape is 2D, adding Z dim.")
            vol = vol[np.newaxis, :, :]
            # --- End Ruff Fix ---
        else:
            raise ValueError(f"Unsupported shape: {vol.shape} (ndim={vol.ndim})")
        # --- Ruff Fix: E701 ---
        if vol.ndim != 3:
            raise ValueError(f"Not 3D after reshape: {vol.shape}")
        # --- End Ruff Fix ---
        log.debug(f"Final extracted original volume shape: {vol.shape}")
        return vol
    except Exception as e:
        # --- Ruff Fix: E701 ---
        log.error(f"Error extracting {tif_path.name}: {e}", exc_info=True)
        return None
        # --- End Ruff Fix ---


# --- save_preview_slice, save_histogram_debug ---
def save_preview_slice(vol_8bit: np.ndarray, path: Path):
    # --- Ruff Fix: E701 at line 197 ---
    if vol_8bit.ndim != 3 or vol_8bit.shape[0] == 0:
        log.warning(f"Cannot save preview: shape {vol_8bit.shape}")
        return
    # --- End Ruff Fix ---
    # --- Ruff Fix: E701 at line 198 ---
    if vol_8bit.dtype != np.uint8:
        log.warning(f"Preview input not uint8: {vol_8bit.dtype}")
    # --- End Ruff Fix ---
    # --- Ruff Fix: E702 at line 200 ---
    mid_z = vol_8bit.shape[0] // 2
    slice_2d = vol_8bit[mid_z]
    # --- End Ruff Fix ---
    try:
        # --- Ruff Fix: E702 at line 201 ---
        img_pil = Image.fromarray(slice_2d)
        log.debug(f"Saving preview: {path}")
        img_pil.save(str(path), format="PNG")
        # --- End Ruff Fix ---
    except Exception as e:
        log.error(f"Failed save preview {path}: {e}", exc_info=True)

def save_histogram_debug(img: np.ndarray, limits: ContrastLimits, out_path: Path, stretch_mode: str):
    # --- Ruff Fix: E701 at line 205 ---
    if not MATPLOTLIB_AVAILABLE or plt is None:
        log.warning("Matplotlib unavailable, skip histogram.")
        return
    # --- End Ruff Fix ---
    assert plt is not None
    fig = None
    try:
        # --- Ruff Fix: E702 at line 209 ---
        img = np.asarray(img)
        nonzero_pixels = img[img > 0].flatten()
        # --- End Ruff Fix ---
        # --- Ruff Fix: E701 at line 212 ---
        if nonzero_pixels.size == 0:
            log.warning(f"No non-zero pixels for hist {out_path.name}. Skip.")
            return
        # --- End Ruff Fix ---
        vmin = limits.actual_min if limits.actual_min is not None else 0.0
        vmax = limits.actual_max if limits.actual_max is not None else 0.0
        # --- Ruff Fix: E701 at line 217 ---
        if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmax <= vmin:
             log.warning(f"Invalid range [{vmin}, {vmax}] for hist {out_path.name}. Skip.")
             return
        # --- End Ruff Fix ---
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(nonzero_pixels, bins=256, range=(vmin, vmax), color="gray", alpha=0.7, log=True, label="Log Histogram (Non-zero Pixels)")
        # --- Ruff Fix: E702 (Dictionary reformatting) ---
        plot_markers = {
            f"Stretch Low ({limits.p_low:.1f})": (limits.p_low, "red", "-"),
            f"Stretch High ({limits.p_high:.1f})": (limits.p_high, "red", "-"),
            (f"1% ({limits.p1:.1f})" if limits.p1 is not None else None):
                (limits.p1, "blue", "--"),
            (f"ImageJ Min ({limits.p035:.1f})" if limits.p035 is not None else None):
                (limits.p035, "orange", "--"),
            (f"Smart Early ({limits.smart_early:.1f})" if limits.smart_early is not None else None):
                (limits.smart_early, "purple", "--"),
            (f"Smart Late ({limits.smart_late:.1f})" if limits.smart_late is not None else None):
                (limits.smart_late, "green", "--"),
            (f"ImageJ Max ({limits.p9965:.1f})" if limits.p9965 is not None else None):
                (limits.p9965, "cyan", "--"),
            (f"Actual Max ({limits.actual_max:.1f})" if limits.actual_max is not None else None):
                (limits.actual_max, "magenta", ":"),
            (f"Actual Min ({limits.actual_min:.1f})" if limits.actual_min is not None else None):
                (limits.actual_min, "yellow", ":"),
        }
        # --- End Ruff Fix ---
        # --- Ruff Fix: E702 ---
        added_labels = set()
        y_min, y_max = ax.get_ylim()
        # --- End Ruff Fix ---
        for label, (value, color, style) in plot_markers.items():
            # --- Ruff Fix: E701 ---
            if label is not None and value is not None and np.isfinite(value) and value >= vmin and value <= vmax:
                # --- Ruff Fix: E701 ---
                if label not in added_labels:
                     ax.vlines(value, y_min, y_max, color=color, linestyle=style, label=label)
                     added_labels.add(label)
                 # --- End Ruff Fix ---
            # --- End Ruff Fix ---
        # --- Ruff Fix: E702 ---
        ax.set_title(f"Intensity Histogram Debug – Stretch Mode: '{stretch_mode}'\n(Bounds Used: [{limits.p_low:.1f} - {limits.p_high:.1f}])")
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Log Frequency (Count)")
        ax.set_ylim(y_min, y_max)
        ax.legend(fontsize="small")
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)
        ax.set_xlim(vmin, vmax)
        log.debug(f"Saving histogram: {out_path}")
        fig.savefig(str(out_path), dpi=100)
        # --- End Ruff Fix ---
    except Exception as e:
        log.error(f"Failed histogram {out_path}: {e}", exc_info=True)
    finally:
        # --- Ruff Fix: E701 ---
        if fig is not None:
            plt.close(fig)
        # --- End Ruff Fix ---


# --- process_channel ---
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
    output_folder_obj = Path(output_folder)
    log.info(f"Processing T:{time_id} C:{ch_id} - Received Globally Cropped Shape: {globally_cropped_vol.shape}")
    result_data: Optional[Dict[str, Any]] = None
    try:
        vol_8bit, _ = apply_autocontrast_8bit(
            globally_cropped_vol,
            stretch_mode="max",
            global_limits_tuple=(limits.p_low, limits.p_high)
        )
        log.debug(f"T:{time_id} C:{ch_id} - Applied Stretched Range: ({limits.p_low:.2f}, {limits.p_high:.2f})")

        # --- Ruff Fix: E701 ---
        if debug and not dry_run:
            # --- Ruff Fix: E702 ---
            hist_filename = f"debug_hist_T{time_id}_C{ch_id}.png"
            hist_path = output_folder_obj / hist_filename
            save_histogram_debug(globally_cropped_vol, limits, hist_path, stretch_mode)
            # --- End Ruff Fix ---
        # --- End Ruff Fix ---

        out_file = f"volume_{time_id}_c{ch_id}.webp"
        # --- Ruff Fix: E701 ---
        if not dry_run:
            tiled_img = Image.new("L", (layout.tile_width, layout.tile_height), color=0)
            log.debug(f"T:{time_id} C:{ch_id} - Creating {layout.tile_width}x{layout.tile_height} tile image...")

            for i in range(layout.depth):
                # --- Ruff Fix: E701 ---
                if i >= vol_8bit.shape[0]:
                     log.error(f"Tiling error: Index {i} OOB for depth {vol_8bit.shape[0]}.")
                     continue
                 # --- End Ruff Fix ---
                slice_img = Image.fromarray(vol_8bit[i])
                # --- Ruff Fix: E702 ---
                paste_col = i % layout.cols
                paste_row = i // layout.cols
                # --- End Ruff Fix ---
                # --- Ruff Fix: E702 ---
                paste_x = paste_col * layout.width
                paste_y = paste_row * layout.height
                # --- End Ruff Fix ---
                # --- Ruff Fix: E701 ---
                if paste_x + layout.width <= layout.tile_width and paste_y + layout.height <= layout.tile_height:
                     tiled_img.paste(slice_img, (paste_x, paste_y))
                # --- End Ruff Fix ---
                else:
                     log.error(f"Paste coords OOB. Skipping slice {i}.")

            # --- Ruff Fix: E702 ---
            out_path = output_folder_obj / out_file
            log.debug(f"Saving tiled WebP image to: {out_path}")
            # --- End Ruff Fix ---
            try:
                # --- Ruff Fix: E702 ---
                tiled_img.save(
                    str(out_path), format="WEBP", quality=WEBP_QUALITY,
                    method=WEBP_METHOD, lossless=WEBP_LOSSLESS
                )
                # --- End Ruff Fix ---
            except Exception as e:
                log.error(f"Failed save WebP {out_path}: {e}", exc_info=True)
                return None
        # --- End Ruff Fix ---

        # --- Ruff Fix: E701 ---
        if debug and not dry_run and ch_id == 0:
            # --- Ruff Fix: E702 ---
            preview_filename = f"preview_T{time_id}_C{ch_id}.png"
            preview_path = output_folder_obj / preview_filename
            save_preview_slice(vol_8bit, preview_path)
            # --- End Ruff Fix ---
        # --- End Ruff Fix ---

        result_data = {
            "time_id": time_id, "channel": ch_id, "filename": out_file,
            "intensity_range": {"p_low": limits.p_low, "p_high": limits.p_high},
        }
        log.info(f"Successfully processed T:{time_id} C:{ch_id}")

    except IndexError as e:
        # --- Ruff Fix: E702 ---
        log.error(f"❌ IndexError T:{time_id} C:{ch_id}: {e}. Vol shape: {globally_cropped_vol.shape}, Layout depth: {layout.depth}", exc_info=True)
        result_data = None
        # --- End Ruff Fix ---
    except Exception as e:
        # --- Ruff Fix: E702 ---
        log.error(f"❌ Unexpected Error T:{time_id} C:{ch_id}: {e}", exc_info=True)
        result_data = None
        # --- End Ruff Fix ---

    return result_data
