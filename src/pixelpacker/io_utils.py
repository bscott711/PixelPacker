# tiff_preprocessor/io_utils.py

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib
import numpy as np
import tifffile
from PIL import Image

# Import ContrastLimits from stretch and VolumeLayout from RENAMED data_models
from .stretch import apply_autocontrast_8bit, ContrastLimits
from .data_models import VolumeLayout # <-- IMPORT UPDATED

# (Matplotlib setup unchanged)
log = logging.getLogger(__name__)
try:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not found or backend 'Agg' failed. Debug histogram saving disabled.")

log = logging.getLogger(__name__)

# (Constants unchanged)
WEBP_QUALITY = 87
WEBP_METHOD = 6
WEBP_LOSSLESS = False

# (extract_volume, save_preview_slice, save_histogram_debug unchanged from previous step)
# ... functions extract_volume, save_preview_slice, save_histogram_debug ...
def extract_volume(tif_path: Path) -> np.ndarray:
    # (Implementation unchanged)
    log.debug(f"Extracting volume from: {tif_path}")
    if not tif_path.is_file():
        raise FileNotFoundError(f"TIFF file not found: {tif_path}")
    with tifffile.TiffFile(str(tif_path)) as tif:
        vol: np.ndarray = tif.asarray()
    log.debug(f"Original TIFF shape: {vol.shape}")
    vol = np.squeeze(vol)
    log.debug(f"Squeezed TIFF shape: {vol.shape}")
    if vol.ndim == 5: # Example logic
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
         raise ValueError(f"Final volume shape is not 3D after processing: {vol.shape}")
    log.debug(f"Final extracted volume shape: {vol.shape}")
    return vol

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
        for label, (value, color, style) in plot_markers.items():
            if label is not None and value is not None and np.isfinite(value) and value >= vmin and value <= vmax:
                if label not in added_labels:
                     ax.axvline(value, color=color, linestyle=style, label=label)
                     added_labels.add(label)
        ax.set_title(f"Intensity Histogram Debug – Stretch Mode: '{stretch_mode}'\n(Bounds Used: [{limits.p_low:.1f} - {limits.p_high:.1f}])")
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Log Frequency (Count)")
        ax.legend(fontsize="small")
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        ax.set_xlim(vmin, vmax)
        log.debug(f"Saving histogram debug plot to: {out_path}")
        fig.savefig(str(out_path), dpi=100)
    except Exception as e:
        log.error(f"Failed to generate or save histogram plot to {out_path}: {e}", exc_info=True)
    finally:
        if fig is not None:
            plt.close(fig)


# process_channel signature already updated previously, uses VolumeLayout from data_models now
def process_channel(
    time_id: str,
    ch_id: int,
    tiff_path: str,
    layout: VolumeLayout, # Accepts VolumeLayout object
    stretch_mode: str,
    dry_run: bool = False,
    debug: bool = False,
    output_folder: str = ".",
    global_limits_tuple: Optional[Tuple[float, float]] = None,
) -> Optional[Dict[str, Any]]:
    """Processes a single channel: extracts volume, stretches, tiles, and saves."""
    # (Implementation unchanged from previous step, already uses layout object)
    tiff_path_obj = Path(tiff_path)
    output_folder_obj = Path(output_folder)
    log.info(f"Processing T:{time_id} C:{ch_id} - File: {tiff_path_obj.name}")
    result_data: Optional[Dict[str, Any]] = None
    try:
        vol = extract_volume(tiff_path_obj)
        expected_shape = (layout.depth, layout.height, layout.width)
        if vol.shape != expected_shape:
            log.warning(
                f"Shape mismatch for T:{time_id} C:{ch_id}. "
                f"Expected {expected_shape} from layout, got {vol.shape}. Skipping channel."
            )
            return None
        vol_8bit, limits = apply_autocontrast_8bit(vol, stretch_mode, global_limits_tuple)
        log.debug(f"T:{time_id} C:{ch_id} - Stretched Range Used: ({limits.p_low:.2f}, {limits.p_high:.2f})")
        if debug and not dry_run:
            hist_filename = f"debug_hist_T{time_id}_C{ch_id}.png"
            hist_path = output_folder_obj / hist_filename
            save_histogram_debug(vol, limits, hist_path, stretch_mode)
        out_file = f"volume_{time_id}_c{ch_id}.webp"
        if not dry_run:
            tiled_img = Image.new("L", (layout.tile_width, layout.tile_height), color=0)
            log.debug(f"T:{time_id} C:{ch_id} - Creating {layout.tile_width}x{layout.tile_height} tile image...")
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
    # ... (exception handling unchanged) ...
    except FileNotFoundError as e:
        log.error(f"❌ Error processing T:{time_id} C:{ch_id} - Input file not found: {e}")
        result_data = None
    except ValueError as e:
        log.error(f"❌ Error processing T:{time_id} C:{ch_id} - Data/Shape error: {e}")
        result_data = None
    except Exception as e:
        log.error(f"❌ Unexpected Error processing T:{time_id} C:{ch_id}: {e}", exc_info=True)
        result_data = None
    return result_data