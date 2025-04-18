# src/pixelpacker/stretch.py

import logging
from typing import Optional, Tuple, Callable, Dict
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

# --- Constants ---
HIST_BINS = 512
LOG_HIST_EPSILON = 1e-6
SMOOTHING_SIGMA = 3.0
SLOPE_THRESHOLD = -0.1
SLOPE_WINDOW_SIZE = 5
MIN_SEARCH_OFFSET = 10

FALLBACK_EARLY_PERCENTILE = 99.0
FALLBACK_LATE_PERCENTILE = 99.9

PERCENTILE_1 = 1.0
PERCENTILE_LOW_IMAGEJ = 0.35
PERCENTILE_HIGH_IMAGEJ = 99.65

log = logging.getLogger(__name__)


# --- Data Structure for Detailed Limits ---
@dataclass
class ContrastLimits:
    """Holds various calculated intensity limits for stretching and debugging."""
    p_low: float = 0.0  # The actual lower bound used for stretching
    p_high: float = 0.0 # The actual upper bound used for stretching
    # --- Values primarily for debug histogram ---
    p1: Optional[float] = None  # 1st percentile
    p035: Optional[float] = None  # 0.35 percentile (ImageJ Low)
    p9965: Optional[float] = None  # 99.65 percentile (ImageJ High)
    smart_early: Optional[float] = None  # Calculated dynamic early cutoff
    smart_late: Optional[float] = None  # Calculated dynamic late cutoff
    actual_min: Optional[float] = None  # Actual minimum of non-zero data
    actual_max: Optional[float] = None  # Actual maximum of non-zero data


# --- Dynamic Cutoff Calculation ---
def compute_dynamic_cutoffs(pixels: np.ndarray) -> Tuple[float, float]:
    """
    Computes dynamic intensity cutoffs based on the slope of the log-histogram.
    """
    if pixels.size == 0:
        log.warning("compute_dynamic_cutoffs received empty pixel array.")
        return 0.0, 0.0

    pixels = pixels[np.isfinite(pixels)]
    if pixels.size == 0:
         log.warning("compute_dynamic_cutoffs received array with only non-finite values.")
         return 0.0, 0.0

    pixels_min = float(pixels.min())
    pixels_max = float(pixels.max())

    if pixels_min == pixels_max:
        return pixels_min, pixels_max

    try:
        hist, bin_edges = np.histogram(
            pixels, bins=HIST_BINS, range=(pixels_min, pixels_max)
        )
        if np.sum(hist) == 0:
             log.warning("Histogram is empty after filtering. Falling back to min/max.")
             return pixels_min, pixels_max

        log_hist = np.log10(hist.astype(np.float32) + LOG_HIST_EPSILON)
        slope = np.gradient(log_hist)
        smoothed_slope = gaussian_filter1d(slope, sigma=SMOOTHING_SIGMA, mode="reflect")

        early_cutoff_candidate: Optional[float] = None
        late_cutoff_candidate: Optional[float] = None
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        search_end_early = len(smoothed_slope) - SLOPE_WINDOW_SIZE - MIN_SEARCH_OFFSET
        # Ensure range is valid before iterating
        if MIN_SEARCH_OFFSET < search_end_early:
            for i in range(MIN_SEARCH_OFFSET, search_end_early):
                window = smoothed_slope[i : i + SLOPE_WINDOW_SIZE]
                if np.all(window < SLOPE_THRESHOLD):
                    early_cutoff_candidate = float(bin_centers[i])
                    break

        search_start_late = len(smoothed_slope) - MIN_SEARCH_OFFSET - 1
        search_end_late = MIN_SEARCH_OFFSET + SLOPE_WINDOW_SIZE - 1
        # Ensure range is valid before iterating
        if search_start_late > search_end_late:
            for i in range(search_start_late, search_end_late, -1):
                if i - SLOPE_WINDOW_SIZE + 1 < 0:
                    continue
                window = smoothed_slope[i - SLOPE_WINDOW_SIZE + 1 : i + 1]
                if np.all(window < SLOPE_THRESHOLD):
                    late_cutoff_candidate = float(bin_centers[i])
                    break

        early_cutoff = early_cutoff_candidate if early_cutoff_candidate is not None else float(np.percentile(pixels, FALLBACK_EARLY_PERCENTILE))
        late_cutoff = late_cutoff_candidate if late_cutoff_candidate is not None else float(np.percentile(pixels, FALLBACK_LATE_PERCENTILE))

        if early_cutoff >= late_cutoff:
            log.warning(f"Dynamic cutoffs crossed or collapsed (early={early_cutoff:.2f}, late={late_cutoff:.2f}). Falling back to min/max.")
            early_cutoff, late_cutoff = pixels_min, pixels_max

        return float(early_cutoff), float(late_cutoff)

    except Exception as e:
        log.error(
            f"Error computing dynamic cutoffs: {e}. Falling back to min/max.",
            exc_info=True,
        )
        return float(pixels_min), float(pixels_max)


# --- Limit Calculation Helper Functions ---

LimitCalculator = Callable[[np.ndarray], ContrastLimits]


def _get_base_stats(data: np.ndarray) -> ContrastLimits:
    """Calculates common stats needed for most modes."""
    limits = ContrastLimits()
    if data.size == 0:
        # Set defaults for all fields if input is empty
        limits.actual_min = 0.0
        limits.actual_max = 0.0
        limits.p1 = 0.0
        limits.p035 = 0.0
        limits.p9965 = 0.0
        limits.smart_early = 0.0
        limits.smart_late = 0.0
    else:
        # Calculate stats only if data is present
        limits.actual_min = float(data.min())
        limits.actual_max = float(data.max())
        limits.p1 = float(np.percentile(data, PERCENTILE_1))
        limits.p035, limits.p9965 = map(
            float, np.percentile(data, (PERCENTILE_LOW_IMAGEJ, PERCENTILE_HIGH_IMAGEJ))
        )
        # compute_dynamic_cutoffs handles empty data internally now
        limits.smart_early, limits.smart_late = compute_dynamic_cutoffs(data)

    return limits


def _get_imagej_limits(data: np.ndarray) -> ContrastLimits:
    """Calculates contrast limits using ImageJ's default percentile method."""
    limits = _get_base_stats(data)
    # *** FIX START: Provide default 0.0 if optional value is None ***
    limits.p_low = limits.p035 if limits.p035 is not None else 0.0
    limits.p_high = limits.p9965 if limits.p9965 is not None else 0.0
    # *** FIX END ***
    return limits


def _get_max_limits(data: np.ndarray) -> ContrastLimits:
    """Calculates contrast limits using the full min/max range of the data."""
    limits = _get_base_stats(data)
    # *** FIX START: Provide default 0.0 if optional value is None ***
    limits.p_low = limits.actual_min if limits.actual_min is not None else 0.0
    limits.p_high = limits.actual_max if limits.actual_max is not None else 0.0
    # *** FIX END ***
    return limits


def _get_smart_limits(data: np.ndarray) -> ContrastLimits:
    """Calculates limits using dynamic 'early' cutoff for high and 1st percentile for low."""
    limits = _get_base_stats(data)
    # *** FIX START: Provide default 0.0 if optional value is None ***
    limits.p_low = limits.p1 if limits.p1 is not None else 0.0
    limits.p_high = limits.smart_early if limits.smart_early is not None else 0.0
    # *** FIX END ***
    return limits


def _get_smart_late_limits(data: np.ndarray) -> ContrastLimits:
    """Calculates limits using dynamic 'late' cutoff for high and 1st percentile for low."""
    limits = _get_base_stats(data)
    # *** FIX START: Provide default 0.0 if optional value is None ***
    limits.p_low = limits.p1 if limits.p1 is not None else 0.0
    limits.p_high = limits.smart_late if limits.smart_late is not None else 0.0
    # *** FIX END ***
    return limits


# Dictionary mapping stretch modes to their limit calculation functions
LIMIT_CALCULATORS: Dict[str, LimitCalculator] = {
    "imagej-auto": _get_imagej_limits,
    "max": _get_max_limits,
    "smart": _get_smart_limits,
    "smart-late": _get_smart_late_limits,
}

# --- Main Function to Calculate Limits ---
def calculate_limits_only(img: np.ndarray, stretch_mode: str) -> ContrastLimits:
    """
    Calculates contrast limits based on stretch mode without applying scaling.
    """
    if img.size == 0:
        log.warning("calculate_limits_only received empty image.")
        return _get_base_stats(np.array([], dtype=img.dtype))

    try:
        data = img[img > 0].flatten()
        if data.size == 0:
            log.warning("No positive pixel values found for limit calculation. Using all pixels.")
            data = img.flatten()
            if data.size == 0:
                 log.warning("Image data is effectively empty after filtering. Returning default limits.")
                 return _get_base_stats(np.array([], dtype=img.dtype))

        calculator = LIMIT_CALCULATORS.get(stretch_mode)
        if calculator:
            limits = calculator(data)
            log.debug(
                f"Calculated limits for mode '{stretch_mode}' (pass 1): p_low={limits.p_low:.2f}, p_high={limits.p_high:.2f}"
            )
        else:
            log.error(
                f"Unknown stretch_mode: '{stretch_mode}' during limit calculation. Using 'max'."
            )
            limits = _get_max_limits(data) # Fallback uses _get_base_stats

        return limits

    except Exception as e:
        log.error(
            f"Error calculating limits only (mode: {stretch_mode}): {e}. Returning default max limits.",
            exc_info=True,
        )
        try:
             # Fallback to simple min/max of original image
             all_data = img.flatten()
             limits = _get_max_limits(all_data) # Use helper to get consistent defaults
        except Exception as fallback_e:
             log.error(f"Error during fallback limit calculation: {fallback_e}. Returning zero limits.")
             limits = ContrastLimits(p_low=0.0, p_high=0.0, actual_min=0.0, actual_max=0.0)
        return limits


def apply_autocontrast_8bit(
    img: np.ndarray,
    stretch_mode: str = "smart",
    global_limits_tuple: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, ContrastLimits]:
    """
    Applies contrast stretching to an image and converts it to 8-bit.
    """
    limits = ContrastLimits()

    if img.size == 0:
        log.warning("apply_autocontrast_8bit received empty image.")
        limits = _get_base_stats(np.array([], dtype=img.dtype))
        return np.zeros_like(img, dtype=np.uint8), limits

    try:
        # --- Determine limits ---
        if global_limits_tuple is not None:
            # Use provided global limits
            data_nz = img[img > 0].flatten()
            if data_nz.size == 0:
                data_nz = img.flatten()
            limits = _get_base_stats(data_nz)
            # Override p_low/p_high with the provided global values
            limits.p_low, limits.p_high = global_limits_tuple
            log.debug(
                f"Using provided global limits: ({limits.p_low:.2f}, {limits.p_high:.2f})"
            )
            actual_stretch_mode = "timeseries-global"
        else:
            # Calculate limits based on stretch_mode using the main function
            limits = calculate_limits_only(img, stretch_mode)
            actual_stretch_mode = stretch_mode
            log.debug(
                 f"Using calculated limits for mode '{stretch_mode}': ({limits.p_low:.2f}, {limits.p_high:.2f})"
            )
            if stretch_mode not in LIMIT_CALCULATORS:
                 actual_stretch_mode = "max (fallback)"

        # --- Validate bounds and Apply Scaling ---
        p_low = limits.p_low
        p_high = limits.p_high

        if p_high <= p_low:
            if p_low == 0 and p_high == 0 and np.all(img == 0):
                scaled_uint8 = np.zeros_like(img, dtype=np.uint8)
            else:
                log.warning(
                    f"Contrast bounds collapsed or invalid (low={p_low:.2f}, high={p_high:.2f}). Clipping sharply at p_low."
                )
                scaled_uint8 = np.where(img <= p_low, 0, 255).astype(np.uint8)
            return scaled_uint8, limits

        img_float = img.astype(np.float32)
        denominator = p_high - p_low
        # Avoid division by zero just in case, though p_high <= p_low check should prevent it
        if denominator == 0:
             log.warning("Scaling denominator is zero after checks, clipping sharply.")
             scaled_uint8 = np.where(img <= p_low, 0, 255).astype(np.uint8)
             return scaled_uint8, limits

        scaled_float = (img_float - p_low) / denominator
        scaled_float = np.clip(scaled_float, 0.0, 1.0)
        scaled_uint8 = (scaled_float * 255.0).astype(np.uint8)

        log.debug(
            f"Contrast stretching applied successfully using mode '{actual_stretch_mode}'."
        )
        return scaled_uint8, limits

    except Exception as e:
        log.error(
            f"Error applying autocontrast (mode: {stretch_mode}): {e}. Falling back to simple max scaling.",
            exc_info=True,
        )
        try:
             # Fallback to simple min/max scaling of original image
             all_data = img.flatten()
             limits = _get_max_limits(all_data) # Use helper for consistency
             actual_min = limits.actual_min if limits.actual_min is not None else 0.0
             actual_max = limits.actual_max if limits.actual_max is not None else 0.0

             if actual_max > actual_min:
                 scaled_fallback = (img.astype(np.float32) - actual_min) / (actual_max - actual_min)
                 scaled_fallback = np.clip(scaled_fallback, 0.0, 1.0)
                 scaled_fallback = (scaled_fallback * 255.0).astype(np.uint8)
             else:
                 scaled_fallback = np.zeros_like(img, dtype=np.uint8)
        except Exception as fallback_e:
             log.error(f"Error during fallback scaling: {fallback_e}. Returning zero image.")
             limits = ContrastLimits(p_low=0.0, p_high=0.0, actual_min=0.0, actual_max=0.0)
             scaled_fallback = np.zeros_like(img, dtype=np.uint8)

        return scaled_fallback, limits

