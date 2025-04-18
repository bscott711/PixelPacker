# tiff_preprocessor/stretch.py

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

PERCENTILE_1 = 1.0  # Renamed from PERCENTILE_LOW_SMART for clarity
PERCENTILE_LOW_IMAGEJ = 0.35
PERCENTILE_HIGH_IMAGEJ = 99.65

log = logging.getLogger(__name__)


# --- Data Structure for Detailed Limits ---
@dataclass
class ContrastLimits:
    """Holds various calculated intensity limits for stretching and debugging."""

    p_low: float = 0.0  # The actual lower bound used for stretching
    p_high: float = 0.0  # The actual upper bound used for stretching
    # --- Values primarily for debug histogram ---
    p1: Optional[float] = None  # 1st percentile
    p035: Optional[float] = None  # 0.35 percentile (ImageJ Low)
    p9965: Optional[float] = None  # 99.65 percentile (ImageJ High)
    smart_early: Optional[float] = None  # Calculated dynamic early cutoff
    smart_late: Optional[float] = None  # Calculated dynamic late cutoff
    actual_min: Optional[float] = None  # Actual minimum of non-zero data
    actual_max: Optional[float] = None  # Actual maximum of non-zero data


# --- Dynamic Cutoff Calculation (Unchanged from previous refactor) ---
def compute_dynamic_cutoffs(pixels: np.ndarray) -> Tuple[float, float]:
    """
    Computes dynamic intensity cutoffs based on the slope of the log-histogram.
    (Implementation details omitted for brevity - assume it's the version from the previous step)
    """
    if pixels.size == 0:
        log.warning("compute_dynamic_cutoffs received empty pixel array.")
        return 0.0, 0.0
    pixels_min = float(pixels.min())
    pixels_max = float(pixels.max())
    if pixels_min == pixels_max:
        return pixels_min, pixels_max
    try:
        # ... (rest of the implementation from previous refactoring step) ...
        hist, bin_edges = np.histogram(
            pixels, bins=HIST_BINS, range=(pixels_min, pixels_max)
        )
        log_hist = np.log10(hist.astype(np.float32) + LOG_HIST_EPSILON)
        slope = np.gradient(log_hist)
        smoothed_slope = gaussian_filter1d(slope, sigma=SMOOTHING_SIGMA, mode="reflect")
        early_cutoff_candidate: Optional[float] = None
        late_cutoff_candidate: Optional[float] = None
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        search_end_early = len(smoothed_slope) - SLOPE_WINDOW_SIZE - MIN_SEARCH_OFFSET
        for i in range(MIN_SEARCH_OFFSET, search_end_early):
            window = smoothed_slope[i : i + SLOPE_WINDOW_SIZE]
            if np.all(window < SLOPE_THRESHOLD):
                early_cutoff_candidate = float(bin_centers[i])
                break
        search_start_late = len(smoothed_slope) - MIN_SEARCH_OFFSET - 1
        search_end_late = MIN_SEARCH_OFFSET + SLOPE_WINDOW_SIZE - 1
        for i in range(search_start_late, search_end_late, -1):
            window = smoothed_slope[i - SLOPE_WINDOW_SIZE + 1 : i + 1]
            if np.all(window < SLOPE_THRESHOLD):
                late_cutoff_candidate = float(bin_centers[i])
                break
        early_cutoff: float
        late_cutoff: float
        if early_cutoff_candidate is None:
            early_cutoff = float(np.percentile(pixels, FALLBACK_EARLY_PERCENTILE))
        else:
            early_cutoff = early_cutoff_candidate
        if late_cutoff_candidate is None:
            late_cutoff = float(np.percentile(pixels, FALLBACK_LATE_PERCENTILE))
        else:
            late_cutoff = late_cutoff_candidate
        if early_cutoff >= late_cutoff:
            early_cutoff, late_cutoff = pixels_min, pixels_max
        return float(early_cutoff), float(late_cutoff)
    except Exception as e:
        log.error(
            f"Error computing dynamic cutoffs: {e}. Falling back to min/max.",
            exc_info=True,
        )
        return pixels_min, pixels_max


# --- Limit Calculation Helper Functions (Modified to return ContrastLimits) ---

LimitCalculator = Callable[[np.ndarray], ContrastLimits]


def _get_base_stats(data: np.ndarray) -> ContrastLimits:
    """Calculates common stats needed for most modes."""
    limits = ContrastLimits()
    if data.size > 0:
        limits.actual_min = float(data.min())
        limits.actual_max = float(data.max())
        # Calculate all potentially needed percentiles/cutoffs once
        limits.p1 = float(np.percentile(data, PERCENTILE_1))
        limits.p035, limits.p9965 = map(
            float, np.percentile(data, (PERCENTILE_LOW_IMAGEJ, PERCENTILE_HIGH_IMAGEJ))
        )
        limits.smart_early, limits.smart_late = compute_dynamic_cutoffs(data)
    else:
        # Handle empty data case
        limits.actual_min = 0.0
        limits.actual_max = 0.0
        limits.p1 = 0.0
        limits.p035 = 0.0
        limits.p9965 = 0.0
        limits.smart_early = 0.0
        limits.smart_late = 0.0
    return limits


def _get_imagej_limits(data: np.ndarray) -> ContrastLimits:
    """Calculates contrast limits using ImageJ's default percentile method."""
    limits = _get_base_stats(data)
    limits.p_low = limits.p035 if limits.p035 is not None else 0.0
    limits.p_high = limits.p9965 if limits.p9965 is not None else 0.0
    return limits


def _get_max_limits(data: np.ndarray) -> ContrastLimits:
    """Calculates contrast limits using the full min/max range of the data."""
    limits = _get_base_stats(data)
    limits.p_low = limits.actual_min if limits.actual_min is not None else 0.0
    limits.p_high = limits.actual_max if limits.actual_max is not None else 0.0
    return limits


def _get_smart_limits(data: np.ndarray) -> ContrastLimits:
    """Calculates limits using dynamic 'early' cutoff for high and 1st percentile for low."""
    limits = _get_base_stats(data)
    limits.p_low = limits.p1 if limits.p1 is not None else 0.0
    limits.p_high = limits.smart_early if limits.smart_early is not None else 0.0
    return limits


def _get_smart_late_limits(data: np.ndarray) -> ContrastLimits:
    """Calculates limits using dynamic 'late' cutoff for high and 1st percentile for low."""
    limits = _get_base_stats(data)
    limits.p_low = limits.p1 if limits.p1 is not None else 0.0
    limits.p_high = limits.smart_late if limits.smart_late is not None else 0.0
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
    Designed for the first pass of global contrast calculation.

    Args:
        img: Input image data (can be high bit depth).
        stretch_mode: The contrast stretching method ('smart', 'smart-late',
                      'imagej-auto', 'max'). 'timeseries-global' is not applicable here.

    Returns:
        A ContrastLimits object containing calculated statistics. Returns default
        limits if image is empty or stretch_mode is invalid.
    """
    limits = ContrastLimits()  # Initialize empty limits
    if img.size == 0:
        log.warning("calculate_limits_only received empty image.")
        return limits

    try:
        data = img[img > 0] if np.any(img > 0) else img.flatten()
        if data.size == 0:
            data = img.flatten()

        calculator = LIMIT_CALCULATORS.get(stretch_mode)
        if calculator:
            limits = calculator(data)  # Calculate all relevant limits
            log.debug(
                f"Calculated limits for mode '{stretch_mode}' (pass 1): p_low={limits.p_low:.2f}, p_high={limits.p_high:.2f}"
            )
        else:
            log.error(
                f"Unknown stretch_mode: '{stretch_mode}' during limit calculation. Using 'max'."
            )
            limits = _get_max_limits(data)  # Fallback

        # ***** FIX: Add missing return statement *****
        return limits  # Return the calculated limits from the try block
        # ***** END FIX *****

    except Exception as e:
        # ... (exception handling block remains the same) ...
        log.error(
            f"Error calculating limits only (mode: {stretch_mode}): {e}. Returning default.",
            exc_info=True,
        )
        limits = ContrastLimits()  # Create new default object
        try:
            limits.actual_min = float(np.min(img))
            limits.actual_max = float(np.max(img))
        except ValueError:
            limits.actual_min = 0.0
            limits.actual_max = 0.0
        limits.p_low = limits.actual_min if limits.actual_min is not None else 0.0
        limits.p_high = limits.actual_max if limits.actual_max is not None else 0.0
        if limits.p_high <= limits.p_low:
            limits.p_high = limits.p_low
        return limits  # Return the fallback limits


def apply_autocontrast_8bit(
    img: np.ndarray,
    stretch_mode: str = "smart",
    global_limits_tuple: Optional[Tuple[float, float]] = None,  # Keep this arg
) -> Tuple[np.ndarray, ContrastLimits]:
    """
    Applies contrast stretching to an image and converts it to 8-bit.
    If global_limits_tuple is provided, it overrides other modes and uses those limits.
    """
    limits = ContrastLimits()  # Initialize empty limits

    if img.size == 0:
        log.warning("apply_autocontrast_8bit received empty image.")
        return np.zeros_like(img, dtype=np.uint8), limits

    try:
        data = img[img > 0] if np.any(img > 0) else img.flatten()
        if data.size == 0:
            data = img.flatten()

        # --- Determine limits ---
        # ****** PRIORITY TO GLOBAL LIMITS ******
        if global_limits_tuple is not None:
            # If global limits are given, USE THEM. Calculate base stats for debug info.
            limits = _get_base_stats(
                data
            )  # Get all stats like min/max, percentiles etc.
            limits.p_low, limits.p_high = global_limits_tuple  # Override p_low/p_high
            log.debug(
                f"Using provided global limits: ({limits.p_low:.2f}, {limits.p_high:.2f})"
            )
            actual_stretch_mode = "timeseries-global"  # Note the mode used
        else:
            # If no global limits, calculate based on stretch_mode
            actual_stretch_mode = stretch_mode  # Use the requested mode
            calculator = LIMIT_CALCULATORS.get(stretch_mode)
            if calculator:
                limits = calculator(data)
                log.debug(
                    f"Calculated limits for mode '{stretch_mode}': ({limits.p_low:.2f}, {limits.p_high:.2f})"
                )
            else:
                log.error(
                    f"Unknown stretch_mode: '{stretch_mode}'. Falling back to 'max' limits."
                )
                limits = _get_max_limits(data)
                actual_stretch_mode = "max (fallback)"

        # --- Validate bounds and Apply Scaling ---
        p_low = limits.p_low
        p_high = limits.p_high

        # ... (rest of scaling logic remains the same as previous version) ...
        if p_high <= p_low:
            if p_low == 0 and p_high == 0:
                scaled_uint8 = np.zeros_like(img, dtype=np.uint8)
            else:
                log.warning(
                    f"Contrast bounds collapsed or invalid (low={p_low:.2f}, high={p_high:.2f}). Clipping sharply."
                )
                scaled_uint8 = np.where(img <= p_low, 0, 255).astype(np.uint8)
            return (
                scaled_uint8,
                limits,
            )  # Return clipped/zero image, but calculated limits

        img_float = img.astype(np.float32)
        denominator = p_high - p_low
        scaled_float = np.clip((img_float - p_low) / denominator, 0.0, 1.0)
        scaled_uint8 = (scaled_float * 255.0).astype(np.uint8)

        log.debug(
            f"Contrast stretching applied successfully using mode '{actual_stretch_mode}'."
        )
        return scaled_uint8, limits

    except Exception as e:
        # ... (fallback logic remains the same) ...
        log.error(
            f"Error applying autocontrast (mode: {stretch_mode}): {e}. Falling back to simple max scaling.",
            exc_info=True,
        )
        limits = ContrastLimits()
        limits.actual_min = float(np.min(img))
        limits.actual_max = float(np.max(img))
        limits.p_low = limits.actual_min
        limits.p_high = limits.actual_max
        if limits.actual_max > limits.actual_min:
            scaled_fallback = (img.astype(np.float32) - limits.actual_min) / (
                limits.actual_max - limits.actual_min
            )
            scaled_fallback = (scaled_fallback * 255.0).astype(np.uint8)
        else:
            scaled_fallback = np.zeros_like(img, dtype=np.uint8)
        return scaled_fallback, limits
