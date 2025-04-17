import logging
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

# Constants for histogram analysis and stretching
HIST_BINS = 512
LOG_HIST_EPSILON = 1e-6  # Avoid log(0)
SMOOTHING_SIGMA = 3.0
SLOPE_THRESHOLD = -0.1  # Threshold for detecting significant drop-off in log histogram
SLOPE_WINDOW_SIZE = 5  # Number of consecutive bins to check for slope threshold
MIN_SEARCH_OFFSET = 10 # Offset from histogram ends for slope search
# --- Define fallback percentiles ---
FALLBACK_EARLY_PERCENTILE = 99.0  # Fallback if early cutoff detection fails
FALLBACK_LATE_PERCENTILE = 99.9   # Fallback if late cutoff detection fails


def compute_dynamic_cutoffs(pixels: np.ndarray) -> Tuple[float, float]:
    """
    Computes intensity cutoffs based on the slope of the log-histogram.

    This method aims to find points where the pixel count significantly drops off,
    indicating potential upper bounds for contrast stretching. It calculates
    an "early" cutoff (first significant drop) and a "late" cutoff (last
    significant drop before the maximum).

    Args:
        pixels: A 1D NumPy array of non-zero pixel intensities.

    Returns:
        A tuple containing (early_cutoff, late_cutoff). Uses fallback percentiles
        or (min, max) if input is empty, flat, or analysis fails.
    """
    if pixels.size == 0:
        logging.warning("compute_dynamic_cutoffs received empty pixel array.")
        return 0.0, 0.0

    # Explicitly cast results of np.min/max to float
    pixels_min: float = float(pixels.min())
    pixels_max: float = float(pixels.max())

    if pixels_min == pixels_max:
        # Handle flat data where histogram analysis is meaningless
        return pixels_min, pixels_max

    try:
        hist, bin_edges = np.histogram(pixels, bins=HIST_BINS, range=(pixels_min, pixels_max))
   

        # Avoid log(0) and ensure float type
        log_hist = np.log10(hist.astype(np.float32) + LOG_HIST_EPSILON)

        # Calculate slope and smooth it
        slope = np.gradient(log_hist)
        smoothed_slope = gaussian_filter1d(slope, sigma=SMOOTHING_SIGMA, mode='reflect')

        # --- Find Cutoffs ---
        early_cutoff_candidate: Optional[float] = None # Use candidate suffix to avoid type conflict before cast
        late_cutoff_candidate: Optional[float] = None

        # Search for the "early" cutoff (first significant drop from left)
        search_end = len(smoothed_slope) - SLOPE_WINDOW_SIZE - MIN_SEARCH_OFFSET
        for i in range(MIN_SEARCH_OFFSET, search_end):
            window = smoothed_slope[i : i + SLOPE_WINDOW_SIZE]
            if np.all(window < SLOPE_THRESHOLD):
                early_cutoff_candidate = float(bin_edges[i]) # Cast here
                break

        # Search for the "late" cutoff (last significant drop from right)
        search_start = len(smoothed_slope) - MIN_SEARCH_OFFSET - 1
        for i in range(search_start, MIN_SEARCH_OFFSET + SLOPE_WINDOW_SIZE -1, -1):
            window = smoothed_slope[i - SLOPE_WINDOW_SIZE + 1 : i + 1]
            if np.all(window < SLOPE_THRESHOLD):
                late_cutoff_candidate = float(bin_edges[i + 1]) # Cast here
                break

        # --- Apply Fallbacks and Check Consistency ---
        early_cutoff: float
        late_cutoff: float

        # Apply fallback percentile if early cutoff wasn't found
        if early_cutoff_candidate is None:
            early_cutoff = float(np.percentile(pixels, FALLBACK_EARLY_PERCENTILE)) # Cast percentile result
            logging.debug(f"Early dynamic cutoff not found, falling back to {FALLBACK_EARLY_PERCENTILE}th percentile: {early_cutoff:.2f}")
        else:
            early_cutoff = early_cutoff_candidate

        # Apply fallback percentile if late cutoff wasn't found
        if late_cutoff_candidate is None:
            late_cutoff = float(np.percentile(pixels, FALLBACK_LATE_PERCENTILE)) # Cast percentile result
            logging.debug(f"Late dynamic cutoff not found, falling back to {FALLBACK_LATE_PERCENTILE}th percentile: {late_cutoff:.2f}")
        else:
            late_cutoff = late_cutoff_candidate


        # Ensure early <= late, otherwise reset completely to min/max as ultimate fallback
        if early_cutoff >= late_cutoff:
            logging.warning(
                f"Dynamic/fallback cutoffs computed out of order or too close "
                f"(early={early_cutoff:.2f}, late={late_cutoff:.2f}). "
                f"Resetting to min/max ({pixels_min:.2f}, {pixels_max:.2f})."
            )
            # pixels_min/max are already floats from start of function
            early_cutoff, late_cutoff = pixels_min, pixels_max

        # Explicitly cast return values to satisfy Tuple[float, float] hint
        # This addresses the error on line 101
        return float(early_cutoff), float(late_cutoff)

    except Exception as e:
        logging.error(f"Error computing dynamic cutoffs: {e}. Falling back to min/max.", exc_info=True)
        # pixels_min/max are already floats
        return pixels_min, pixels_max


def apply_autocontrast_8bit(
    img: np.ndarray,
    stretch_mode: str = "smart",
    global_limits: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Applies contrast stretching to an image and converts it to 8-bit.

    Args:
        img: Input image as a NumPy array.
        stretch_mode: The contrast stretching method to use. Options:
            'smart', 'smart-late', 'imagej-auto', 'max', 'timeseries-global'.
        global_limits: A tuple (min_intensity, max_intensity) to use when
            stretch_mode is 'timeseries-global'.

    Returns:
        A tuple containing:
            - The stretched image as an 8-bit NumPy array.
            - The calculated lower intensity bound (p_low).
            - The calculated upper intensity bound (p_high).
    """
    p_low: float = 0.0
    p_high: float = 0.0

    if img.size == 0:
        logging.warning("apply_autocontrast_8bit received empty image.")
        return np.zeros_like(img, dtype=np.uint8), p_low, p_high

    try:
        # Consider only non-zero pixels for calculating statistics,
        # unless the image is all zeros.
        data = img[img > 0] if np.any(img > 0) else img.flatten()

        if data.size == 0: # Handles case where img > 0 is all False but img is not empty
             data = img.flatten() # Use all pixels if only zeros present

        # Determine p_low and p_high based on stretch_mode
        if stretch_mode == "timeseries-global":
            if global_limits is not None:
                # Assume global_limits provides Python floats
                p_low, p_high = global_limits
            else:
                logging.warning("Stretch mode 'timeseries-global' selected but no global_limits provided. Falling back to 'max'.")
                stretch_mode = "max" # Fallback if limits missing

        # Calculate limits if not using global ones
        if stretch_mode != "timeseries-global":
            if data.size == 0 or np.all(data == 0):
                 p_low, p_high = 0.0, 0.0 # Cannot compute stats on empty/zero data
            else:
                # Define constants used locally
                PERCENTILE_LOW_SMART = 1.0
                PERCENTILE_LOW_IMAGEJ = 0.35
                PERCENTILE_HIGH_IMAGEJ = 99.65

                if stretch_mode == "imagej-auto":
                    # Explicitly cast results of np.percentile
                    # This addresses the error near line 181
                    p_low_np, p_high_np = np.percentile(data, (PERCENTILE_LOW_IMAGEJ, PERCENTILE_HIGH_IMAGEJ))
                    p_low, p_high = float(p_low_np), float(p_high_np)
                elif stretch_mode == "max":
                    # Explicitly cast results of np.min/max
                    p_low, p_high = float(data.min()), float(data.max())
                elif stretch_mode in ["smart", "smart-late"]:
                    # compute_dynamic_cutoffs now returns Tuple[float, float]
                    dynamic_early, dynamic_late = compute_dynamic_cutoffs(data)
                    # Explicitly cast result of np.percentile
                    p_low = float(np.percentile(data, PERCENTILE_LOW_SMART))
                    # dynamic_early/late are already floats
                    p_high = dynamic_early if stretch_mode == "smart" else dynamic_late
                else:
                    raise ValueError(f"Unknown stretch_mode: {stretch_mode}")

        # Ensure bounds are valid floats (redundant now with casting above, but safe)
        p_low = float(p_low)
        p_high = float(p_high)

        # Handle cases where bounds are invalid or equal
        if p_high <= p_low:
            if p_low == 0 and p_high == 0:
                 return np.zeros_like(img, dtype=np.uint8), p_low, p_high
            else:
                logging.warning(f"Contrast bounds collapsed or invalid (low={p_low}, high={p_high}). Clipping sharply.")
                scaled = np.where(img <= p_low, 0, 255)
                return scaled.astype(np.uint8), p_low, p_high


        # Apply scaling, ensuring denominator is not zero (handled by p_high <= p_low check)
        denominator = p_high - p_low
        img_float = img.astype(np.float32)
        scaled = np.clip((img_float - p_low) / denominator, 0, 1) * 255

        # Final return requires floats
        return scaled.astype(np.uint8), float(p_low), float(p_high)

    except Exception as e:
        logging.error(f"Error applying autocontrast (mode: {stretch_mode}): {e}. Falling back to simple max scaling.", exc_info=True)
        # Safer fallback: check max value before division
        img_min_val = float(np.min(img)) # Cast fallback bounds
        img_max_val = float(np.max(img)) # Cast fallback bounds

        if img_max_val > 0:
            # Ensure intermediate calculation uses float32 or float64
            fallback = ((img.astype(np.float32) / img_max_val) * 255)
        else:
            fallback = np.zeros_like(img, dtype=np.float32)
        # Return float bounds
        return fallback.astype(np.uint8), img_min_val, img_max_val