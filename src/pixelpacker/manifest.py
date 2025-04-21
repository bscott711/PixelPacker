# src/pixelpacker/manifest.py

import json
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

# Import necessary config and data models
from .data_models import PreprocessingConfig, ProcessingResult, VolumeLayout
from .utils import log  # Use centralized logger

TimepointResult = Dict[str, Any]  # Define type alias locally or import


# === Metadata Finalization & Manifest Writing ===


def finalize_metadata(
    results: List[ProcessingResult],
    layout: VolumeLayout,
    global_z_range: Tuple[int, int],
    global_contrast_ranges_used: Optional[Dict[int, Tuple[float, float]]],
) -> Dict[str, Any]:
    """
    Aggregates results from Pass 2 into the final metadata structure.

    Args:
        results: List of ProcessingResult objects from successful Pass 2 tasks.
        layout: The calculated VolumeLayout.
        global_z_range: The Z-range (start, end) used for cropping.
        global_contrast_ranges_used: The global contrast ranges if they were applied,
                                     otherwise None. Used to populate the
                                     'global_intensity' field directly if available.

    Returns:
        A dictionary representing the final metadata (manifest content).
    """
    log.info("📝 Finalizing metadata for manifest...")
    metadata: Dict[str, Any] = {
        "tile_layout": {"cols": layout.cols, "rows": layout.rows},
        "volume_size": {
            "width": layout.width,
            "height": layout.height,
            "depth": layout.depth,  # Depth after global Z-crop
        },
        "channels": 0,  # Will be calculated
        "global_z_crop_range": list(global_z_range),  # [start, end]
        "timepoints": [],
        # This field structure might change based on multi-channel needs
        "global_intensity": {},
    }

    if not results:
        log.warning(
            "No successful processing results found. Manifest may be incomplete."
        )
        # Return the basic structure even if empty
        return metadata

    # Group results by timepoint
    timepoints_results: DefaultDict[str, TimepointResult] = defaultdict(
        lambda: {"time": None, "files": {}}  # Structure per timepoint
    )
    max_channel = -1

    for res in results:
        res_time = res.time_id
        res_ch = res.channel
        max_channel = max(max_channel, res_ch)  # Track highest channel index

        # Store time ID if not already set for this timepoint
        if timepoints_results[res_time]["time"] is None:
            timepoints_results[res_time]["time"] = res_time

        # Add file info under the correct channel key (e.g., "c0", "c1")
        timepoints_results[res_time]["files"][f"c{res_ch}"] = {
            "file": res.filename,
            "p_low": res.p_low,  # Contrast limits actually used
            "p_high": res.p_high,
        }

    # Set total number of channels (0-indexed, so max_channel + 1)
    metadata["channels"] = max_channel + 1

    # Sort timepoints chronologically based on the time ID string
    # (assuming format like "stack0000", "stack0001")
    processed_time_ids = sorted(timepoints_results.keys())
    metadata["timepoints"] = [
        timepoints_results[tid]
        for tid in processed_time_ids
        if timepoints_results[tid][
            "files"
        ]  # Only include timepoints with processed files
    ]

    # Populate the 'global_intensity' field
    final_global_intensity = {}
    if global_contrast_ranges_used:
        # If global contrast was applied, use those ranges directly
        log.info("Populating manifest 'global_intensity' from applied global ranges.")
        for ch_id, (low, high) in global_contrast_ranges_used.items():
            final_global_intensity[f"c{ch_id}"] = {"p_low": low, "p_high": high}
    else:
        # If per-image contrast was used, calculate the overall min/max
        # *retrospectively* from the individual results for informational purposes.
        log.info(
            "Calculating retrospective 'global_intensity' from per-image results."
            " Note: These ranges were not necessarily applied globally."
        )
        all_channels = set(res.channel for res in results)
        for ch in all_channels:
            channel_results = [res for res in results if res.channel == ch]
            if channel_results:
                # Find min of lows and max of highs for this channel across all timepoints
                low = min(res.p_low for res in channel_results)
                high = max(res.p_high for res in channel_results)
                final_global_intensity[f"c{ch}"] = {"p_low": low, "p_high": high}
            else:
                # Should not happen if all_channels is derived from results, but safe fallback
                final_global_intensity[f"c{ch}"] = {"p_low": 0.0, "p_high": 0.0}
    metadata["global_intensity"] = final_global_intensity

    log.info("Metadata finalized successfully.")
    return metadata


def write_manifest(metadata: Dict[str, Any], config: PreprocessingConfig):
    """
    Writes the finalized metadata dictionary to 'manifest.json' in the output folder.

    Args:
        metadata: The dictionary returned by `finalize_metadata`.
        config: The global preprocessing configuration (used for output path and dry_run).
    """
    if config.dry_run:
        log.info("--dry-run enabled, skipping manifest file writing.")
        # Optionally log the manifest content in debug mode?
        # log.debug(f"Dry run manifest content:\n{json.dumps(metadata, indent=2)}")
        return

    # Check if there's actually anything to write
    if not metadata.get("timepoints"):
        log.warning("No timepoints recorded in metadata. Skipping manifest writing.")
        return

    manifest_path = config.output_folder / "manifest.json"
    log.info(f"Writing metadata manifest to {manifest_path}...")

    try:
        with open(manifest_path, "w") as f:
            # Dump JSON with indentation for readability
            json.dump(metadata, f, indent=2)
        log.info("Manifest file saved successfully.")
    except OSError as e:
        log.error(
            f"❌ Failed to write manifest file '{manifest_path}': {e}",
            exc_info=config.debug,
        )
    except Exception as e:
        # Catch other potential errors during JSON serialization etc.
        log.error(f"❌ Unexpected error writing manifest: {e}", exc_info=config.debug)
