# src/pixelpacker/__init__.py

import importlib.metadata
import logging

log = logging.getLogger(__name__)

try:
    # Read the version from the installed package's metadata
    __version__ = importlib.metadata.version("pixelpacker")
    log.debug(f"pixelpacker version loaded from metadata: {__version__}")
except importlib.metadata.PackageNotFoundError:
    # Fallback if the package is not installed (e.g., running from source)
    log.warning("pixelpacker package not installed, version information unavailable.")
    __version__ = "0.0.0-unknown"

# --- Basic Logging Setup ---
# Add a NullHandler to the package's root logger to avoid "No handler found"
# warnings if used as a library and the application doesn't configure logging.
# The application (like cli.py) is still responsible for adding actual handlers.
handler = logging.NullHandler()
log.addHandler(handler)
log.debug("NullHandler added to pixelpacker root logger.")