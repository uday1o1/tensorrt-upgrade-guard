"""TensorRT UpgradeGuard host control plane.

This package intentionally does not import TensorRT, CUDA Python, or another GPU runtime.
"""

from upgrade_guard._version import __version__

__all__ = ["__version__"]
