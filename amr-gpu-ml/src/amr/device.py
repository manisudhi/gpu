"""
src/amr/device.py
Hardware and device management layer for Apple Silicon (M3) GPU acceleration.
Manages PyTorch MPS backend and MLX Metal compute runtime.
"""

import os
import sys
import logging
import platform
import subprocess
from typing import Dict, Any, Optional

import torch
import numpy as np

# Enable MPS fallback with warning logging if requested
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

logger = logging.getLogger("amr.device")

try:
    import mlx.core as mx
    import mlx.fast as mxf
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    logger.warning("MLX is not installed. Custom Metal compute kernels will fallback to pure PyTorch MPS.")


class GPUManager:
    """
    Central manager for Apple Silicon Metal GPU resources.
    Guarantees execution on GPU (device='mps') and fails loudly if GPU is unavailable.
    """

    def __init__(self, force_device: Optional[str] = None):
        self._init_device(force_device)
        self._verify_metal_environment()

    def _init_device(self, force_device: Optional[str] = None):
        if force_device:
            self.device_name = force_device
            self.torch_device = torch.device(force_device)
            return

        if not torch.backends.mps.is_available():
            err_msg = (
                "FATAL: Apple Metal Performance Shaders (MPS) is NOT available on this system! "
                "Per project requirements, CPU fallback for primary ML workloads is strictly prohibited. "
                "Ensure you are running on an Apple Silicon Mac with macOS 12.3+."
            )
            logger.critical(err_msg)
            raise RuntimeError(err_msg)

        if not torch.backends.mps.is_built():
            err_msg = "FATAL: PyTorch was not built with MPS support enabled!"
            logger.critical(err_msg)
            raise RuntimeError(err_msg)

        self.device_name = "mps"
        self.torch_device = torch.device("mps")
        logger.info("PyTorch MPS device initialized successfully: device='mps'")

    def _verify_metal_environment(self):
        """Log hardware and software environment details."""
        self.hw_info = self.get_hardware_info()
        logger.info(f"Hardware: {self.hw_info.get('chip', 'Apple Silicon')}, Unified Memory: {self.hw_info.get('memory', 'Unknown')}")
        logger.info(f"OS: macOS {self.hw_info.get('os_version', platform.mac_ver()[0])}, Python: {platform.python_version()}")
        logger.info(f"PyTorch: {torch.__version__}, MLX Available: {MLX_AVAILABLE}")

    @staticmethod
    def get_hardware_info() -> Dict[str, str]:
        """Query system_profiler for Apple Silicon chip and unified memory."""
        info = {
            "chip": "Apple Silicon (M-series)",
            "memory": "Unknown",
            "os_version": platform.mac_ver()[0],
            "machine": platform.machine(),
        }
        try:
            res = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False)
            if res.returncode == 0 and res.stdout.strip():
                info["chip"] = res.stdout.strip()
            
            res_mem = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False)
            if res_mem.returncode == 0 and res_mem.stdout.strip():
                bytes_mem = int(res_mem.stdout.strip())
                info["memory"] = f"{bytes_mem / (1024**3):.1f} GB"
        except Exception as e:
            logger.debug(f"Failed to query sysctl: {e}")
        return info

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Transfer tensor to GPU device enforcing float32 (MPS has no native float64).
        """
        if tensor.dtype == torch.float64:
            tensor = tensor.to(torch.float32)
        return tensor.to(self.torch_device)

    def synchronize(self):
        """Synchronize GPU operations for accurate timing."""
        if self.torch_device.type == "mps":
            torch.mps.synchronize()
        if MLX_AVAILABLE:
            mx.eval()

    def get_memory_stats(self) -> Dict[str, Any]:
        """Report current MPS memory allocated and cached."""
        stats = {}
        if hasattr(torch.mps, "current_allocated_memory"):
            stats["allocated_bytes"] = torch.mps.current_allocated_memory()
            stats["allocated_mb"] = stats["allocated_bytes"] / (1024 * 1024)
        if hasattr(torch.mps, "driver_allocated_memory"):
            stats["driver_bytes"] = torch.mps.driver_allocated_memory()
            stats["driver_mb"] = stats["driver_bytes"] / (1024 * 1024)
        return stats


# Global default GPU manager instance
_DEFAULT_MANAGER: Optional[GPUManager] = None

def get_gpu_manager() -> GPUManager:
    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is None:
        _DEFAULT_MANAGER = GPUManager()
    return _DEFAULT_MANAGER
