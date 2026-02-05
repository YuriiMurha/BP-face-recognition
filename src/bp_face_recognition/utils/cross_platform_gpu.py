#!/usr/bin/env python3
"""
Cross-Platform GPU Detection and Configuration Utilities

This module provides comprehensive GPU detection across different platforms:
- Windows (CPU-only, with WSL detection)
- Linux (native GPU support)
- macOS (limited GPU support)
- WSL (GPU passthrough support)
"""

import platform
import subprocess
import sys
import os
import logging
from typing import Dict, Tuple, Optional
from pathlib import Path

try:
    import tensorflow as tf

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

logger = logging.getLogger(__name__)


class PlatformDetector:
    """Detect current platform and WSL environment."""

    @staticmethod
    def get_platform_info() -> Dict:
        """Get comprehensive platform information."""
        info = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "is_wsl": False,
            "wsl_version": None,
            "is_windows": platform.system() == "Windows",
            "is_linux": platform.system() == "Linux",
            "is_macos": platform.system() == "Darwin",
        }

        # Check for WSL environment
        if info["is_linux"]:
            try:
                with open("/proc/version", "r") as f:
                    version_info = f.read().lower()
                    if "microsoft" in version_info or "wsl" in version_info:
                        info["is_wsl"] = True
                        # Extract WSL version if available
                        if "wsl2" in version_info:
                            info["wsl_version"] = 2
                        else:
                            info["wsl_version"] = 1
            except FileNotFoundError:
                pass

        return info


class WSLGpuDetector:
    """Specialized GPU detection for WSL environments."""

    @staticmethod
    def detect_wsl_gpu() -> Tuple[bool, str]:
        """Detect GPU availability in WSL environment."""
        try:
            # Check if nvidia-smi is available
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0:
                # Parse nvidia-smi output for basic info
                lines = result.stdout.split("\n")
                for line in lines:
                    if "NVIDIA" in line and "Driver" in line:
                        return True, f"WSL NVIDIA GPU detected: {line.strip()}"

                return True, "WSL NVIDIA GPU available"
            else:
                return False, "nvidia-smi not available in WSL"

        except FileNotFoundError:
            return False, "nvidia-smi not found"
        except subprocess.TimeoutExpired:
            return False, "nvidia-smi timeout"
        except Exception as e:
            return False, f"WSL GPU detection error: {e}"

    @staticmethod
    def check_wsl_cuda_support() -> bool:
        """Check CUDA toolkit installation in WSL."""
        try:
            # Check nvcc compiler
            result = subprocess.run(
                ["nvcc", "--version"], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
        except subprocess.TimeoutExpired:
            return False


class NativeGpuDetector:
    """GPU detection for native environments (Windows/Linux/macOS)."""

    @staticmethod
    def detect_tensorflow_gpus() -> Dict:
        """Detect TensorFlow GPU availability."""
        if not TENSORFLOW_AVAILABLE:
            return {
                "available": False,
                "count": 0,
                "devices": [],
                "error": "TensorFlow not available",
            }

        try:
            gpus = tf.config.list_physical_devices("GPU")

            gpu_info = {
                "available": len(gpus) > 0,
                "count": len(gpus),
                "devices": [str(gpu) for gpu in gpus],
                "cuda_available": tf.test.is_built_with_cuda()
                if hasattr(tf.test, "is_built_with_cuda")
                else False,
                "error": None,
            }

            # Try to get GPU details
            if gpu_info["available"]:
                try:
                    gpu_details = []
                    for gpu in gpus:
                        details = tf.config.experimental.get_device_details(gpu)
                        gpu_details.append(details)
                    gpu_info["device_details"] = gpu_details
                except Exception as e:
                    logger.warning(f"Failed to get GPU details: {e}")

            return gpu_info

        except Exception as e:
            return {"available": False, "count": 0, "devices": [], "error": str(e)}

    @staticmethod
    def check_opengl_support() -> bool:
        """Check OpenGL support (required for MediaPipe GPU delegate)."""
        try:
            if platform.system() == "Windows":
                # Windows OpenGL check
                try:
                    import OpenGL.GL as gl

                    try:
                        # Try to get OpenGL version
                        version = gl.glGetString(gl.GL_VERSION)
                        return version is not None
                    except:
                        return False
                except ImportError:
                    # PyOpenGL not available
                    return False

            elif platform.system() == "Linux":
                # Linux OpenGL check
                try:
                    # Try glxinfo
                    result = subprocess.run(
                        ["glxinfo", "-B"], capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0 and "OpenGL renderer" in result.stdout:
                        return True

                    # Fallback: check for OpenGL libraries
                    result = subprocess.run(
                        ["ldconfig", "-p"], capture_output=True, text=True, timeout=5
                    )
                    return "libGL" in result.stdout

                except (subprocess.TimeoutExpired, FileNotFoundError):
                    return False

            elif platform.system() == "Darwin":
                # macOS generally has OpenGL support
                return True

            return False

        except Exception as e:
            logger.warning(f"OpenGL detection failed: {e}")
            return False


class CrossPlatformGpuValidator:
    """Main class for cross-platform GPU validation."""

    def __init__(self):
        self.platform_info = PlatformDetector.get_platform_info()
        self.native_detector = NativeGpuDetector()
        self.wsl_detector = WSLGpuDetector()

    def get_comprehensive_gpu_info(self) -> Dict:
        """Get comprehensive GPU information for current platform."""
        info = {
            "platform_info": self.platform_info,
            "tensorflow_gpu": None,
            "opengl_support": False,
            "mediapipe_gpu_compatible": False,
            "mediapipe_gpu_reason": "",
            "recommended_configuration": {},
            "wsl_specific": {},
        }

        # TensorFlow GPU detection
        info["tensorflow_gpu"] = self.native_detector.detect_tensorflow_gpus()

        # OpenGL detection
        info["opengl_support"] = self.native_detector.check_opengl_support()

        # WSL-specific checks
        if self.platform_info["is_wsl"]:
            wsl_gpu_available, wsl_gpu_msg = self.wsl_detector.detect_wsl_gpu()
            info["wsl_specific"] = {
                "gpu_available": wsl_gpu_available,
                "message": wsl_gpu_msg,
                "cuda_support": self.wsl_detector.check_wsl_cuda_support(),
            }

        # MediaPipe GPU compatibility assessment
        info["mediapipe_gpu_compatible"], info["mediapipe_gpu_reason"] = (
            self._assess_mediapipe_gpu_compatibility(info)
        )

        # Recommended configuration
        info["recommended_configuration"] = self._get_recommended_configuration(info)

        return info

    def _assess_mediapipe_gpu_compatibility(self, info: Dict) -> Tuple[bool, str]:
        """Assess MediaPipe GPU delegate compatibility."""

        # Basic requirements
        if not MEDIAPIPE_AVAILABLE:
            return False, "MediaPipe not available"

        if not info["tensorflow_gpu"]["available"]:
            return False, "No TensorFlow GPUs available"

        if not info["opengl_support"]:
            return False, "OpenGL/EGL support not available"

        # Platform-specific considerations
        if self.platform_info["is_windows"] and not self.platform_info["is_wsl"]:
            return False, "Windows native GPU not supported (requires WSL)"

        if self.platform_info["is_macos"]:
            return False, "macOS GPU delegate not fully supported"

        # WSL-specific checks
        if self.platform_info["is_wsl"]:
            if not info["wsl_specific"].get("gpu_available", False):
                return False, "WSL GPU passthrough not available"

            if not info["wsl_specific"].get("cuda_support", False):
                return False, "WSL CUDA toolkit not installed"

        return True, "MediaPipe GPU delegate compatible"

    def _get_recommended_configuration(self, info: Dict) -> Dict:
        """Get recommended configuration based on platform and GPU availability."""
        config = {
            "mediapipe_delegate": "CPU",
            "auto_gpu_detection": True,
            "force_gpu": False,
            "use_wsl": False,
            "recommendations": [],
        }

        # WSL with GPU support
        if self.platform_info["is_wsl"] and info["wsl_specific"].get("gpu_available"):
            config["mediapipe_delegate"] = "GPU"
            config["auto_gpu_detection"] = True
            config["use_wsl"] = True
            config["recommendations"].append(
                "WSL GPU detected - enable GPU acceleration"
            )

        # Native Linux with GPU
        elif self.platform_info["is_linux"] and not self.platform_info["is_wsl"]:
            if info["mediapipe_gpu_compatible"]:
                config["mediapipe_delegate"] = "GPU"
                config["recommendations"].append(
                    "Native Linux GPU detected - enable GPU acceleration"
                )

        # Windows (no WSL)
        elif self.platform_info["is_windows"] and not self.platform_info["is_wsl"]:
            config["mediapipe_delegate"] = "CPU"
            config["recommendations"].append(
                "Windows detected - use WSL2 for GPU acceleration"
            )
            config["recommendations"].append(
                "Follow .maintenance/WSL_GPU_SETUP.md guide"
            )

        # macOS
        elif self.platform_info["is_macos"]:
            config["mediapipe_delegate"] = "CPU"
            config["recommendations"].append(
                "macOS detected - CPU fallback recommended"
            )

        # If no GPU available anywhere
        if not info["tensorflow_gpu"]["available"]:
            config["recommendations"].append("No GPU detected - CPU processing only")

        return config

    def print_diagnostics(self):
        """Print comprehensive diagnostics."""
        info = self.get_comprehensive_gpu_info()

        print("=== Cross-Platform GPU Diagnostics ===")
        print(f"Platform: {info['platform_info']['system']}")
        if info["platform_info"]["is_wsl"]:
            print(f"WSL Version: {info['platform_info']['wsl_version']}")

        print(f"TensorFlow GPUs: {info['tensorflow_gpu']['count']}")
        if info["tensorflow_gpu"]["available"]:
            for device in info["tensorflow_gpu"]["devices"]:
                print(f"  - {device}")

        print(f"OpenGL Support: {info['opengl_support']}")
        print(f"MediaPipe GPU Compatible: {info['mediapipe_gpu_compatible']}")
        print(f"Reason: {info['mediapipe_gpu_reason']}")

        if info["wsl_specific"]:
            print("WSL Specific:")
            print(f"  GPU Available: {info['wsl_specific']['gpu_available']}")
            print(f"  CUDA Support: {info['wsl_specific']['cuda_support']}")

        print("Recommendations:")
        for rec in info["recommended_configuration"]["recommendations"]:
            print(f"  - {rec}")

        print(
            f"Recommended Delegate: {info['recommended_configuration']['mediapipe_delegate']}"
        )
        print("=== End Diagnostics ===")


def main():
    """Run cross-platform GPU diagnostics."""
    validator = CrossPlatformGpuValidator()
    validator.print_diagnostics()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
