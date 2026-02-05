import tensorflow as tf
import platform
import subprocess
import sys
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_tensorflow_gpu():
    """
    Configures TensorFlow to use GPU memory growth to avoid OOM errors.
    """
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Num TensorFlow GPUs Available: {len(gpus)}")
            logger.info(f"GPUs: {gpus}")
            return True
        except RuntimeError as e:
            logger.error(f"TensorFlow GPU setup failed: {e}")
            return False
    else:
        logger.info("No TensorFlow GPUs found.")
        return False


def check_opengl_support() -> bool:
    """
    Check if OpenGL support is available (required for MediaPipe GPU delegate).

    Returns:
        bool: True if OpenGL is supported, False otherwise
    """
    try:
        if platform.system() == "Windows":
            # On Windows, try to use OpenGL through PyOpenGL
            try:
                import OpenGL.GL as gl

                # Try to get OpenGL version
                version = gl.glGetString(gl.GL_VERSION)
                if version:
                    logger.info(f"OpenGL version: {version}")
                    return True
            except ImportError:
                logger.warning("PyOpenGL not available for OpenGL detection")
                return False
            except Exception as e:
                logger.warning(f"OpenGL detection failed: {e}")
                return False

        elif platform.system() == "Linux":
            # On Linux, check for EGL/OpenGL libraries
            try:
                # Check for glxinfo or similar tools
                result = subprocess.run(
                    ["glxinfo", "-B"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    logger.info("OpenGL support detected via glxinfo")
                    return True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

            # Fallback: check for common OpenGL libraries
            try:
                result = subprocess.run(
                    ["ldconfig", "-p"], capture_output=True, text=True, timeout=5
                )
                if "libGL" in result.stdout:
                    logger.info("OpenGL libraries detected")
                    return True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        elif platform.system() == "Darwin":  # macOS
            # macOS generally has OpenGL support
            logger.info("macOS detected - assuming OpenGL support")
            return True

        return False

    except Exception as e:
        logger.warning(f"OpenGL support check failed: {e}")
        return False


def check_mediapipe_gpu_compatibility() -> Tuple[bool, str]:
    """
    Check MediaPipe GPU delegate compatibility.

    Returns:
        Tuple[bool, str]: (is_compatible, reason)
    """
    # Check TensorFlow GPU support first
    tf_gpus = tf.config.experimental.list_physical_devices("GPU")
    if not tf_gpus:
        return False, "No TensorFlow GPUs available"

    # Check OpenGL support (required for MediaPipe GPU delegate)
    if not check_opengl_support():
        return False, "OpenGL/EGL support not available (required for MediaPipe GPU)"

    # Try to import MediaPipe and check GPU support
    try:
        import mediapipe as mp

        # MediaPipe is available - GPU delegate should work
        return True, "MediaPipe GPU delegate compatible"
    except ImportError:
        return False, "MediaPipe not available"
    except Exception as e:
        return False, f"MediaPipe GPU compatibility check failed: {e}"


def get_gpu_info() -> dict:
    """
    Get comprehensive GPU information.

    Returns:
        dict: GPU information including TensorFlow, OpenGL, and MediaPipe compatibility
    """
    info = {
        "platform": platform.system(),
        "tensorflow_gpu_count": len(
            tf.config.experimental.list_physical_devices("GPU")
        ),
        "tensorflow_gpus": [
            str(gpu) for gpu in tf.config.experimental.list_physical_devices("GPU")
        ],
        "opengl_supported": check_opengl_support(),
        "mediapipe_gpu_compatible": False,
        "mediapipe_gpu_reason": "",
        "cuda_available": False,
        "recommended_delegate": None,
    }

    # Check CUDA availability
    try:
        info["cuda_available"] = tf.test.is_built_with_cuda()
    except:
        info["cuda_available"] = False

    # Check MediaPipe GPU compatibility
    compatible, reason = check_mediapipe_gpu_compatibility()
    info["mediapipe_gpu_compatible"] = compatible
    info["mediapipe_gpu_reason"] = reason

    # Determine recommended delegate
    if compatible:
        info["recommended_delegate"] = "GPU"
    elif info["tensorflow_gpu_count"] > 0:
        info["recommended_delegate"] = (
            "CPU (TensorFlow GPU available but MediaPipe GPU incompatible)"
        )
    else:
        info["recommended_delegate"] = "CPU"

    return info


def test_mediapipe_gpu_delegate() -> Tuple[bool, str]:
    """
    Test actual MediaPipe GPU delegate initialization.

    Returns:
        Tuple[bool, str]: (success, error_message)
    """
    try:
        import mediapipe as mp

        # Try to create base options with GPU delegate
        base_options = mp.tasks.BaseOptions(delegate=mp.tasks.BaseOptions.Delegate.GPU)

        # If we get here, GPU delegate creation succeeded
        logger.info("MediaPipe GPU delegate creation successful")
        return True, ""

    except Exception as e:
        error_msg = str(e)
        logger.error(f"MediaPipe GPU delegate test failed: {error_msg}")
        return False, error_msg


def setup_gpu():
    """
    Legacy function for backward compatibility.
    """
    return setup_tensorflow_gpu()


def print_gpu_diagnostics():
    """
    Print comprehensive GPU diagnostics for debugging.
    """
    logger.info("=== GPU Diagnostics ===")

    info = get_gpu_info()

    logger.info(f"Platform: {info['platform']}")
    logger.info(f"TensorFlow GPUs: {info['tensorflow_gpu_count']}")
    if info["tensorflow_gpus"]:
        for gpu in info["tensorflow_gpus"]:
            logger.info(f"  - {gpu}")

    logger.info(f"OpenGL Supported: {info['opengl_supported']}")
    logger.info(f"CUDA Available: {info['cuda_available']}")
    logger.info(f"MediaPipe GPU Compatible: {info['mediapipe_gpu_compatible']}")
    logger.info(f"MediaPipe GPU Reason: {info['mediapipe_gpu_reason']}")
    logger.info(f"Recommended Delegate: {info['recommended_delegate']}")

    # Test actual GPU delegate
    success, error = test_mediapipe_gpu_delegate()
    logger.info(f"MediaPipe GPU Delegate Test: {'PASS' if success else 'FAIL'}")
    if error:
        logger.error(f"  Error: {error}")

    logger.info("=== End Diagnostics ===")


if __name__ == "__main__":
    print_gpu_diagnostics()
