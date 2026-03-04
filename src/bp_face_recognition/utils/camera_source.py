"""
Camera Source Module - Multi-source camera configuration

Supports:
- Local webcam (device index)
- RTSP stream URLs
- USB-connected Android device via ADB
"""

import logging
import subprocess
import re
from abc import ABC, abstractmethod
from typing import Optional, List
from dataclasses import dataclass

import cv2
import numpy as np

from bp_face_recognition.config.settings import settings, CameraSourceType

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    source_type: str
    device_index: Optional[int] = None
    rtsp_url: Optional[str] = None
    width: int = 1280
    height: int = 720
    fps: int = 30


class CameraSource(ABC):
    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def read_frame(self) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def release(self) -> None:
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        pass


class WebcamSource(CameraSource):
    def __init__(
        self, device_index: int = 0, width: int = 1280, height: int = 720, fps: int = 30
    ):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None

    def connect(self) -> bool:
        backends = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF]

        for backend in backends:
            self.cap = cv2.VideoCapture(self.device_index, backend)
            if self.cap.isOpened():
                actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                logger.info(
                    f"Webcam connected (backend: {backend}): {actual_width}x{actual_height}"
                )
                return True
            if self.cap is not None:
                self.cap.release()
                self.cap = None

        logger.error(
            f"Failed to open webcam device {self.device_index} with any backend"
        )
        return False

    def read_frame(self) -> Optional[np.ndarray]:
        if self.cap is None or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()

        if not ret:
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def is_connected(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def read_frame(self) -> Optional[np.ndarray]:
        if self.cap is None or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def is_connected(self) -> bool:
        return self.cap is not None and self.cap.isOpened()


class RTSPSource(CameraSource):
    def __init__(
        self, rtsp_url: str, width: int = 1280, height: int = 720, fps: int = 30
    ):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5

    def connect(self) -> bool:
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        if not self.cap.isOpened():
            logger.error(f"Failed to open RTSP stream: {self.rtsp_url}")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        logger.info(f"RTSP stream connected: {self.rtsp_url}")
        self.reconnect_attempts = 0
        return True

    def _try_reconnect(self) -> bool:
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            return False

        self.reconnect_attempts += 1
        logger.info(
            f"Attempting reconnection {self.reconnect_attempts}/{self.max_reconnect_attempts}"
        )
        return self.connect()

    def read_frame(self) -> Optional[np.ndarray]:
        if self.cap is None:
            if not self.connect():
                return None

        assert self.cap is not None
        ret, frame = self.cap.read()
        if not ret:
            logger.warning(
                "Failed to read frame from RTSP stream, attempting reconnection"
            )
            if self._try_reconnect():
                assert self.cap is not None
                ret, frame = self.cap.read()
                if not ret:
                    return None
            else:
                return None

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def is_connected(self) -> bool:
        return self.cap is not None and self.cap.isOpened()


class USBDeviceSource(CameraSource):
    ADB_DEVICES_PATTERN = re.compile(r"(\S+)\s+device\b")

    def __init__(
        self,
        device_serial: Optional[str] = None,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ):
        self.device_serial = device_serial
        self.width = width
        self.height = height
        self.fps = fps
        self.socat_process: Optional[subprocess.Popen] = None
        self.virtual_device_index: Optional[int] = None
        self._available_devices: Optional[List[int]] = None
        self.cap: Optional[cv2.VideoCapture] = None

    @staticmethod
    def list_adb_devices() -> List[str]:
        try:
            result = subprocess.run(
                ["adb", "devices"], capture_output=True, text=True, timeout=10
            )
            devices = []
            for line in result.stdout.splitlines()[1:]:
                match = USBDeviceSource.ADB_DEVICES_PATTERN.search(line)
                if match:
                    devices.append(match.group(1))
            return devices
        except FileNotFoundError:
            logger.error(
                "ADB not found. Make sure Android SDK platform-tools are in PATH"
            )
            return []
        except subprocess.TimeoutExpired:
            logger.error("ADB devices command timed out")
            return []

    @staticmethod
    def get_available_video_devices() -> List[int]:
        devices = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                devices.append(i)
                cap.release()
        return devices

    def connect(self) -> bool:
        if self.device_serial:
            return self._connect_with_adb()
        else:
            return self._connect_to_existing_device()

    def _connect_with_adb(self) -> bool:
        devices = self.list_adb_devices()
        if not devices:
            logger.error("No ADB devices found")
            return False

        if self.device_serial not in devices:
            logger.error(f"Device {self.device_serial} not found in ADB devices")
            return False

        adb_command = [
            "adb",
            "-s",
            self.device_serial,
            "shell",
            "screenrecord",
            "--size",
            f"{self.width}x{self.height}",
            "--bit-rate",
            "4000000",
            "/sdcard/screen.mp4",
        ]

        try:
            subprocess.Popen(
                adb_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

            forward_command = [
                "adb",
                "-s",
                self.device_serial,
                "reverse",
                "tcp:8554",
                "tcp:8554",
            ]
            subprocess.run(forward_command, check=True, timeout=5)

            self.socat_process = subprocess.Popen(
                ["socat", "-", "TCP:localhost:8554"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            self._available_devices = self.get_available_video_devices()
            if self._available_devices:
                self.virtual_device_index = self._available_devices[0]
                self.cap = cv2.VideoCapture(self.virtual_device_index)
                logger.info(f"Connected to USB device {self.device_serial} via ADB")
                return True

        except FileNotFoundError:
            logger.error("Required tools (adb, socat) not found in PATH")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to USB device: {e}")
            return False

        return False

    def _connect_to_existing_device(self) -> bool:
        self._available_devices = self.get_available_video_devices()

        if not self._available_devices:
            logger.error("No available video devices found")
            return False

        for idx in self._available_devices:
            self.virtual_device_index = idx
            self.cap = cv2.VideoCapture(idx)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            if self.cap.isOpened():
                logger.info(f"Connected to USB device at index {idx}")
                return True

        logger.error("No USB camera device found")
        return False

        for idx in self._available_devices:
            if idx != 0:
                self.virtual_device_index = idx
                self.cap = cv2.VideoCapture(idx)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)

                if self.cap.isOpened():
                    logger.info(f"Connected to USB device at index {idx}")
                    return True

        logger.error("No USB camera device found")
        return False

    def read_frame(self) -> Optional[np.ndarray]:
        if self.cap is None or not self.cap.isOpened():
            if not self.connect():
                return None

        assert self.cap is not None
        ret, frame = self.cap.read()
        if not ret:
            return None

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.socat_process is not None:
            self.socat_process.terminate()
            self.socat_process = None

    def is_connected(self) -> bool:
        return self.cap is not None and self.cap.isOpened()


class CameraManager:
    def __init__(self, config: Optional[CameraConfig] = None):
        if config is None:
            config = self._load_config_from_settings()

        self.config = config
        self.source: Optional[CameraSource] = None
        self._connect()

    def _load_config_from_settings(self) -> CameraConfig:
        source_type = settings.CAMERA_SOURCE.lower()

        if source_type == CameraSourceType.RTSP:
            rtsp_url = settings.CAMERA_RTSP_URL
            if not rtsp_url:
                raise ValueError(
                    "RTSP URL not configured. Set CAMERA_RTSP_URL environment variable"
                )
            return CameraConfig(
                source_type=source_type,
                rtsp_url=rtsp_url,
                width=settings.CAMERA_WIDTH,
                height=settings.CAMERA_HEIGHT,
                fps=settings.CAMERA_FPS,
            )
        elif source_type == CameraSourceType.USB:
            return CameraConfig(
                source_type=source_type,
                width=settings.CAMERA_WIDTH,
                height=settings.CAMERA_HEIGHT,
                fps=settings.CAMERA_FPS,
            )
        else:
            return CameraConfig(
                source_type=CameraSourceType.WEBCAM,
                device_index=settings.CAMERA_DEVICE,
                width=settings.CAMERA_WIDTH,
                height=settings.CAMERA_HEIGHT,
                fps=settings.CAMERA_FPS,
            )

    def _connect(self) -> None:
        if self.config.source_type == CameraSourceType.RTSP:
            rtsp_url = self.config.rtsp_url or ""
            self.source = RTSPSource(
                rtsp_url,
                self.config.width,
                self.config.height,
                self.config.fps,
            )
            if not self.source.connect():
                raise RuntimeError(f"Failed to connect to RTSP stream: {rtsp_url}")
        else:
            device_index = (
                self.config.device_index if self.config.device_index is not None else 0
            )
            self.source = WebcamSource(
                device_index,
                self.config.width,
                self.config.height,
                self.config.fps,
            )

        if not self.source.connect():
            raise RuntimeError(
                f"Failed to connect to camera source: {self.config.source_type}"
            )

        logger.info(f"Camera source connected: {self.config.source_type}")

    def read_frame(self) -> Optional[np.ndarray]:
        if self.source is None:
            return None
        return self.source.read_frame()

    def switch_source(self, new_config: CameraConfig) -> bool:
        self.release()
        self.config = new_config
        self._connect()
        return self.source is not None and self.source.is_connected()

    def release(self) -> None:
        if self.source is not None:
            self.source.release()
            self.source = None

    def is_connected(self) -> bool:
        return self.source is not None and self.source.is_connected()


def create_camera_manager(
    source_type: Optional[str] = None,
    device_index: Optional[int] = None,
    rtsp_url: Optional[str] = None,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
) -> CameraManager:
    if source_type is None:
        source_type = settings.CAMERA_SOURCE

    config = CameraConfig(
        source_type=source_type,
        device_index=device_index,
        rtsp_url=rtsp_url,
        width=width,
        height=height,
        fps=fps,
    )

    return CameraManager(config)
