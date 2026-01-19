from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple


class FaceDetector(ABC):
    """
    Abstract base class for all face detection methods.
    Enforces a common interface for easy swapping of algorithms.
    """

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in the given image.

        Args:
            image (np.ndarray): The input image (BGR or RGB depending on implementation).

        Returns:
            List[Tuple[int, int, int, int]]: A list of bounding boxes (x, y, w, h).
        """
        pass

    def detect_with_confidence(
        self, image: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Detect faces and return boxes with confidence scores.
        Defaults to 1.0 confidence if not supported by the algorithm.
        """
        boxes = self.detect(image)
        return [(box, 1.0) for box in boxes]


class FaceRecognizer(ABC):
    """
    Abstract base class for face recognition/embedding extraction.
    """

    @abstractmethod
    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract embedding vector from a cropped face image.

        Args:
            face_image (np.ndarray): The cropped face image.

        Returns:
            np.ndarray: The embedding vector.
        """
        pass
