"""Base utilities and common functionality for vision detection methods."""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

from bp_face_recognition.vision.interfaces import FaceDetector

logger = logging.getLogger(__name__)


class BaseDetector(FaceDetector):
    """
    Base class for all face detectors with common utilities.
    """
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize base detector with common settings.
        
        Args:
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self._initialized = False
    
    def detect_with_confidence(
        self, image: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """
        Detect faces and return boxes with confidence scores.
        
        Args:
            image: Input image array
            
        Returns:
            List of tuples ((x, y, w, h), confidence)
        """
        boxes = self.detect(image)
        
        # Base implementation assumes 1.0 confidence for detectors that don't support it
        return [(box, 1.0) for box in boxes]
    
    def _filter_detections_by_confidence(
        self, 
        detections: List[Union[Tuple[int, int, int, int], Tuple[Tuple[int, int, int, int], float]]]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Filter detections based on confidence threshold.
        
        Args:
            detections: List of detection results
            
        Returns:
            Filtered list of bounding boxes
        """
        filtered_boxes = []
        
        for detection in detections:
            if isinstance(detection, tuple) and len(detection) == 2:
                box, confidence = detection
                if confidence >= self.confidence_threshold:
                    filtered_boxes.append(box)
            else:
                # Assume it's a box without confidence
                filtered_boxes.append(detection)
        
        return filtered_boxes
    
    def _validate_image(self, image: np.ndarray) -> bool:
        """
        Validate input image format.
        
        Args:
            image: Input image array
            
        Returns:
            True if valid, False otherwise
        """
        if image is None:
            logger.error("Input image is None")
            return False
        
        if not isinstance(image, np.ndarray):
            logger.error(f"Input must be numpy array, got {type(image)}")
            return False
        
        if len(image.shape) not in [2, 3]:
            logger.error(f"Invalid image shape: {image.shape}")
            return False
        
        if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
            logger.error(f"Invalid image channels: {image.shape[2]}")
            return False
        
        return True
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Common image preprocessing.
        
        Args:
            image: Input image array
            
        Returns:
            Preprocessed image array
        """
        # Default implementation - return as-is
        # Subclasses can override for specific preprocessing
        return image.copy()
    
    def _log_detection_stats(
        self, 
        detections: List[Union[Tuple[int, int, int, int], Tuple[Tuple[int, int, int, int], float]]],
        image_shape: Tuple[int, ...]
    ) -> None:
        """
        Log detection statistics for debugging.
        
        Args:
            detections: List of detection results
            image_shape: Shape of input image
        """
        logger.debug(f"Input image shape: {image_shape}")
        logger.debug(f"Total detections before filtering: {len(detections)}")
        
        confident_detections = [
            d for d in detections 
            if isinstance(d, tuple) and len(d) == 2 and d[1] >= self.confidence_threshold
        ]
        
        logger.debug(f"Detections above threshold {self.confidence_threshold}: {len(confident_detections)}")
    
    def _handle_detection_error(self, error: Exception, context: str) -> List[Tuple[int, int, int, int]]:
        """
        Handle detection errors gracefully.
        
        Args:
            error: Exception that occurred
            context: Context where error occurred
            
        Returns:
            Empty list of detections
        """
        logger.error(f"Detection error in {context}: {error}")
        return []
    
    def get_detector_info(self) -> Dict[str, Any]:
        """
        Get information about the detector.
        
        Returns:
            Dictionary with detector metadata
        """
        return {
            "name": self.__class__.__name__,
            "confidence_threshold": self.confidence_threshold,
            "initialized": self._initialized,
            "type": "face_detector"
        }


class DetectionResult:
    """
    Standardized detection result container.
    """
    
    def __init__(
        self,
        boxes: List[Tuple[int, int, int, int]],
        confidences: Optional[List[float]] = None,
        landmarks: Optional[List[List[Tuple[int, int]]]] = None,
        processing_time: Optional[float] = None
    ):
        """
        Initialize detection result.
        
        Args:
            boxes: List of bounding boxes (x, y, w, h)
            confidences: Optional confidence scores
            landmarks: Optional facial landmarks
            processing_time: Optional processing time in seconds
        """
        self.boxes = boxes
        self.confidences = confidences or [1.0] * len(boxes)
        self.landmarks = landmarks
        self.processing_time = processing_time
        self.num_faces = len(boxes)
    
    def get_boxes_with_confidence(self) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Get boxes with their confidence scores."""
        return list(zip(self.boxes, self.confidences))
    
    def get_average_confidence(self) -> float:
        """Get average confidence of all detections."""
        return float(np.mean(self.confidences)) if self.confidences else 0.0
    
    def filter_by_confidence(self, threshold: float) -> 'DetectionResult':
        """Filter detections by confidence threshold."""
        if not self.confidences:
            return self
        
        valid_indices = [i for i, conf in enumerate(self.confidences) if conf >= threshold]
        
        return DetectionResult(
            boxes=[self.boxes[i] for i in valid_indices],
            confidences=[self.confidences[i] for i in valid_indices],
            landmarks=[self.landmarks[i] for i in valid_indices] if self.landmarks else None,
            processing_time=self.processing_time
        )


def create_detection_result_from_legacy(
    detections: List[Union[Tuple[int, int, int, int], Tuple[Tuple[int, int, int, int], float]]],
    processing_time: Optional[float] = None
) -> DetectionResult:
    """
    Create standardized DetectionResult from legacy detection format.
    
    Args:
        detections: Legacy detection results
        processing_time: Optional processing time
        
    Returns:
        Standardized DetectionResult
    """
    boxes = []
    confidences = []
    
    for detection in detections:
        if isinstance(detection, tuple) and len(detection) == 2:
            box, confidence = detection
            boxes.append(box)
            confidences.append(confidence)
        else:
            boxes.append(detection)
            confidences.append(1.0)
    
    return DetectionResult(
        boxes=boxes,
        confidences=confidences,
        processing_time=processing_time
    )