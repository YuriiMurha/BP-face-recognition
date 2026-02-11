"""
Pipeline Service - End-to-End Workflow Orchestration

This service provides high-level orchestration for complete
face recognition pipelines including detection, recognition, and database operations.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import time

from bp_face_recognition.vision.core.face_tracker import FaceTracker
from bp_face_recognition.services.database_service import DatabaseService
from bp_face_recognition.vision.factory import RecognizerFactory

logger = logging.getLogger(__name__)


class PipelineService:
    """
    High-level service for orchestrating complete face recognition workflows.

    Provides end-to-end functionality including detection, recognition,
    database operations, and performance monitoring.
    """

    def __init__(
        self,
        detector_type: str = "mediapipe_v1",
        recognizer_type: str = "custom_cnn_v1",
        database_service: Optional[DatabaseService] = None,
        recognition_threshold: float = 0.7,
        **kwargs: Any,
    ):
        """
        Initialize pipeline service.

        Args:
            detector_type: Type of detector to use
            recognizer_type: Type of recognizer to use
            database_service: Optional pre-configured database service
            recognition_threshold: Face recognition confidence threshold
            **kwargs: Additional configuration options
        """
        self.detector_type = detector_type
        self.recognizer_type = recognizer_type
        self.recognition_threshold = recognition_threshold

        # Initialize components
        self.face_tracker = FaceTracker(
            detector_type=detector_type, recognizer_type=recognizer_type, **kwargs
        )

        if database_service is None:
            self.database_service = DatabaseService()
        else:
            self.database_service = database_service

        # Performance tracking
        self.processing_stats = {
            "total_images": 0,
            "total_faces_detected": 0,
            "total_faces_recognized": 0,
            "total_processing_time": 0.0,
            "errors": [],
        }

        logger.info(
            f"PipelineService initialized with {detector_type} + {recognizer_type}"
        )

    def process_image(
        self,
        image: np.ndarray,
        register_unknown: bool = False,
        update_database: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a single image through the complete pipeline.

        Args:
            image: Input image array
            register_unknown: Whether to register unknown faces
            update_database: Whether to update database with recognized faces

        Returns:
            Complete pipeline processing results
        """
        start_time = time.time()

        try:
            # Step 1: Detect and recognize faces
            tracking_result = self.face_tracker.track_faces(image, update_history=True)

            # Step 2: Recognize detected faces against database
            recognized_faces = []
            for i, face_info in enumerate(tracking_result["faces"]):
                if face_info["embedding"] is None or len(face_info["embedding"]) == 0:
                    # Invalid embedding, skip
                    recognized_faces.append(
                        {
                            **face_info,
                            "identity": "Unknown",
                            "recognized": False,
                            "recognition_confidence": 0.0,
                            "error": "Invalid embedding",
                        }
                    )
                    continue

                # Use database service for recognition
                recognition_result = self.database_service.recognize_face(
                    face_info["embedding"], self.recognition_threshold
                )

                # Merge results
                recognized_face = {
                    **face_info,
                    "identity": recognition_result["identity"],
                    "recognized": recognition_result["recognized"],
                    "recognition_confidence": recognition_result["confidence"],
                    "threshold_used": recognition_result.get(
                        "threshold_used", self.recognition_threshold
                    ),
                }
                recognized_faces.append(recognized_face)

                # Step 3: Update database if needed
                if update_database and recognition_result["recognized"]:
                    self._update_face_record(recognized_face, image.shape)

            # Update processing stats
            processing_time = time.time() - start_time
            self._update_processing_stats(
                tracking_result["num_faces"],
                len([f for f in recognized_faces if f["recognized"]]),
                processing_time,
            )

            return {
                "success": True,
                "detection_result": tracking_result,
                "recognition_result": {
                    "faces": recognized_faces,
                    "num_faces": len(recognized_faces),
                    "num_recognized": len(
                        [f for f in recognized_faces if f["recognized"]]
                    ),
                },
                "processing_time": processing_time,
                "image_shape": image.shape,
                "pipeline_stats": self.get_processing_stats(),
            }

        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            self.processing_stats["errors"].append(str(e))

            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "pipeline_stats": self.get_processing_stats(),
            }

    def process_batch(
        self,
        images: List[np.ndarray],
        register_unknown: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a batch of images through the pipeline.

        Args:
            images: List of image arrays
            register_unknown: Whether to register unknown faces
            show_progress: Whether to show progress updates

        Returns:
            Batch processing results
        """
        start_time = time.time()
        results = []

        logger.info(f"Processing batch of {len(images)} images")

        for i, image in enumerate(images):
            if show_progress and i % 10 == 0:
                logger.info(f"Processing image {i+1}/{len(images)}")

            result = self.process_image(image, register_unknown=register_unknown)
            results.append(result)

        total_time = time.time() - start_time

        # Batch statistics
        successful_results = [r for r in results if r["success"]]
        total_faces_detected = sum(
            r["recognition_result"]["num_faces"] for r in successful_results
        )
        total_faces_recognized = sum(
            r["recognition_result"]["num_recognized"] for r in successful_results
        )

        batch_result = {
            "success": True,
            "total_images": len(images),
            "successful_images": len(successful_results),
            "total_faces_detected": total_faces_detected,
            "total_faces_recognized": total_faces_recognized,
            "recognition_rate": total_faces_recognized / max(total_faces_detected, 1),
            "total_processing_time": total_time,
            "avg_processing_time": total_time / len(images),
            "individual_results": results,
            "pipeline_stats": self.get_processing_stats(),
        }

        logger.info(
            f"Batch completed: {total_faces_recognized}/{total_faces_detected} faces recognized "
            f"({batch_result['recognition_rate']:.2%})"
        )

        return batch_result

    def register_person(
        self,
        name: str,
        images: List[np.ndarray],
        metadata: Optional[Dict[str, Any]] = None,
        min_embeddings: int = 5,
    ) -> Dict[str, Any]:
        """
        Register a new person with multiple face images.

        Args:
            name: Person's name/identity
            images: List of face images
            metadata: Optional additional information
            min_embeddings: Minimum number of valid embeddings required

        Returns:
            Registration result with statistics
        """
        logger.info(f"Registering person: {name} with {len(images)} images")

        # Process each image to extract embeddings
        valid_embeddings = []
        face_detections = []

        for i, image in enumerate(images):
            tracking_result = self.face_tracker.track_faces(image, update_history=False)

            if tracking_result["num_faces"] > 0:
                embedding = (
                    tracking_result["embeddings"][0]
                    if tracking_result["embeddings"]
                    else None
                )
                if embedding is not None and len(embedding) > 0:
                    valid_embeddings.append(embedding)

                face_detections.append(
                    {
                        "image_index": i,
                        "faces_found": tracking_result["num_faces"],
                        "embedding_valid": embedding is not None,
                        "detection_confidence": tracking_result["detection_confidence"][
                            0
                        ]
                        if tracking_result["detection_confidence"]
                        else 0.0,
                    }
                )
            else:
                face_detections.append(
                    {
                        "image_index": i,
                        "faces_found": 0,
                        "embedding_valid": False,
                        "detection_confidence": 0.0,
                    }
                )

        # Check if we have enough embeddings
        if len(valid_embeddings) < min_embeddings:
            logger.warning(
                f"Only {len(valid_embeddings)} valid embeddings extracted "
                f"(minimum {min_embeddings} required)"
            )

        # Register with database service
        registration_success = self.database_service.register_person(
            name, valid_embeddings, metadata
        )

        return {
            "success": registration_success,
            "person_name": name,
            "total_images": len(images),
            "valid_embeddings": len(valid_embeddings),
            "required_embeddings": min_embeddings,
            "face_detections": face_detections,
            "metadata": metadata or {},
        }

    def _update_face_record(
        self, recognized_face: Dict[str, Any], image_shape: Tuple[int, ...]
    ) -> None:
        """
        Update face record in database (internal method).

        Args:
            recognized_face: Recognized face information
            image_shape: Shape of the original image
        """
        if (
            recognized_face["recognized"]
            and recognized_face["recognition_confidence"] > 0.8
        ):
            # High confidence recognition - could update last seen timestamp
            person_name = recognized_face["identity"]
            metadata = {
                "last_seen": time.time(),
                "last_confidence": recognized_face["recognition_confidence"],
                "image_shape": image_shape,
            }

            self.database_service.update_person_metadata(person_name, metadata)

    def _update_processing_stats(
        self, faces_detected: int, faces_recognized: int, processing_time: float
    ) -> None:
        """
        Update internal processing statistics.

        Args:
            faces_detected: Number of faces detected
            faces_recognized: Number of faces successfully recognized
            processing_time: Processing time in seconds
        """
        self.processing_stats["total_images"] += 1
        self.processing_stats["total_faces_detected"] += faces_detected
        self.processing_stats["total_faces_recognized"] += faces_recognized
        self.processing_stats["total_processing_time"] += processing_time

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get current processing statistics.

        Returns:
            Dictionary with processing statistics
        """
        stats = self.processing_stats.copy()

        # Calculate derived statistics
        if stats["total_images"] > 0:
            stats["avg_faces_per_image"] = (
                stats["total_faces_detected"] / stats["total_images"]
            )
            stats["recognition_success_rate"] = stats["total_faces_recognized"] / max(
                stats["total_faces_detected"], 1
            )
            stats["avg_processing_time"] = (
                stats["total_processing_time"] / stats["total_images"]
            )
        else:
            stats["avg_faces_per_image"] = 0.0
            stats["recognition_success_rate"] = 0.0
            stats["avg_processing_time"] = 0.0

        # Add system info
        stats["detector_type"] = self.detector_type
        stats["recognizer_type"] = self.recognizer_type
        stats["recognition_threshold"] = self.recognition_threshold
        stats["error_count"] = len(stats["errors"])

        return stats

    def reset_processing_stats(self) -> None:
        """Reset processing statistics."""
        self.processing_stats = {
            "total_images": 0,
            "total_faces_detected": 0,
            "total_faces_recognized": 0,
            "total_processing_time": 0.0,
            "errors": [],
        }
        logger.info("Processing statistics reset")

    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information.

        Returns:
            Dictionary with system information
        """
        return {
            "pipeline_service": {
                "detector_type": self.detector_type,
                "recognizer_type": self.recognizer_type,
                "recognition_threshold": self.recognition_threshold,
            },
            "face_tracker": self.face_tracker.get_system_info(),
            "database_service": self.database_service.get_database_stats(),
            "processing_stats": self.get_processing_stats(),
        }
