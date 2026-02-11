"""
Database Service - Clean Database Abstraction

This service provides a high-level interface for database operations,
abstracting away the complexity of different database backends.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

from bp_face_recognition.database.database import FaceDatabase

logger = logging.getLogger(__name__)


class DatabaseService:
    """
    High-level service for database operations.

    Provides clean interface for face registration, embedding storage,
    and identity retrieval with error handling and logging.
    """

    def __init__(self, database: Optional[FaceDatabase] = None):
        """
        Initialize database service.

        Args:
            database: Optional pre-configured database instance
        """
        if database is None:
            self.database = FaceDatabase()
        else:
            self.database = database

        logger.info("DatabaseService initialized")

    def register_person(
        self,
        name: str,
        embeddings: List[np.ndarray],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Register a new person with their face embeddings.

        Args:
            name: Person's name/identity
            embeddings: List of face embedding vectors
            metadata: Optional additional information about the person

        Returns:
            True if registration successful, False otherwise
        """
        try:
            if not embeddings:
                logger.error(f"No embeddings provided for person: {name}")
                return False

            # Register each embedding
            success_count = 0
            for embedding in embeddings:
                try:
                    success = self.database.add_embedding(
                        name, embedding, metadata or {}
                    )
                    if success:
                        success_count += 1
                except Exception as e:
                    logger.error(f"Failed to add embedding for {name}: {e}")

            # Consider successful if at least one embedding registered
            overall_success = success_count > 0

            if overall_success:
                logger.info(
                    f"Successfully registered {name} with {success_count}/{len(embeddings)} embeddings"
                )
            else:
                logger.error(f"Failed to register {name} - all embeddings failed")

            return overall_success

        except Exception as e:
            logger.error(f"Person registration failed for {name}: {e}")
            return False

    def recognize_face(
        self, embedding: np.ndarray, threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Identify a face by comparing with stored embeddings.

        Args:
            embedding: Face embedding vector to identify
            threshold: Minimum similarity threshold for recognition

        Returns:
            Recognition result with identity and confidence
        """
        try:
            if embedding is None or len(embedding) == 0:
                return {
                    "recognized": False,
                    "identity": "Unknown",
                    "confidence": 0.0,
                    "error": "Invalid embedding",
                }

            # Get all known embeddings
            known_embeddings = self.database.get_all_embeddings()

            if not known_embeddings:
                return {
                    "recognized": False,
                    "identity": "Unknown",
                    "confidence": 0.0,
                    "error": "No registered faces",
                }

            # Find best match
            best_match = None
            best_confidence = 0.0

            for identity, embeddings_list in known_embeddings.items():
                for known_embedding in embeddings_list:
                    if len(known_embedding) != len(embedding):
                        continue

                    # Calculate cosine similarity
                    similarity = np.dot(embedding, known_embedding) / (
                        np.linalg.norm(embedding) * np.linalg.norm(known_embedding)
                    )

                    if similarity > best_confidence:
                        best_confidence = similarity
                        best_match = identity

            # Determine recognition result
            if best_confidence >= threshold:
                return {
                    "recognized": True,
                    "identity": best_match,
                    "confidence": float(best_confidence),
                    "threshold_used": threshold,
                }
            else:
                return {
                    "recognized": False,
                    "identity": "Unknown",
                    "confidence": float(best_confidence),
                    "threshold_used": threshold,
                }

        except Exception as e:
            logger.error(f"Face recognition failed: {e}")
            return {
                "recognized": False,
                "identity": "Unknown",
                "confidence": 0.0,
                "error": str(e),
            }

    def get_person_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a registered person.

        Args:
            name: Person's name/identity

        Returns:
            Person information dictionary or None if not found
        """
        try:
            return self.database.get_person_info(name)
        except Exception as e:
            logger.error(f"Failed to get person info for {name}: {e}")
            return None

    def update_person_metadata(self, name: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a registered person.

        Args:
            name: Person's name/identity
            metadata: New metadata dictionary

        Returns:
            True if update successful, False otherwise
        """
        try:
            success = self.database.update_metadata(name, metadata)
            if success:
                logger.info(f"Updated metadata for {name}")
            else:
                logger.warning(f"Failed to update metadata for {name}")
            return success
        except Exception as e:
            logger.error(f"Failed to update metadata for {name}: {e}")
            return False

    def delete_person(self, name: str) -> bool:
        """
        Delete a person from the database.

        Args:
            name: Person's name/identity

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            success = self.database.delete_person(name)
            if success:
                logger.info(f"Deleted person: {name}")
            else:
                logger.warning(f"Failed to delete person: {name}")
            return success
        except Exception as e:
            logger.error(f"Failed to delete person {name}: {e}")
            return False

    def list_all_people(self) -> List[str]:
        """
        Get list of all registered people.

        Returns:
            List of person names/identities
        """
        try:
            return self.database.list_all_people()
        except Exception as e:
            logger.error(f"Failed to list people: {e}")
            return []

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with database statistics
        """
        try:
            stats = self.database.get_stats()

            # Add service-level statistics
            stats.update(
                {"service_initialized": True, "service_type": "DatabaseService"}
            )

            return stats
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {"total_people": 0, "total_embeddings": 0, "error": str(e)}

    def backup_database(self, backup_path: str) -> bool:
        """
        Create backup of the database.

        Args:
            backup_path: Path where backup should be saved

        Returns:
            True if backup successful, False otherwise
        """
        try:
            success = self.database.backup(backup_path)
            if success:
                logger.info(f"Database backed up to: {backup_path}")
            else:
                logger.warning(f"Database backup failed: {backup_path}")
            return success
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return False

    def restore_database(self, backup_path: str) -> bool:
        """
        Restore database from backup.

        Args:
            backup_path: Path to backup file

        Returns:
            True if restore successful, False otherwise
        """
        try:
            success = self.database.restore(backup_path)
            if success:
                logger.info(f"Database restored from: {backup_path}")
            else:
                logger.warning(f"Database restore failed: {backup_path}")
            return success
        except Exception as e:
            logger.error(f"Database restore failed: {e}")
            return False
