import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from bp_face_recognition.models.methods.mtcnn_detector import MTCNNDetector

class TestMTCNNDetector(unittest.TestCase):
    
    @patch('bp_face_recognition.models.methods.mtcnn_detector.MTCNN')
    def test_detect_faces(self, mock_mtcnn_cls):
        # Setup mock instance
        mock_mtcnn_instance = MagicMock()
        mock_mtcnn_cls.return_value = mock_mtcnn_instance
        
        # Setup mock return value for detect_faces
        # MTCNN returns list of dicts: [{'box': [x, y, w, h], ...}]
        mock_mtcnn_instance.detect_faces.return_value = [
            {'box': [10, 20, 30, 40], 'confidence': 0.99, 'keypoints': {}}
        ]
        
        detector = MTCNNDetector()
        
        # Create a dummy image (height, width, channels)
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Call detect
        faces = detector.detect(dummy_image)
        
        # Verify
        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0], (10, 20, 30, 40))
        mock_mtcnn_instance.detect_faces.assert_called()

    def test_detect_with_confidence(self):
        with patch('bp_face_recognition.models.methods.mtcnn_detector.MTCNN') as mock_mtcnn_cls:
            mock_mtcnn_instance = MagicMock()
            mock_mtcnn_cls.return_value = mock_mtcnn_instance
            mock_mtcnn_instance.detect_faces.return_value = [
                {'box': [10, 20, 30, 40], 'confidence': 0.99, 'keypoints': {}}
            ]
            
            detector = MTCNNDetector()
            dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
            
            faces = detector.detect_with_confidence(dummy_image)
            
            self.assertEqual(len(faces), 1)
            self.assertEqual(faces[0][0], (10, 20, 30, 40))
            self.assertEqual(faces[0][1], 0.99)

    def test_detect_none(self):
        # We don't need to mock MTCNN for None input if the code handles it before calling MTCNN
        with patch('bp_face_recognition.models.methods.mtcnn_detector.MTCNN'):
            detector = MTCNNDetector()
            faces = detector.detect(None)
            self.assertEqual(faces, [])

if __name__ == '__main__':
    unittest.main()
