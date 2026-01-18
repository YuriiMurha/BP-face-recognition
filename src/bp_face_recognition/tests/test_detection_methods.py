import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from bp_face_recognition.evaluation.detection_methods import detect_faces_haar

class TestDetectionMethods(unittest.TestCase):
    
    @patch('bp_face_recognition.evaluation.detection_methods.cv2')
    @patch('bp_face_recognition.evaluation.detection_methods.face_cascade')
    def test_detect_faces_haar(self, mock_face_cascade, mock_cv2):
        # Mock cv2.cvtColor
        mock_cv2.cvtColor.return_value = MagicMock()
        mock_cv2.COLOR_BGR2GRAY = 1
        
        # Mock face_cascade.detectMultiScale
        # Returns list of rectangles [x, y, w, h]
        mock_face_cascade.detectMultiScale.return_value = [(10, 10, 50, 50)]
        
        # Dummy image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        faces, time_taken = detect_faces_haar(image)
        
        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0], (10, 10, 50, 50))
        self.assertIsInstance(time_taken, float)

if __name__ == '__main__':
    unittest.main()
