import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from bp_face_recognition.database.database import FaceDatabase

class TestFaceDatabase(unittest.TestCase):
    
    @patch('bp_face_recognition.database.database.psycopg2.connect')
    def test_init_postgres(self, mock_connect):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        FaceDatabase(db_type='postgres')
        
        mock_connect.assert_called()
        mock_cursor.execute.assert_called() # Should create tables
        
    @patch('bp_face_recognition.database.database.pd.DataFrame')
    @patch('bp_face_recognition.database.database.os.path.exists')
    def test_init_csv_new(self, mock_exists, mock_df):
        mock_exists.return_value = False # File doesn't exist
        mock_df_instance = MagicMock()
        mock_df.return_value = mock_df_instance
        
        FaceDatabase(db_type='csv', csv_path='dummy.csv')
        
        mock_df_instance.to_csv.assert_called_with('dummy.csv', index=False)

    @patch('bp_face_recognition.database.database.psycopg2.connect')
    def test_add_face_postgres(self, mock_connect):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        # Mock fetchone to return an ID
        mock_cursor.fetchone.return_value = [123]
        
        db = FaceDatabase(db_type='postgres')
        embedding = np.array([0.1, 0.2], dtype=np.float32)
        face_id = db.add_face(embedding, name='Test')
        
        self.assertEqual(face_id, 123)
        mock_cursor.execute.assert_called()
        self.assertIn(b'INSERT INTO faces', mock_cursor.execute.call_args[0][0].encode())

    @patch('bp_face_recognition.database.database.pd.concat')
    @patch('bp_face_recognition.database.database.pd.read_csv')
    @patch('bp_face_recognition.database.database.os.path.exists')
    def test_add_face_csv(self, mock_exists, mock_read_csv, mock_concat):
        mock_exists.return_value = True
        # Mock existing DF
        mock_df = MagicMock()
        mock_df.__len__.return_value = 10
        mock_read_csv.return_value = mock_df
        
        # Mock result of concat
        mock_new_df = MagicMock()
        mock_concat.return_value = mock_new_df
        
        db = FaceDatabase(db_type='csv', csv_path='dummy.csv')
        embedding = np.array([0.1, 0.2])
        face_id = db.add_face(embedding)
        
        self.assertEqual(face_id, 11) # 10 + 1
        mock_new_df.to_csv.assert_called_with('dummy.csv', index=False)

    @patch('bp_face_recognition.database.database.psycopg2.connect')
    def test_log_detection_postgres(self, mock_connect):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        db = FaceDatabase(db_type='postgres')
        db.log_detection(1, 'Person 1')
        
        mock_cursor.execute.assert_called()
        self.assertIn(b'INSERT INTO logs', mock_cursor.execute.call_args[0][0].encode())

    @patch('bp_face_recognition.database.database.settings')
    @patch('builtins.open', new_callable=MagicMock)
    def test_log_detection_csv(self, mock_open, mock_settings):
        # Setup settings mock
        mock_path = MagicMock()
        mock_settings.LOGS_DIR = mock_path
        mock_path.__truediv__.return_value = 'dummy_logs.txt'
        
        db = FaceDatabase(db_type='csv')
        db.log_detection(1, 'Person 1')
        
        mock_open.assert_called_with('dummy_logs.txt', 'a')
        mock_open.return_value.__enter__.return_value.write.assert_called()

if __name__ == '__main__':
    unittest.main()
