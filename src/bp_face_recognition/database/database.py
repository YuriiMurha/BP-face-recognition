import os
import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime
from bp_face_recognition.config.settings import settings

class FaceDatabase:
    def __init__(self, db_type='postgres', conn_params=None, csv_path=None):
        self.db_type = db_type
        if db_type == 'postgres':
            if conn_params is None:
                conn_params = {
                    'dbname': settings.DB_NAME,
                    'user': settings.DB_USER,
                    'password': settings.DB_PASSWORD,
                    'host': settings.DB_HOST,
                    'port': settings.DB_PORT
                }
            self.conn = psycopg2.connect(**conn_params)
            self.cursor = self.conn.cursor()
            self._init_postgres()
        else:
            self.csv_path = csv_path or str(settings.DATA_DIR / 'faces.csv')
            self._init_csv()

    def _init_postgres(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id SERIAL PRIMARY KEY,
                embedding BYTEA NOT NULL,
                name TEXT
            );
            CREATE TABLE IF NOT EXISTS logs (
                id SERIAL PRIMARY KEY,
                face_id INTEGER,
                timestamp TIMESTAMP,
                label TEXT
            );
        """)
        self.conn.commit()

    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            pd.DataFrame(columns=['id', 'embedding']).to_csv(self.csv_path, index=False)

    def add_face(self, embedding, name=None):
        if self.db_type == 'postgres':
            self.cursor.execute("INSERT INTO faces (embedding, name) VALUES (%s, %s) RETURNING id",
                                (embedding.tobytes(), name))
            face_id = self.cursor.fetchone()[0]
            self.conn.commit()
        else:
            df = pd.read_csv(self.csv_path)
            face_id = len(df) + 1
            # Using concat instead of append (deprecated in pandas)
            new_row = pd.DataFrame({'id': [face_id], 'embedding': [embedding.tolist()]})
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(self.csv_path, index=False)
        return face_id

    def get_all_embeddings(self):
        if self.db_type == 'postgres':
            self.cursor.execute("SELECT id, embedding FROM faces")
            return [(row[0], np.frombuffer(row[1], dtype=np.float32)) for row in self.cursor.fetchall()]
        else:
            if not os.path.exists(self.csv_path):
                return []
            df = pd.read_csv(self.csv_path)
            return [(row['id'], np.array(eval(row['embedding']))) for _, row in df.iterrows()]

    def log_detection(self, face_id, label):
        timestamp = datetime.now()
        if self.db_type == 'postgres':
            self.cursor.execute("INSERT INTO logs (face_id, timestamp, label) VALUES (%s, %s, %s)",
                                (face_id, timestamp, label))
            self.conn.commit()
        else:
            log_file = settings.LOGS_DIR / 'logs.txt'
            settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
            with open(log_file, 'a') as f:
                f.write(f"{face_id},{timestamp},{label}\n")
