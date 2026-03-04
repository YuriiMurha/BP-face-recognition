import os
import json
import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime
from bp_face_recognition.config.settings import settings


class FaceDatabase:
    def __init__(self, db_type="postgres", conn_params=None, csv_path=None):
        self.db_type = db_type
        if db_type == "postgres":
            if conn_params is None:
                conn_params = {
                    "dbname": settings.DB_NAME,
                    "user": settings.DB_USER,
                    "password": settings.DB_PASSWORD,
                    "host": settings.DB_HOST,
                    "port": settings.DB_PORT,
                }
            self.conn = psycopg2.connect(**conn_params)
            self.cursor = self.conn.cursor()
            self._init_postgres()
        else:
            self.csv_path = csv_path or str(settings.DATA_DIR / "faces.csv")
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
            pd.DataFrame({"id": [], "name": [], "embedding": []}).to_csv(
                self.csv_path, index=False
            )

    def add_face(self, embedding, name=None):
        if self.db_type == "postgres":
            self.cursor.execute(
                "INSERT INTO faces (embedding, name) VALUES (%s, %s) RETURNING id",
                (embedding.tobytes(), name),
            )
            face_id = self.cursor.fetchone()[0]
            self.conn.commit()
        else:
            df = pd.read_csv(self.csv_path)
            face_id = len(df) + 1
            # Using concat instead of append (deprecated in pandas)
            new_row = pd.DataFrame(
                {"id": [face_id], "name": [name], "embedding": [embedding.tolist()]}
            )
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(self.csv_path, index=False)
        return face_id

    def get_all_embeddings(self):
        if self.db_type == "postgres":
            self.cursor.execute("SELECT name, embedding FROM faces")
            results = self.cursor.fetchall()

            embeddings_dict = {}
            for name, emb_bytes in results:
                identity = name or "Unknown"
                if identity not in embeddings_dict:
                    embeddings_dict[identity] = []
                embeddings_dict[identity].append(
                    np.frombuffer(emb_bytes, dtype=np.float32)
                )
            return embeddings_dict
        else:
            if not os.path.exists(self.csv_path):
                return {}
            df = pd.read_csv(self.csv_path)

            embeddings_dict = {}
            for _, row in df.iterrows():
                identity = row.get("name", "Unknown")
                if pd.isna(identity):
                    identity = "Unknown"

                if identity not in embeddings_dict:
                    embeddings_dict[identity] = []

                emb = np.array(json.loads(str(row["embedding"])))
                embeddings_dict[identity].append(emb)
            return embeddings_dict

    def log_detection(self, face_id, label):
        timestamp = datetime.now()
        if self.db_type == "postgres":
            self.cursor.execute(
                "INSERT INTO logs (face_id, timestamp, label) VALUES (%s, %s, %s)",
                (face_id, timestamp, label),
            )
            self.conn.commit()
        else:
            log_file = settings.LOGS_DIR / "logs.txt"
            settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a") as f:
                f.write(f"{face_id},{timestamp},{label}\n")

    def list_all_people(self):
        if self.db_type == "postgres":
            self.cursor.execute("SELECT DISTINCT name FROM faces")
            return [row[0] for row in self.cursor.fetchall() if row[0]]
        else:
            if not os.path.exists(self.csv_path):
                return []
            df = pd.read_csv(self.csv_path)
            return df["name"].dropna().unique().tolist()

    def get_person_info(self, name):
        # Basic implementation for now
        return {"name": name}

    def update_metadata(self, name, metadata):
        # Stub for now
        return True

    def delete_person(self, name):
        if self.db_type == "postgres":
            self.cursor.execute("DELETE FROM faces WHERE name = %s", (name,))
            self.conn.commit()
            return True
        else:
            if not os.path.exists(self.csv_path):
                return False
            df = pd.read_csv(self.csv_path)
            df = df[df["name"] != name]
            df.to_csv(self.csv_path, index=False)
            return True

    def get_stats(self):
        if self.db_type == "postgres":
            self.cursor.execute("SELECT count(*), count(DISTINCT name) FROM faces")
            row = self.cursor.fetchone()
            return {"total_embeddings": row[0], "total_people": row[1]}
        else:
            if not os.path.exists(self.csv_path):
                return {"total_embeddings": 0, "total_people": 0}
            df = pd.read_csv(self.csv_path)
            return {
                "total_embeddings": len(df),
                "total_people": len(df["name"].dropna().unique()),
            }

    def backup(self, path):
        if self.db_type == "csv":
            import shutil

            shutil.copy2(self.csv_path, path)
            return True
        return False

    def restore(self, path):
        if self.db_type == "csv":
            import shutil

            shutil.copy2(path, self.csv_path)
            return True
        return False
