import psycopg2
import sys
import os

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), "src"))

from bp_face_recognition.config.settings import settings


def setup_database():
    print(
        f"Connecting to PostgreSQL database '{settings.DB_NAME}' on {settings.DB_HOST}..."
    )
    try:
        conn = psycopg2.connect(
            dbname=settings.DB_NAME,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD,
            host=settings.DB_HOST,
            port=settings.DB_PORT,
        )
        conn.autocommit = True
        cursor = conn.cursor()

        print("Creating tables...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id SERIAL PRIMARY KEY,
                embedding BYTEA NOT NULL,
                name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS logs (
                id SERIAL PRIMARY KEY,
                face_id INTEGER REFERENCES faces(id),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                label TEXT
            );
            
            CREATE INDEX IF NOT EXISTS idx_faces_name ON faces(name);
            CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp);
        """)

        print("Database setup complete.")
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error setting up database: {e}")
        print("\nMake sure:")
        print(f"1. PostgreSQL is running on {settings.DB_HOST}:{settings.DB_PORT}")
        print(
            f"2. Database '{settings.DB_NAME}' exists (create it with 'CREATE DATABASE {settings.DB_NAME};')"
        )
        print(
            f"3. Credentials for user '{settings.DB_USER}' are correct in .env or settings.py"
        )


if __name__ == "__main__":
    setup_database()
