import os
import argparse
import sys
import cv2
import numpy as np
from tqdm import tqdm

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), "src"))

from bp_face_recognition.database.database import FaceDatabase
from bp_face_recognition.models.model import FaceTracker
from bp_face_recognition.config.settings import settings


def register_person(name, image_dir, db_type="csv", threshold=0.9):
    """
    Processes a directory of images for a single person, averages embeddings,
    and registers them in the database.
    """
    if not os.path.isdir(image_dir):
        print(f"Error: Directory not found: {image_dir}")
        return

    db = FaceDatabase(db_type=db_type)
    tracker = FaceTracker()

    embeddings = []
    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not image_files:
        print(f"No valid images found in {image_dir}")
        return

    print(f"Processing {len(image_files)} images for {name}...")

    for img_name in tqdm(image_files, desc="Extracting embeddings"):
        img_path = os.path.join(image_dir, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # Detect faces
        faces = tracker.detect_faces(frame)
        if not faces:
            continue

        # Take the largest face if multiple detected
        faces.sort(key=lambda x: x[0][2] * x[0][3], reverse=True)
        (x, y, w, h), conf = faces[0]

        if conf < threshold:
            continue

        face_crop = frame[max(0, y) : y + h, max(0, x) : x + w]
        if face_crop.size == 0:
            continue

        embedding = tracker.get_embedding(face_crop)
        embeddings.append(embedding)

    if not embeddings:
        print("Error: Could not extract high-quality embeddings from any image.")
        return

    # Average the embeddings for a more robust representation
    avg_embedding = np.mean(embeddings, axis=0)
    face_id = db.add_face(avg_embedding, name=name)

    print(
        f"Successfully registered {name} (ID: {face_id}) using {len(embeddings)} images."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Register a person from a directory of images"
    )
    parser.add_argument("name", type=str, help="Name of the person")
    parser.add_argument("dir", type=str, help="Directory containing images")
    parser.add_argument(
        "--db",
        type=str,
        default="csv",
        choices=["csv", "postgres"],
        help="Database type",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.9, help="Detection confidence threshold"
    )

    args = parser.parse_args()
    register_person(args.name, args.dir, args.db, args.threshold)
