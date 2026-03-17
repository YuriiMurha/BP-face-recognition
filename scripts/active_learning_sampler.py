import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), "src"))

from bp_face_recognition.database.database import FaceDatabase
from bp_face_recognition.models.model import FaceTracker
from bp_face_recognition.config.settings import settings


def active_learning_sampler(
    input_dir, lower_bound=0.35, upper_bound=0.6, output_dir=None
):
    """
    Scans a directory of images, runs face recognition, and saves crops
    where the model is 'uncertain' (similarity between lower and upper bounds).
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input directory {input_dir} not found.")
        return

    if output_dir is None:
        output_dir = settings.DATA_DIR / "active_learning" / "to_label"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Initializing Active Learning Sampler...")
    print(f"🔍 Range: [{lower_bound} - {upper_bound}]")
    print(f"📂 Output: {output_dir}")

    tracker = FaceTracker()
    db = FaceDatabase(db_type="csv")
    known_embeddings = db.get_all_embeddings()

    if not known_embeddings:
        print(
            "⚠️ Database is empty. All high-confidence faces will be considered strangers."
        )
        # If DB is empty, we might want to just save all detected faces as potential new data

    image_files = [
        f
        for f in os.listdir(input_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    uncertain_count = 0

    for img_name in tqdm(image_files, desc="Sampling images"):
        img_path = input_path / img_name
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        results = tracker.detect_faces(frame)

        for i, (box, det_conf) in enumerate(results):
            if det_conf < 0.8:  # Only consider reliable detections
                continue

            x, y, w, h = box
            x, y = max(0, x), max(0, y)
            face_crop = frame[y : y + h, x : x + w]

            if face_crop.size == 0:
                continue

            embedding = tracker.get_embedding(face_crop)

            # Find best match in DB
            max_sim = -1.0
            best_id = None

            for face_id, known_emb in known_embeddings:
                # Cosine similarity
                sim = np.dot(embedding, known_emb) / (
                    np.linalg.norm(embedding) * np.linalg.norm(known_emb)
                )
                if sim > max_sim:
                    max_sim = sim
                    best_id = face_id

            # Active Learning Logic: Is it in the 'uncertainty' zone?
            if lower_bound <= max_sim <= upper_bound:
                uncertain_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_name = f"uncertain_{timestamp}_{img_name.split('.')[0]}_face{i}_sim{max_sim:.2f}.jpg"
                cv2.imwrite(str(output_dir / save_name), face_crop)

                # Also save a small text file with metadata for labeling
                with open(output_dir / f"{save_name}.json", "w") as f:
                    import json

                    json.dump(
                        {
                            "original_image": str(img_name),
                            "similarity": float(max_sim),
                            "best_match_id": int(best_id) if best_id else None,
                            "detection_confidence": float(det_conf),
                        },
                        f,
                    )

    print(f"✅ Finished. Sampled {uncertain_count} uncertain faces to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample uncertain faces for active learning"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to unlabelled images"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save uncertain crops"
    )
    parser.add_argument(
        "--lower", type=float, default=0.35, help="Lower similarity bound"
    )
    parser.add_argument(
        "--upper", type=float, default=0.6, help="Upper similarity bound"
    )

    args = parser.parse_args()
    active_learning_sampler(args.input, args.lower, args.upper, args.output)
