import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN

class FaceTracker:
    def __init__(self, detection_model='mtcnn', recognition_model_path='custom_model.keras'):
        self.detector = MTCNN()
        self.recognizer = tf.keras.models.load_model(recognition_model_path, compile=False)

    def detect_faces(self, image):
        faces = self.detector.detect_faces(image)
        return [(face['box'], face['confidence']) for face in faces]

    def get_embedding(self, face_crop):
        face_crop = cv2.resize(face_crop, (224, 224))  # VGG16 input size
        face_crop = tf.image.convert_image_dtype(face_crop, tf.float32)[None, ...]  # Add batch dimension
        return self.recognizer.predict(face_crop)[0]
    
    def evaluate_embeddings(self, test_dataset, threshold=0.7):
        embeddings_db = {}  # Simulated database of known embeddings
        correct = 0
        total = 0
        
        for images, labels in test_dataset:
            embeddings = self.recognizer.predict(images)  # Get embeddings
            for i, emb in enumerate(embeddings):
                label = labels[i].numpy()  # Adjust based on your label format
                if label not in embeddings_db:  # Register first occurrence
                    embeddings_db[label] = emb
                else:
                    # Compute cosine similarity with known embedding
                    known_emb = embeddings_db[label]
                    similarity = np.dot(emb, known_emb) / (np.linalg.norm(emb) * np.linalg.norm(known_emb))
                    if similarity > threshold:
                        correct += 1
                    total += 1
        
        accuracy = correct / total
        print(f"Accuracy at threshold {threshold}: {accuracy:.4f}")
        return accuracy