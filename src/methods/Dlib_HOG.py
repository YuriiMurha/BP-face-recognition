import dlib
import cv2
import time

# Load dlib's HOG face detector
hog_face_detector = dlib.get_frontal_face_detector()

def detect_faces_hog(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    start_time = time.time()
    faces = hog_face_detector(gray, 1)
    detection_time = time.time() - start_time

    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image, detection_time, len(faces)

# Example usage
image_path = 'face-recognition\Basic\ImagesBasic\webcam_example.jpg'
output_image, detection_time, num_faces = detect_faces_hog(image_path)
print(f"HOG - Detected {num_faces} faces in {detection_time:.4f} seconds")
cv2.imshow("Detected Faces", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
