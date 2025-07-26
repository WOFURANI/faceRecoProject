import cv2
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
haar_cascade_path = BASE_DIR / 'haarcascade_frontalface_default.xml'
train_dir = BASE_DIR / 'dataset' / 'train'
test_dir = BASE_DIR / 'dataset' / 'test'
face_cascade = cv2.CascadeClassifier(str(haar_cascade_path))

def extract_face(image_path, required_size=(160, 160)):

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Unable to load image {image_path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    if len(faces) == 0:
        print(f"No face detected in {image_path}")
        return None

    # There is only one face expected per image
    x, y, w, h = faces[0]
    face = image[y:y+h, x:x+w]
    face = cv2.resize(face, required_size)
    return face
def load_faces(directory):

    faces = []
    for image_path in Path(directory).iterdir():
        if image_path.is_file():
            face = extract_face(image_path)
            if face is not None:
                faces.append(face)
    return faces
def load_dataset(parent_directory):

    x, y = [], []
    for directory in Path(parent_directory).iterdir():
        if directory.is_dir():
            faces = load_faces(directory)
            labels = [directory.name] * len(faces)
            x.extend(faces)
            y.extend(labels)
    return np.array(x), np.array(y)
trainx, trainy = load_dataset(train_dir)
print(f"Training faces: {trainx.shape}, labels: {trainy.shape}")

testx, testy = load_dataset(test_dir)
print(f"Testing faces: {testx.shape}, labels: {testy.shape}")

np.savez_compressed(BASE_DIR / "processed_faces.npz", trainx=trainx, trainy=trainy, testx=testx, testy=testy)
print("Dataset saved to 'processed_faces.npz'")