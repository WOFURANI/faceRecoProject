import numpy as np
from pathlib import Path
from tensorflow import keras
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.preprocessing.image import img_to_array

# Load processed faces from Phase 1
data = np.load("processed_faces.npz", allow_pickle=True)
trainX, trainy = data["trainx"], data["trainy"]
testX, testy = data["testx"], data["testy"]

# Load pre-trained VGG16 and remove the top classification layer
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(160, 160, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)
print("✅ VGG16 loaded")

# Embedding function using VGG16
def get_embedding(model, face_image):
    face_image = img_to_array(face_image)
    face_image = np.expand_dims(face_image, axis=0)
    face_image = preprocess_input(face_image)  # VGG16 preprocessing
    feature_map = model.predict(face_image)
    return feature_map.flatten()  # Flatten to 1D vector

# Generate embeddings
embedded_trainX = np.asarray([get_embedding(model, face) for face in trainX])
embedded_testX = np.asarray([get_embedding(model, face) for face in testX])

print("✅ Embeddings generated")
print("Train Embeddings Shape:", embedded_trainX.shape)
print("Test Embeddings Shape:", embedded_testX.shape)

# Optional: Check distance between two embeddings
from numpy.linalg import norm
print("Distance between 0 and 1:", norm(embedded_trainX[0] - embedded_trainX[2]))

# Save embeddings
BASE_DIR = Path(__file__).resolve().parent
np.savez_compressed(BASE_DIR / "face_embeddings_vgg16.npz",
                    trainX=embedded_trainX, trainy=trainy,
                    testX=embedded_testX, testy=testy)
print("💾 Embeddings saved to 'face_embeddings_vgg16.npz'")