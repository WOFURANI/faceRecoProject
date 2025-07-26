import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.preprocessing.image import img_to_array
from pathlib import Path

# -------------------------------
# Config
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent
HAAR_CASCADE_PATH = BASE_DIR / "haarcascade_frontalface_default.xml"
MODEL_PATH = BASE_DIR / "random_forest_model.pkl"
EMBEDDING_FILE = BASE_DIR / "face_embeddings_vgg16.npz"
IMG_SIZE = (160, 160)

# -------------------------------
# Load resources
# -------------------------------
@st.cache_resource
def load_embedding_model():
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(160, 160, 3))
    return Model(inputs=base_model.input, outputs=base_model.output)

@st.cache_resource
def load_classifier_and_labels():
    model = joblib.load(MODEL_PATH)
    data = np.load(EMBEDDING_FILE, allow_pickle=True)
    return model, data['trainy']

@st.cache_resource
def load_haar_cascade():
    return cv2.CascadeClassifier(str(HAAR_CASCADE_PATH))

embedding_model = load_embedding_model()
classifier, labels = load_classifier_and_labels()
face_cascade = load_haar_cascade()

# -------------------------------
# Functions
# -------------------------------
def extract_face_haar(image_np, required_size=(160, 160)):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face = image_np[y:y+h, x:x+w]
    face = cv2.resize(face, required_size)
    return face

def get_embedding(model, face_image):
    face_image = img_to_array(face_image)
    face_image = np.expand_dims(face_image, axis=0)
    face_image = preprocess_input(face_image)
    feature_map = model.predict(face_image, verbose=0)
    return feature_map.flatten()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Face Recognition App", layout="centered")
st.title("🧠 Face Recognition with VGG16 + HAAR Cascade")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    face = extract_face_haar(image_np, required_size=IMG_SIZE)

    if face is None:
        st.error("❌ No face detected in the uploaded image.")
    else:
        st.image(face, caption="Detected Face", width=160)

        embedding = get_embedding(embedding_model, face)
        prediction = classifier.predict([embedding])[0]
        prediction_proba = classifier.predict_proba([embedding])[0]
        confidence = np.max(prediction_proba) * 100

        st.success(f"🎯 Prediction: **{prediction}**")
        st.info(f"🔍 Confidence: {confidence:.2f}%")
