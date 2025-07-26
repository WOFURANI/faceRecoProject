import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib

# Load face embeddings
data = np.load("face_embeddings_vgg16.npz", allow_pickle=True)
trainX, trainy = data['trainX'], data['trainy']
testX, testy = data['testX'], data['testy']

# Define classifiers
classifiers = {

    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}
accuracies = {}
best_model = None
best_score = 0
best_name = ""

# Train and evaluate each classifier
for name, clf in classifiers.items():
    clf.fit(trainX, trainy)
    preds = clf.predict(testX)
    acc = accuracy_score(testy, preds)
    accuracies[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

    # Save the best model
    if acc > best_score:
        best_score = acc
        best_model = clf
        best_name = name

# Save the best model
joblib.dump(best_model, f"{best_name.lower().replace(' ', '_')}_model.pkl")
print(f"✅ Best model '{best_name}' saved as {best_name.lower().replace(' ', '_')}_model.pkl")