import cv2
import os
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from joblib import dump, load

# Paths for positive and negative sample directories
positive_path = 'data/positives/'
negative_path = 'data/negatives/'

# Parameters for HOG
hog_params = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys'
}

# Function to extract HOG features
def extract_hog_features(image):
    features, _ = hog(
        image,
        visualize=True,
        **hog_params
    )
    return features

# Load images and extract HOG features
X = []
y = []

for img_name in os.listdir(positive_path):
    img = cv2.imread(os.path.join(positive_path, img_name), cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (128, 128))
    features = extract_hog_features(img_resized)
    X.append(features)
    y.append(1)  # Label for positive class

for img_name in os.listdir(negative_path):
    img = cv2.imread(os.path.join(negative_path, img_name), cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (128, 128))
    features = extract_hog_features(img_resized)
    X.append(features)
    y.append(0)  # Label for negative class

X = np.array(X)
y = np.array(y)

# Train a linear SVM
clf = LinearSVC()
clf.fit(X, y)

# Save the trained model
dump(clf, 'hog_svm_model.joblib')

print("Training complete. Model saved as 'hog_svm_model.joblib'.")
