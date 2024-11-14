import cv2
from skimage.feature import hog
from joblib import load
import requests
import numpy as np
import imutils
import serial 
import time

# pretrained SVM model
clf = load('hog_svm_model.joblib')

arduino = serial.Serial(port='COM7', baudrate=9600, timeout=1)
time.sleep(2)  # Give the connection a second to settle

# URL to get feed from smartphone camera
url = "http://192.168.235.104:8080/shot.jpg"

hog_params = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys'
}

def extract_hog_features(image):
    features, _ = hog(image, visualize=True, **hog_params)
    return features

def detect_object(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_resized = cv2.resize(gray_frame, (128, 128))
    features = extract_hog_features(frame_resized)
    prediction = clf.predict([features])

    if prediction == 1:
        cv2.putText(frame, "Object Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        arduino.write(b'MOVE\n')  # Send a command to the Arduino to trigger movement

    return frame

# webcam feed
cap = cv2.VideoCapture(0)

while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    frame = imutils.resize(img, width=720, height=480)

    frame = detect_object(frame)
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
