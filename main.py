import cv2
import time
from flask import Flask, render_template, Response
import threading
from collections import deque

# Flask application
app = Flask(__name__)

# Load models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(8-12)','(13-16)','(17-19)','(15-20)','(21-24)','(25-32)','(33-37)','(38-43)']
genderList = ['Male', 'Female']

# Load the networks
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Padding for face detection
padding = 20

# Define deque to store last few predictions for smoothing
gender_history = deque(maxlen=5)
age_history = deque(maxlen=5)

# Function to get face bounding box
def getFaceBox(net, frame, conf_threshold=0.75):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)

    return frameOpencvDnn, bboxes

# Function to generate video feed
def gen_frames():
    global gender_history, age_history

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for better performance
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Get faces and annotations
        frameFace, bboxes = getFaceBox(faceNet, small_frame, conf_threshold=0.5)  # Lowered threshold

        if not bboxes:
            continue

        # Process each detected face
        for bbox in bboxes:
            face = small_frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                               max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
            
            # Resize the face region for consistency
            face_resized = cv2.resize(face, (227, 227))  # Resize to model input size
            blob = cv2.dnn.blobFromImage(face_resized, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            # Gender prediction
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            # Age prediction
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]

            # Add current predictions to the history
            gender_history.append(gender)
            age_history.append(age)

            # Get most frequent prediction from the history (smoothing)
            if len(gender_history) > 1:
                gender = max(set(gender_history), key=gender_history.count)
            if len(age_history) > 1:
                age = max(set(age_history), key=age_history.count)

            label = "{}, {}".format(gender, age)
            cv2.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # Encode frame in JPEG format and yield it for streaming
        ret, buffer = cv2.imencode('.jpg', frameFace)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Video feed route
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
