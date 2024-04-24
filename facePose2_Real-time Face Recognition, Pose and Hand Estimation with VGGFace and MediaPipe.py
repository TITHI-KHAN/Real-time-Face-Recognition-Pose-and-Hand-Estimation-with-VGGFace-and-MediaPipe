import cv2
import os
import numpy as np
import mediapipe as mp
from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.preprocessing import image


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)


model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')


known_faces = {}
training_data_path = 'training_data'
for dir_name in os.listdir(training_data_path):
    subject_path = os.path.join(training_data_path, dir_name)
    if not os.path.isdir(subject_path):
        continue

    face_images = os.listdir(subject_path)
    if face_images:

        features = model.predict(preprocess_input(np.expand_dims(image.img_to_array(image.load_img(os.path.join(subject_path, face_images[0]), target_size=(224, 224))), axis=0)))
        known_faces[dir_name] = features

def extract_features(face):
    face_array = np.asarray(face, dtype='float32')
    face_array = np.expand_dims(face_array, axis=0)
    face_array = preprocess_input(face_array)
    return model.predict(face_array)

# Real-time face recognition and pose estimation
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break


    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    

    results_pose = pose.process(frame_rgb)
    results_hands = hands.process(frame_rgb)


    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


    faces = cv2.resize(frame, (224, 224))  # Resize frame to match face recognition model expected input
    captured_features = extract_features(faces)


    min_dist = float('inf')
    identity = None


    for name, features in known_faces.items():
        dist = cosine(features, captured_features)
        if dist < min_dist:
            min_dist = dist
            identity = name

    label_text = f"{identity if identity else 'Unknown'}, {min_dist:.2f}"
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Real-time Face Recognition, Pose and Hand Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pose.close()
hands.close()
video_capture.release()
cv2.destroyAllWindows()
