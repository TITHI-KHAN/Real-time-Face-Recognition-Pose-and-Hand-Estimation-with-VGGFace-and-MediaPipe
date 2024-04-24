# Real-time Face Recognition, Pose and Hand Estimation with VGGFace and MediaPipe

## To run the code, please either use "Visual Studio" or "Jupyter Notebook from Anaconda Navigator".

### Thank you.

<br>

## Code Explanation:

1. **Importing Libraries**: The code imports necessary libraries including `cv2` for OpenCV operations, `os` for file operations, `numpy` for numerical computations, `mediapipe` for pose and hand estimation, and specific modules from `keras_vggface` and `keras.preprocessing` for working with the VGGFace model and preprocessing images.

2. **Loading MediaPipe Models**: It loads the `Pose` and `Hands` models from MediaPipe for pose and hand estimation.

3. **Loading VGGFace Model**: It loads the VGGFace model with ResNet50 architecture (`model='resnet50'`) for feature extraction.

4. **Loading Known Faces**: It iterates through the directories in the 'training_data' directory, extracts features from the first image of each person (assuming one image per person), and stores the features along with the person's name in the `known_faces` dictionary.

5. **Feature Extraction**: It defines a function `extract_features()` to preprocess and extract features from a given face using the loaded VGGFace model.

6. **Capturing Video and Processing Frames**: It initializes a video capture object using `cv2.VideoCapture(0)` to capture frames from the default camera (index 0). It then continuously captures frames from the video feed and converts each frame to RGB format for processing.

7. **Pose and Hand Estimation**: It processes each frame for pose and hand estimation using the MediaPipe models. It draws landmarks and connections for pose and hand landmarks on the frame using `mp_drawing.draw_landmarks()`.

8. **Face Recognition**: It resizes each frame to match the input size expected by the face recognition model. It extracts features from the detected faces using the VGGFace model and calculates the cosine similarity with known faces to recognize the identity of each detected face.

9. **Displaying Results**: It displays the processed frame with landmarks, hand landmarks, and predicted identities along with cosine distances using `cv2.imshow()`.

10. **Exiting**: The program exits the loop if the 'q' key is pressed, releases resources used for pose and hand estimation, video capture, and closes all OpenCV windows.

## Key Points:
- Utilizes VGGFace model pre-trained on ResNet50 architecture for feature extraction.
- Uses MediaPipe for real-time pose and hand estimation.
- Processes video frames for face recognition, pose estimation, and hand estimation simultaneously.
- Draws landmarks and connections for pose and hand landmarks on the frames.
- Recognizes faces and displays predicted identities along with cosine distances.
- Allows quitting the application by pressing the 'q' key.   
