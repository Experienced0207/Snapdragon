import cv2
import mediapipe as mp
import numpy as np

class SignLandmarkExtractor:
    def __init__(self):
        # Initialize MediaPipe Holistic (captures pose, face, and both hands)
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def extract_keypoints(self, results):
        """
        Flattens the required landmarks into a single 1D NumPy array for the Transformer.
        If a hand isn't visible, it fills with zeros to maintain tensor shape.
        """
        # Pose: 33 landmarks * 4 values (x, y, z, visibility) = 132
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        
        # Left Hand: 21 landmarks * 3 values (x, y, z) = 63
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        
        # Right Hand: 21 landmarks * 3 values (x, y, z) = 63
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        
        # Total features per frame: 132 + 63 + 63 = 258 features
        return np.concatenate([pose, lh, rh])

    def process_frame(self, frame):
        # MediaPipe requires RGB, OpenCV captures in BGR
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

# --- Testing the Extractor ---
if __name__ == "__main__":
    extractor = SignLandmarkExtractor()
    cap = cv2.VideoCapture(0) # 0 is the default Mac webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        image, results = extractor.process_frame(frame)
        
        # Extract the numerical features for the model
        keypoints = extractor.extract_keypoints(results)
        
        # Draw landmarks on the image for visual debugging
        extractor.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, extractor.mp_holistic.HAND_CONNECTIONS)
        extractor.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, extractor.mp_holistic.HAND_CONNECTIONS)
        extractor.mp_drawing.draw_landmarks(image, results.pose_landmarks, extractor.mp_holistic.POSE_CONNECTIONS)

        cv2.putText(image, f"Extracted Features: {len(keypoints)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('SignBridge - Raw Vision Pipeline', image)

        # Press 'q' to quit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()