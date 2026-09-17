import os
import sys
import cv2
import numpy as np
import time

# Add the backend directory to Python's path to import our extractor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../apps/backend')))
from app.vision.extractor import SignLandmarkExtractor

# --- Configuration ---
DATA_PATH = os.path.join(os.path.dirname(__file__), 'raw_numpy')
# Start with a small vocabulary for your first test
SIGNS = np.array(['hello', 'water', 'need']) 
NO_SEQUENCES = 30     # 30 videos per sign
SEQUENCE_LENGTH = 30  # 30 frames per video

def setup_folders():
    for sign in SIGNS:
        for sequence in range(NO_SEQUENCES):
            try:
                os.makedirs(os.path.join(DATA_PATH, sign, str(sequence)))
            except OSError:
                pass

def collect_data():
    setup_folders()
    extractor = SignLandmarkExtractor()
    cap = cv2.VideoCapture(0)

    for sign in SIGNS:
        print(f"\n--- Get ready to sign: {sign.upper()} ---")
        time.sleep(3) # Give yourself 3 seconds to get in position
        
        for sequence in range(NO_SEQUENCES):
            sequence_data = []
            
            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret:
                    break

                image, results = extractor.process_frame(frame)
                
                # UI Feedback on the screen
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for {sign} Video Number {sequence}', (15, 12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('SignBridge - Data Collection', image)
                    cv2.waitKey(2000) # 2-second pause between videos to reset your hands
                else:
                    cv2.putText(image, f'Collecting frames for {sign} Video Number {sequence}', (15, 12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('SignBridge - Data Collection', image)

                # Extract and store the numerical data
                keypoints = extractor.extract_keypoints(results)
                sequence_data.append(keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # Save the 30-frame sequence as a NumPy array
            npy_path = os.path.join(DATA_PATH, sign, str(sequence), f"seq_{sequence}.npy")
            np.save(npy_path, np.array(sequence_data))
            print(f"Saved: {npy_path}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting Dataset Collection...")
    collect_data()
    print("\nCollection Complete! Data saved to training/datasets/raw_numpy/")