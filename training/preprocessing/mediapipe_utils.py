"""
MediaPipe Landmark Extractor & Temporal Resampler Utility for SignBridge.
Extracts 258-dimensional spatial landmark coordinates (Pose: 132, Left Hand: 63, Right Hand: 63)
and handles temporal padding/downsampling to guarantee a uniform shape [30, 258].
"""

import os
import cv2
import numpy as np
import torch
from typing import Union, List, Optional

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False


class MediaPipeFeatureExtractor:
    """
    Extracts spatial coordinates from MediaPipe Holistic solution.
    Outputs a flattened 1D array of 258 features per frame:
      - Pose: 33 landmarks * 4 (x, y, z, visibility) = 132
      - Left Hand: 21 landmarks * 3 (x, y, z) = 63
      - Right Hand: 21 landmarks * 3 (x, y, z) = 63
      Total = 258 features
    """

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        if not HAS_MEDIAPIPE:
            raise ImportError("mediapipe is required for MediaPipeFeatureExtractor. Install via `pip install mediapipe`.")
        
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def extract_keypoints_from_results(self, results) -> np.ndarray:
        """Flattens holistic landmark results into a [258] float32 array."""
        # Pose landmarks (33 * 4 = 132)
        if results and results.pose_landmarks:
            pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark], dtype=np.float32).flatten()
        else:
            pose = np.zeros(33 * 4, dtype=np.float32)

        # Left hand landmarks (21 * 3 = 63)
        if results and results.left_hand_landmarks:
            lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark], dtype=np.float32).flatten()
        else:
            lh = np.zeros(21 * 3, dtype=np.float32)

        # Right hand landmarks (21 * 3 = 63)
        if results and results.right_hand_landmarks:
            rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark], dtype=np.float32).flatten()
        else:
            rh = np.zeros(21 * 3, dtype=np.float32)

        return np.concatenate([pose, lh, rh], axis=0)

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Processes a single BGR image frame and returns a [258] feature vector."""
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.holistic.process(image_rgb)
        return self.extract_keypoints_from_results(results)

    def process_video_path(self, video_path: str) -> np.ndarray:
        """
        Reads a video file from video_path, processes frames, and returns array of shape [N, 258].
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        frame_features = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            feats = self.process_frame(frame)
            frame_features.append(feats)

        cap.release()

        if len(frame_features) == 0:
            return np.zeros((0, 258), dtype=np.float32)

        return np.array(frame_features, dtype=np.float32)

    def close(self):
        if hasattr(self, 'holistic') and self.holistic:
            self.holistic.close()


def temporal_resample_or_pad(features: np.ndarray, target_len: int = 30) -> np.ndarray:
    """
    Normalizes feature sequence temporal dimension to exactly target_len frames [30, 258].
    
    Args:
        features: NumPy array of shape (N, 258) or (N, D).
        target_len: Target sequence length (default 30).
        
    Returns:
        NumPy array of shape (target_len, 258), dtype float32.
    """
    if features is None or len(features) == 0:
        return np.zeros((target_len, 258), dtype=np.float32)

    num_frames, feature_dim = features.shape

    if feature_dim != 258:
        # If input features have a different dimension, adjust or pad/truncate to 258
        if feature_dim < 258:
            pad_dim = np.zeros((num_frames, 258 - feature_dim), dtype=np.float32)
            features = np.concatenate([features, pad_dim], axis=1)
        else:
            features = features[:, :258]

    if num_frames == target_len:
        return features.astype(np.float32)

    elif num_frames < target_len:
        # Pad with zeros at the end
        padding = np.zeros((target_len - num_frames, 258), dtype=np.float32)
        return np.vstack([features, padding]).astype(np.float32)

    else:
        # Uniformly downsample to target_len frames
        indices = np.linspace(0, num_frames - 1, target_len).astype(int)
        return features[indices].astype(np.float32)
