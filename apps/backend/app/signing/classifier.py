"""
ONNX Runtime Sign Classifier Service for SignBridge.
Provides hardware-adaptive execution provider selection (CoreML on Mac, QNN on Snapdragon ARM64, CPU fallback)
and performs inference on [30, 258] landmark sequences.
"""

import os
import sys
import numpy as np
import scipy.special
from typing import Dict, List, Optional, Union, Tuple

try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False

# Ensure python path resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from training.datasets.dataset import MasterGlossMap


def select_execution_providers() -> List[str]:
    """
    Selects the optimal ONNX Runtime execution providers based on hardware architecture.
    Hierarchy:
      1. QNNExecutionProvider (Qualcomm Snapdragon NPU on Windows ARM64)
      2. CoreMLExecutionProvider (Apple Silicon Neural Engine on macOS)
      3. CUDAExecutionProvider (NVIDIA GPU)
      4. CPUExecutionProvider (Universal Fallback)
    """
    if not HAS_ONNXRUNTIME:
        return ["CPUExecutionProvider"]

    available = ort.get_available_providers()
    preferred_order = [
        "QNNExecutionProvider",
        "CoreMLExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]

    selected = [p for p in preferred_order if p in available]
    if not selected:
        selected = ["CPUExecutionProvider"]
    return selected


class SignClassifier:
    """
    Sign Classifier executing ONNX inference on MediaPipe 258-dim spatial sequences.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        gloss_map_path: Optional[str] = None
    ):
        if not HAS_ONNXRUNTIME:
            raise ImportError("onnxruntime is required for SignClassifier. Run `pip install onnxruntime`.")

        # Default paths resolution
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
        
        if model_path is None:
            model_path = os.path.join(base_dir, "models/sign/sign_transformer.onnx")
        if gloss_map_path is None:
            gloss_map_path = os.path.join(base_dir, "experiments/checkpoints/master_gloss_map.json")

        self.model_path = os.path.abspath(model_path)
        self.gloss_map_path = os.path.abspath(gloss_map_path)

        # 1. Load Gloss Vocabulary Map
        if os.path.exists(self.gloss_map_path):
            self.gloss_map = MasterGlossMap.load(self.gloss_map_path)
        else:
            print(f"[SignClassifier] Gloss map not found at {self.gloss_map_path}. Using default fallback map.")
            dummy_glosses = ["hello", "thank_you", "water", "need", "please", "yes", "no", "help",
                             "father", "mother", "school", "friend", "home", "good", "bad", "name"]
            self.gloss_map = MasterGlossMap(dummy_glosses)

        # 2. Select Execution Providers
        self.providers = select_execution_providers()
        print(f"[SignClassifier] Initializing ONNX session with providers: {self.providers}")

        # Session Options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Create ONNX Session
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"[SignClassifier] ONNX model file not found at: {self.model_path}")

        self.session = ort.InferenceSession(self.model_path, sess_options, providers=self.providers)

        # Get input/output tensor names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def get_selected_provider(self) -> str:
        """Returns active ONNX execution provider."""
        return self.session.get_providers()[0] if self.session else "Unknown"

    def predict(self, sequence: np.ndarray) -> Dict[str, Union[str, float, int]]:
        """
        Runs sign classification on a [30, 258] or [1, 30, 258] landmark array.

        Args:
            sequence: NumPy array of shape (30, 258) or (1, 30, 258)

        Returns:
            Dict containing:
              - "gloss": predicted sign label string
              - "confidence": confidence probability score [0.0, 1.0]
              - "class_id": predicted integer class ID
        """
        if not isinstance(sequence, np.ndarray):
            sequence = np.array(sequence, dtype=np.float32)

        if sequence.ndim == 2:
            if sequence.shape != (30, 258):
                # Auto-pad/downsample if necessary
                from training.preprocessing.mediapipe_utils import temporal_resample_or_pad
                sequence = temporal_resample_or_pad(sequence, target_len=30)
            sequence = np.expand_dims(sequence, axis=0)  # Shape: (1, 30, 258)

        elif sequence.ndim == 3:
            if sequence.shape[1:] != (30, 258):
                raise ValueError(f"Expected sequence shape (Batch, 30, 258), got {sequence.shape}")
        else:
            raise ValueError(f"Invalid sequence dimensions: {sequence.ndim}. Expected 2D or 3D array.")

        sequence = sequence.astype(np.float32)

        # Execute ONNX Inference
        outputs = self.session.run([self.output_name], {self.input_name: sequence})
        logits = outputs[0][0]  # Shape: [num_classes]

        # Apply Softmax for probabilities
        probs = scipy.special.softmax(logits)
        predicted_id = int(np.argmax(probs))
        confidence = float(probs[predicted_id])
        predicted_gloss = self.gloss_map.get_gloss(predicted_id)

        return {
            "gloss": predicted_gloss,
            "confidence": confidence,
            "class_id": predicted_id
        }


if __name__ == "__main__":
    classifier = SignClassifier()
    print(f"Selected Provider: {classifier.get_selected_provider()}")
    dummy_seq = np.random.randn(30, 258).astype(np.float32)
    res = classifier.predict(dummy_seq)
    print("Inference Result:", res)
