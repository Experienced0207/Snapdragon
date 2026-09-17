"""
PyTorch-to-ONNX Model Export and Verification Script for SignBridge.

Exports the Lightweight Temporal Transformer to ONNX format (models/sign/sign_transformer.onnx)
with dynamic batch axis support and verifies numerical parity between PyTorch and ONNX Runtime outputs.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Dict

try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False

# Ensure workspace root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from training.models.model import TemporalTransformer
from training.datasets.dataset import MasterGlossMap


def find_checkpoint_and_gloss_map() -> Tuple[str, str]:
    """Finds available checkpoint and master_gloss_map paths."""
    search_dirs = [
        os.path.join(os.path.dirname(__file__), "../../experiments/checkpoints"),
        os.path.join(os.path.dirname(__file__), "../experiments/checkpoints"),
        "./experiments/checkpoints",
        "./training/experiments/checkpoints"
    ]

    pth_path = ""
    json_path = ""

    for sdir in search_dirs:
        abs_sdir = os.path.abspath(sdir)
        p_pth = os.path.join(abs_sdir, "best_model.pth")
        p_json = os.path.join(abs_sdir, "master_gloss_map.json")
        if os.path.exists(p_pth) and not pth_path:
            pth_path = p_pth
        if os.path.exists(p_json) and not json_path:
            json_path = p_json

    return pth_path, json_path


def export_and_verify():
    print("=" * 60)
    print("🚀 Starting SignBridge PyTorch -> ONNX Model Export & Verification")
    print("=" * 60)

    pth_path, json_path = find_checkpoint_and_gloss_map()

    # Load Gloss Map or construct fallback map
    if json_path and os.path.exists(json_path):
        print(f"Loading Master Gloss Map from: {json_path}")
        gloss_map = MasterGlossMap.load(json_path)
    else:
        print("Warning: Master gloss map not found. Creating fallback 16-class map.")
        dummy_glosses = ["hello", "thank_you", "water", "need", "please", "yes", "no", "help",
                         "father", "mother", "school", "friend", "home", "good", "bad", "name"]
        gloss_map = MasterGlossMap(dummy_glosses)
        os.makedirs("./experiments/checkpoints", exist_ok=True)
        json_path = "./experiments/checkpoints/master_gloss_map.json"
        gloss_map.save(json_path)

    num_classes = len(gloss_map)
    print(f"Gloss map loaded with {num_classes} total classes.")

    # Instantiate PyTorch Model
    d_model = 128
    num_layers = 4
    if pth_path and os.path.exists(pth_path):
        print(f"Loading weights from checkpoint: {pth_path}")
        checkpoint = torch.load(pth_path, map_location="cpu")
        args = checkpoint.get("args", {})
        d_model = args.get("d_model", 128)
        num_layers = args.get("num_layers", 4)
        model = TemporalTransformer(
            input_dim=258,
            num_classes=num_classes,
            seq_len=30,
            d_model=d_model,
            num_encoder_layers=num_layers
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Warning: Checkpoint best_model.pth not found. Initializing model with random weights.")
        model = TemporalTransformer(
            input_dim=258,
            num_classes=num_classes,
            seq_len=30,
            d_model=d_model,
            num_encoder_layers=num_layers
        )

    model.eval()

    # Output ONNX file path
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/sign"))
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "sign_transformer.onnx")

    # Dummy Input for ONNX Export: shape [1, 30, 258]
    dummy_input = torch.randn(1, 30, 258, dtype=torch.float32)

    # Dynamic Batch Axes definition
    input_names = ['landmark_sequence']
    output_names = ['logits']
    dynamic_axes = {
        'landmark_sequence': {0: 'batch_size'},
        'logits': {0: 'batch_size'}
    }

    print(f"\nExporting PyTorch model to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    print("✅ PyTorch model exported to ONNX successfully!")

    # Verification Test with ONNX Runtime
    if not HAS_ONNXRUNTIME:
        print("Warning: onnxruntime not installed. Skipping numeric verification test.")
        return

    print("\n--- Running Numeric Parity Verification Test ---")
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # Test 1: Batch Size 1
    test_input_b1 = np.random.randn(1, 30, 258).astype(np.float32)
    with torch.no_grad():
        pytorch_out_b1 = model(torch.from_numpy(test_input_b1)).numpy()

    onnx_out_b1 = ort_session.run(output_names, {input_names[0]: test_input_b1})[0]

    np.testing.assert_allclose(
        pytorch_out_b1,
        onnx_out_b1,
        rtol=1e-3,
        atol=1e-5,
        err_msg="PyTorch and ONNX Runtime outputs differ for Batch Size 1!"
    )
    print("  ✓ Batch Size 1 Verification PASSED (Max absolute difference: "
          f"{np.max(np.abs(pytorch_out_b1 - onnx_out_b1)):.2e})")

    # Test 2: Dynamic Batch Size 4
    test_input_b4 = np.random.randn(4, 30, 258).astype(np.float32)
    with torch.no_grad():
        pytorch_out_b4 = model(torch.from_numpy(test_input_b4)).numpy()

    onnx_out_b4 = ort_session.run(output_names, {input_names[0]: test_input_b4})[0]

    np.testing.assert_allclose(
        pytorch_out_b4,
        onnx_out_b4,
        rtol=1e-3,
        atol=1e-5,
        err_msg="PyTorch and ONNX Runtime outputs differ for Dynamic Batch Size 4!"
    )
    print("  ✓ Dynamic Batch Size 4 Verification PASSED (Max absolute difference: "
          f"{np.max(np.abs(pytorch_out_b4 - onnx_out_b4)):.2e})")

    print("\n🎉 ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    from typing import Tuple
    export_and_verify()
