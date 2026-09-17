# SignBridge: Edge AI Sign Language Translator

<div align="center">

![Platform](https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon%20%7C%20Windows%2011%20ARM64-blue?style=for-the-badge&logo=apple)
![Hardware Acceleration](https://img.shields.io/badge/Hardware%20NPU-Qualcomm%20Hexagon%20%7C%20Apple%20ANE-0096D6?style=for-the-badge&logo=qualcomm)
![Privacy](https://img.shields.io/badge/Privacy-100%25%20Local%20%7C%20Cloud%20Req%3A%200-10B981?style=for-the-badge&logo=shield)
![Framework](https://img.shields.io/badge/Desktop-Tauri%202%20%2B%20React%2018-06B6D4?style=for-the-badge&logo=tauri)
![AI Engine](https://img.shields.io/badge/AI%20Engine-PyTorch%20%7C%20ONNX%20Runtime%20%7C%20QNN-F43F5E?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-slate?style=for-the-badge)

<p align="center">
  <strong>Bridging Communication Barriers in Real-Time with Local On-Device AI</strong>
  <br />
  Recognizes Indian Sign Language (ISL) gestures via client-side vision, translates them into coherent grammatical sentences via on-device NLP, and verbalizes speech via low-latency neural TTS.
</p>

</div>

---

## 📌 Executive Summary

**SignBridge** is a native, privacy-preserving desktop application engineered for seamless two-way communication between Deaf/Hard-of-Hearing individuals and hearing individuals. Built according to enterprise **HP Accessibility** guidelines, SignBridge operates **100% locally on-device**, eliminating cloud latency, recurring API costs, and privacy vulnerabilities.

- **Zero Cloud Dependence**: All vision feature extraction, sign classification, language reconstruction, and speech synthesis execute locally.
- **Dual-Platform Architecture**: Developed and profiled on **macOS Apple Silicon (CoreML/ANE)** and compiled for **Windows 11 ARM64 Qualcomm Snapdragon X Elite (Hexagon NPU)**.
- **Sub-50ms Translation Loop**: Sustained 30 FPS video landmark capture paired with an ultra-lightweight 4-layer Temporal Transformer.

---

## 🏛️ System Architecture: Mac Dev Mode vs. Snapdragon Production Mode

```
+===================================================================================================+
|                                        SIGNBRIDGE ARCHITECTURE                                    |
+===================================================================================================+
                                                   |
                   +-------------------------------+-------------------------------+
                   |                                                               |
                   v                                                               v
+---------------------------------------------+   +-------------------------------------------------+
|          DEVELOPMENT MODE (macOS)           |   |       PRODUCTION MODE (Windows ARM64)           |
|      Apple Silicon iMac / M-Series SoC      |   |       HP Snapdragon X Elite / Hexagon NPU       |
+---------------------------------------------+   +-------------------------------------------------+
|                                             |   |                                                 |
| 1. Vision Capture (30 FPS):                 |   | 1. Vision Capture (30 FPS):                     |
|    - MediaPipe Tasks Vision (WebAssembly)   |   |    - MediaPipe Tasks Vision (WebAssembly)       |
|    - Normalization -> [30, 258] Tensor      |   |    - Normalization -> [30, 258] Tensor          |
|                                             |   |                                                 |
| 2. Sign Classifier:                         |   | 2. Sign Classifier:                             |
|    - ONNX Runtime CoreMLExecutionProvider   |   |    - ONNX Runtime QNNExecutionProvider          |
|    - Apple Neural Engine / Metal GPU        |   |    - Qualcomm Hexagon NPU Context Binary        |
|    - Latency: ~11.4 ms                      |   |    - Latency: ~4.82 ms                          |
|                                             |   |                                                 |
| 3. NLP Sentence Reconstruction:             |   | 3. NLP Sentence Reconstruction:                 |
|    - Local Qwen2.5/Qwen3-Instruct / Grammar |   |    - Qualcomm GenieX / QAIRT Runtime            |
|    - Rule-Based SOV -> SVO ISL Engine       |   |    - Quantized INT4/INT8 Edge LLM               |
|                                             |   |                                                 |
| 4. Speech Synthesis (TTS):                  |   | 4. Speech Synthesis (TTS):                      |
|    - MeloTTS / High-Speed PCM16 WAV Synth   |   |    - MeloTTS NPU Engine / Windows SAPI Audio    |
|                                             |   |                                                 |
| 5. Client Presentation:                     |   | 5. Client Presentation:                         |
|    - Tauri 2 + React 18 + Tailwind CSS      |   |    - Tauri 2 Native Windows ARM64 Binary        |
+---------------------------------------------+   +-------------------------------------------------+
```

---

## 🛠️ Complete Technology Stack

| Layer | Component | Technology / Library | Role & Function |
| :--- | :--- | :--- | :--- |
| **Desktop Shell** | Native Wrapper | **Tauri 2 (Rust)** | High-efficiency native OS window container with minimal memory overhead (< 45 MB). |
| **User Interface** | Frontend Core | **React 18, TypeScript, Tailwind CSS** | Enterprise HP Accessibility UI (Slate-950, high-contrast focus rings, WCAG AAA). |
| **State & Stream** | State Store | **Zustand, Lucide React** | Real-time WebSocket management, sliding 30-frame buffer, debouncing, telemetry. |
| **Edge Vision** | Landmarks | **MediaPipe Tasks Vision (WASM)** | In-browser 30 FPS extraction of 33 pose landmarks and 21 landmarks per hand. |
| **Backend API** | App Server | **FastAPI, Uvicorn, Python 3.13** | Local REST endpoints (`/health`, `/translate/sequence`, `/conversation/speak`) and WebSocket (`/translate/live`). |
| **Deep Learning** | Classifier | **PyTorch 2.8, Lightweight Temporal Transformer** | Sequence classification across 258 spatial coordinates over 30 temporal frames. |
| **Inference Engine**| Cross-Platform | **ONNX Runtime, CoreML, Qualcomm QNN** | Hardware-adaptive execution provider prioritizing Hexagon NPU, Apple Neural Engine, or CPU. |
| **Language NLP** | Reconstruction | **Qwen2.5 / Qwen3-Instruct & Heuristics** | Translates raw ISL gloss arrays (`["I", "WATER", "NEED"]`) into fluent English (`"I need water."`). |
| **Voice Synthesis**| Audio | **MeloTTS / Local WAV Synth** | Converts finalized English sentences into high-clarity 16-bit PCM WAV audio streams. |
| **NPU Toolchain** | Deployment | **Qualcomm AI Hub (`qai_hub`)** | Cloud compilation and latency profiling targeting the Snapdragon X Elite CRD. |

---

## 🔬 Federated Dataset Architecture (The Modality Mismatch Solution)

Indian Sign Language research datasets are notoriously fragmented across varied formats, camera angles, resolutions, and annotation conventions. SignBridge solves this through a **Federated ConcatDataset pipeline** (`training/datasets/dataset.py`) harmonizing 5 structurally distinct data sources:

```
                            +---------------------------------------+
                            |   Unified MasterGlossMap Vocabulary   |
                            +-------------------+-------------------+
                                                |
            +-------------------+---------------+---------------+-------------------+
            |                   |               |               |                   |
      +-----+-----+       +-----+-----+   +-----+-----+   +-----+-----+       +-----+-----+
      | BridgeConn|       |  INCLUDE  |   |   CISLR   |   |  Mendeley |       |   Kaggle  |
      | WebDataset|       |  Nested   |   |    CSV    |   | JSON Map  |       | JSON Map  |
      +-----+-----+       +-----+-----+   +-----+-----+   +-----+-----+       +-----+-----+
            |                   |               |               |                   |
            +-------------------+---------------+---------------+-------------------+
                                                |
                                                v
                            +---------------------------------------+
                            |     Uniform Feature Normalizer        |
                            |       Shape: [30 Frames, 258 Dim]     |
                            +---------------------------------------+
```

1. **Bridge Connectivity Sign Dictionary**: Streams `.tar` shards via WebDataset, extracting pre-computed `pose-mediapipe.pose` arrays.
2. **INCLUDE Dataset (IIIT Hyderabad)**: Ingests raw `.MOV`/`.mp4` nested directories, processing each frame through MediaPipe Holistic.
3. **CISLR Corpus**: Dynamically maps `.mp4` video files to gloss labels via `dataset.csv`.
4. **Mendeley Data ISL**: Ingests varied lighting clips and aligns directory names via custom JSON mapping.
5. **Kaggle ISL (harsh0239 & arvindvinod)**: Ingests temporal gesture clips with standardized dictionary alignment.

### Crucial Feature Guarantee: `[30, 258]`
- **Spatial Coordinates**: 33 Pose landmarks $\times$ 4 values $(x, y, z, \text{visibility}) = 132$, Left Hand (21 $\times$ 3 = 63), Right Hand (21 $\times$ 3 = 63). Total per frame: **258 dimensions**.
- **Temporal Normalization**:
  - $\text{Frames} < 30$: Zero-padded at the end to 30 frames.
  - $\text{Frames} > 30$: Uniformly resampled across the temporal index (`np.linspace(0, N-1, 30)`).

---

## ⚡ Lightweight Temporal Transformer Architecture

Input: `[Batch, 30, 258]` $\rightarrow$ Output: `[Batch, Num_Classes]`

- **Linear Projection**: Projects 258 spatial coordinates to hidden dimension $d_{\text{model}} = 128$.
- **Positional Encoding**: Sinusoidal positional encoding injecting sequence order over the 30-frame window.
- **Transformer Encoder**: 4 stacked `TransformerEncoderLayer` blocks ($d_{\text{model}}=128$, $\text{heads}=8$, $d_{\text{ff}}=512$, $\text{dropout}=0.1$, GELU activation).
- **Temporal Pooling**: Global Average Pooling across the temporal dimension.
- **Classifier Head**: LayerNorm $\rightarrow$ Linear(128, 128) $\rightarrow$ GELU $\rightarrow$ Linear(128, Num_Classes).

---

## 🚀 Quickstart: Running Locally on macOS (Dev Mode)

### 1. Prerequisites
- **Node.js**: v18+ (tested on Node v20/v26)
- **Python**: 3.10+ (tested on Python 3.9/3.13)
- **Rust & Cargo**: Standard toolchain (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)

### 2. Start the FastAPI Backend AI Engine
```bash
# Clone the repository
git clone https://github.com/your-org/signbridge.git
cd signbridge

# Activate or create Python virtual environment
python3 -m venv apps/backend/.venv
source apps/backend/.venv/bin/activate

# Install backend dependencies
pip install -r apps/backend/requirements.txt
pip install onnx onnxruntime httpx qai-hub

# Launch the FastAPI engine (Port 8000)
PYTHONPATH=. uvicorn apps.backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Verify backend health at: `http://localhost:8000/health`

### 3. Launch the Desktop Application (Tauri 2 / React)
Open a second terminal window:
```bash
cd apps/desktop

# Install frontend dependencies
npm install

# Run in Browser Development Mode (Vite Port 1420)
npm run dev

# Or launch the Native Tauri Desktop Window
npm run tauri dev
```

---

## 💻 Deployment on Windows 11 ARM64 (Qualcomm Snapdragon)

For deployment on native HP Snapdragon Windows 11 ARM64 hardware (featuring the **Qualcomm Hexagon NPU**):

### 1. Automated One-Click Bootstrap
Open PowerShell as Administrator on the target Snapdragon laptop and run:
```powershell
Set-ExecutionPolicy RemoteSigned -Scope Process
.\deployment\windows-arm64\setup_snapdragon.ps1
```
This automated script:
1. Detects and installs native ARM64 Node.js and Python via `winget`.
2. Creates the Python virtual environment and installs `onnxruntime-qnn` targeting the Hexagon NPU.
3. Installs frontend dependencies in `apps/desktop`.
4. Outputs: `SignBridge Snapdragon Environment Ready. Run 'npm run tauri build' to compile the native ARM64 executable.`

### 2. Compile Native ARM64 Production Binary
```powershell
cd apps\desktop
npm run tauri build
```
The compiled native executable will be available in:
`apps/desktop/src-tauri/target/release/SignBridge.exe`

---

## ☁️ Qualcomm AI Hub Compilation & Profiling

To compile and profile the SignBridge ONNX model directly on Qualcomm cloud test devices:

```bash
# Set your Qualcomm AI Hub API Token
export QAI_HUB_API_TOKEN="your_token_from_app.aihub.qualcomm.com"

# Run automated compilation, profiling, and binary download
python scripts/qualcomm_hub_pipeline.py --device "Snapdragon X Elite CRD"
```

### Qualcomm Hexagon NPU Benchmarks
- **Target Device**: Snapdragon X Elite CRD
- **Inference Latency**: **4.82 ms** per 30-frame sequence
- **Throughput**: **207.4 inferences / second**
- **Peak NPU Memory**: **14.6 MB**
- **Compiled Asset**: `deployment/snapdragon/sign_transformer_npu.bin`

---

## 📡 API & WebSocket Reference

### `GET /health`
Returns system status, active ONNX execution provider, and loaded model metadata.

### `POST /translate/sequence`
Accepts a 30-frame sequence of 258 landmarks:
```json
{
  "sequence": [[0.1, 0.2, ...], ...]
}
```
Returns:
```json
{
  "gloss": "WATER",
  "confidence": 0.94,
  "class_id": 2,
  "reconstructed_sentence": "I need water."
}
```

### `POST /conversation/speak`
Converts input text into base64 WAV audio or direct audio stream.

### `WebSocket /translate/live`
Streams per-frame landmark arrays (`[258]`). Maintains a 30-frame sliding buffer, executes sign inference, debounces detected glosses, and streams real-time JSON responses:
```json
{
  "current_frame_landmarks_detected": true,
  "buffer_size": 30,
  "latest_gloss": "WATER",
  "confidence": 0.94,
  "sentence_accumulator": ["I", "WATER", "NEED"],
  "reconstructed_sentence": "I need water."
}
```

---

## 🛡️ Privacy & Security Design

- **Camera Data Never Leaves Device**: Frame capture and landmark extraction run strictly inside the client webview via WebAssembly.
- **No Remote Telemetry**: Zero analytics, zero cloud speech API calls.
- **Accessible Design**: High contrast visual palette, screen reader announcements (`aria-live="polite"`), and full keyboard navigation.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
