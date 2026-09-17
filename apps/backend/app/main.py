"""
SignBridge FastAPI Production Backend Application.
Provides REST and WebSocket API endpoints for real-time ISL sign translation,
gloss-to-sentence language reconstruction, and text-to-speech audio synthesis.
"""

import os
import sys
import base64
import numpy as np
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from apps.backend.app.signing.classifier import SignClassifier
from apps.backend.app.language.reconstructor import GlossToSentenceEngine
from apps.backend.app.speech.tts import TextToSpeechService

# Global service instances
classifier_service: Optional[SignClassifier] = None
reconstructor_service: Optional[GlossToSentenceEngine] = None
tts_service: Optional[TextToSpeechService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler initializing backend ML & NLP services."""
    global classifier_service, reconstructor_service, tts_service
    print("🚀 Starting SignBridge Backend Services...")

    # Initialize ONNX Sign Classifier
    try:
        classifier_service = SignClassifier()
        print(f"  ✓ SignClassifier online (Provider: {classifier_service.get_selected_provider()})")
    except Exception as e:
        print(f"  ⚠️ SignClassifier warning: {e}")
        classifier_service = None

    # Initialize Gloss-to-Sentence Reconstruction Engine
    try:
        reconstructor_service = GlossToSentenceEngine(mode="DEV_MODE")
        print("  ✓ GlossToSentenceEngine online")
    except Exception as e:
        print(f"  ⚠️ GlossToSentenceEngine warning: {e}")
        reconstructor_service = GlossToSentenceEngine(mode="DEV_MODE")

    # Initialize Text-to-Speech Service
    try:
        tts_service = TextToSpeechService()
        print("  ✓ TextToSpeechService online")
    except Exception as e:
        print(f"  ⚠️ TextToSpeechService warning: {e}")
        tts_service = TextToSpeechService()

    yield

    print("🛑 Shutting down SignBridge Backend Services.")


# Initialize FastAPI Application
app = FastAPI(
    title="SignBridge Backend API",
    description="Local AI Infrastructure for Isolated Indian Sign Language (ISL) Recognition & Translation",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for Tauri Desktop / Web Frontend
origins = [
    "http://localhost:1420",
    "http://127.0.0.1:1420",
    "tauri://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request / Response Pydantic Schemas ---
class SequenceRequest(BaseModel):
    sequence: List[List[float]] = Field(
        ...,
        description="30-frame temporal sequence of 258 spatial landmark coordinates [30, 258]"
    )

class SequenceResponse(BaseModel):
    gloss: str
    confidence: float
    class_id: int
    reconstructed_sentence: str

class SpeakRequest(BaseModel):
    text: str = Field(..., description="Text content to synthesize into spoken audio")

class SpeakResponse(BaseModel):
    audio_base64: str
    media_type: str = "audio/wav"


# --- REST API Endpoints ---

@app.get("/health", summary="Check backend health & model provider status")
async def health_check() -> Dict[str, Any]:
    provider = classifier_service.get_selected_provider() if classifier_service else "Unavailable"
    num_classes = len(classifier_service.gloss_map) if classifier_service else 0

    return {
        "status": "healthy",
        "app_name": "SignBridge Backend",
        "version": "1.0.0",
        "execution_provider": provider,
        "classifier_loaded": classifier_service is not None,
        "language_engine_loaded": reconstructor_service is not None,
        "tts_service_loaded": tts_service is not None,
        "num_vocabulary_classes": num_classes
    }


@app.post("/translate/sequence", response_model=SequenceResponse, summary="Batch sign inference on [30, 258] landmarks")
async def translate_sequence(request: SequenceRequest):
    if classifier_service is None:
        raise HTTPException(status_code=500, detail="SignClassifier service is not initialized.")

    seq_np = np.array(request.sequence, dtype=np.float32)
    if seq_np.shape != (30, 258):
        # Apply temporal padding/resampling if necessary
        from training.preprocessing.mediapipe_utils import temporal_resample_or_pad
        seq_np = temporal_resample_or_pad(seq_np, target_len=30)

    # 1. Run Sign Classifier
    res = classifier_service.predict(seq_np)
    predicted_gloss = res["gloss"]

    # 2. Reconstruct Sentence
    sentence = reconstructor_service.reconstruct([predicted_gloss]) if reconstructor_service else predicted_gloss

    return SequenceResponse(
        gloss=predicted_gloss,
        confidence=res["confidence"],
        class_id=res["class_id"],
        reconstructed_sentence=sentence
    )


@app.post("/conversation/speak", summary="Synthesize text into WAV audio")
async def speak_text(request: SpeakRequest, direct_wav: bool = False):
    if tts_service is None:
        raise HTTPException(status_code=500, detail="TextToSpeech service is not initialized.")

    wav_bytes = tts_service.synthesize(request.text)

    if direct_wav:
        return Response(content=wav_bytes, media_type="audio/wav")

    b64_str = base64.b64encode(wav_bytes).decode('utf-8')
    return SpeakResponse(audio_base64=b64_str)


# --- WebSocket Streaming Inference ---

@app.websocket("/translate/live")
async def translate_live_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video stream landmark processing.
    Maintains a 30-frame sliding buffer per client connection, executes sign inference
    on rolling sequences, accumulates stable glosses, and streams real-time JSON responses.
    """
    await websocket.accept()
    print("🔌 WebSocket client connected to /translate/live")

    frame_buffer: List[List[float]] = []
    sentence_accumulator: List[str] = []
    last_detected_gloss: str = ""
    debounce_counter: int = 0

    try:
        while True:
            data = await websocket.receive_json()

            # Extract single frame landmarks [258] or frame array
            if "landmarks" in data:
                frame_feats = data["landmarks"]
            elif "sequence" in data:
                frame_feats = data["sequence"]
            elif isinstance(data, list):
                frame_feats = data
            else:
                frame_feats = data.get("data", [])

            landmarks_detected = len(frame_feats) == 258 and any(v != 0 for v in frame_feats)

            if len(frame_feats) == 258:
                frame_buffer.append(frame_feats)
                # Keep sliding buffer to 30 frames max
                if len(frame_buffer) > 30:
                    frame_buffer.pop(0)

            latest_gloss = ""
            confidence = 0.0

            # Execute sign inference when buffer has 30 frames
            if len(frame_buffer) == 30 and classifier_service is not None:
                seq_arr = np.array(frame_buffer, dtype=np.float32)
                pred = classifier_service.predict(seq_arr)
                latest_gloss = pred["gloss"]
                confidence = pred["confidence"]

                # Debouncing logic: require confidence > 0.45 before adding to sentence accumulator
                if confidence >= 0.45 and latest_gloss != "unknown":
                    if latest_gloss != last_detected_gloss:
                        debounce_counter += 1
                        if debounce_counter >= 2:  # 2 consecutive matches
                            sentence_accumulator.append(latest_gloss)
                            last_detected_gloss = latest_gloss
                            debounce_counter = 0

            # Reconstruct sentence from accumulated glosses
            reconstructed_sentence = ""
            if reconstructor_service and sentence_accumulator:
                reconstructed_sentence = reconstructor_service.reconstruct(sentence_accumulator)

            response_payload = {
                "current_frame_landmarks_detected": landmarks_detected,
                "buffer_size": len(frame_buffer),
                "latest_gloss": latest_gloss,
                "confidence": confidence,
                "sentence_accumulator": sentence_accumulator,
                "reconstructed_sentence": reconstructed_sentence
            }

            await websocket.send_json(response_payload)

    except WebSocketDisconnect:
        print("🔌 WebSocket client disconnected.")
    except Exception as e:
        print(f"⚠️ WebSocket error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
