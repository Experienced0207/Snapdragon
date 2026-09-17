/**
 * Client-Side MediaPipe WebAssembly Vision Pipeline Hook for SignBridge.
 *
 * Initializes `@mediapipe/tasks-vision` via WebAssembly, handles user webcam capture,
 * extracts 33 Pose landmarks (132 values), 21 Left Hand landmarks (63 values),
 * and 21 Right Hand landmarks (63 values), flattening them into a uniform 258-element
 * Float32Array [30, 258] and pushing them in real-time over the WebSocket.
 */

import { useRef, useState, useCallback, useEffect } from "react";
import {
  FilesetResolver,
  PoseLandmarker,
  HandLandmarker,
  DrawingUtils,
} from "@mediapipe/tasks-vision";
import { useSignStore } from "../store/useSignStore";

export interface VisionPipelineOptions {
  wasmUrl?: string;
  poseModelUrl?: string;
  handModelUrl?: string;
  targetFps?: number;
}

export interface UseVisionPipelineReturn {
  isWasmReady: boolean;
  isDetecting: boolean;
  cameraError: string | null;
  startCamera: (video: HTMLVideoElement, canvas?: HTMLCanvasElement) => Promise<void>;
  stopCamera: () => void;
  toggleCamera: (video: HTMLVideoElement, canvas?: HTMLCanvasElement) => Promise<void>;
}

const DEFAULT_WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const DEFAULT_POSE_MODEL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const DEFAULT_HAND_MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

export function useVisionPipeline(options: VisionPipelineOptions = {}): UseVisionPipelineReturn {
  const {
    wasmUrl = DEFAULT_WASM_URL,
    poseModelUrl = DEFAULT_POSE_MODEL,
    handModelUrl = DEFAULT_HAND_MODEL,
  } = options;

  const { sendLandmarkFrame, updatePerformance } = useSignStore();

  const [isWasmReady, setIsWasmReady] = useState<boolean>(false);
  const [isDetecting, setIsDetecting] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  // References to MediaPipe instances & media streams
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameId = useRef<number | null>(null);
  const isRunningRef = useRef<boolean>(false);

  // FPS & Performance tracking
  const lastVideoTime = useRef<number>(-1);
  const frameCount = useRef<number>(0);
  const lastFpsTimestamp = useRef<number>(performance.now());

  // 1. Initialize MediaPipe WebAssembly Vision Tasks
  useEffect(() => {
    let isCancelled = false;

    async function initMediaPipeWasm() {
      try {
        console.log("[useVisionPipeline] Loading MediaPipe WebAssembly assets from CDN...");
        const vision = await FilesetResolver.forVisionTasks(wasmUrl);

        if (isCancelled) return;

        // Initialize Pose Landmarker (33 landmarks)
        const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: poseModelUrl,
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numPoses: 1,
          minPoseDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        if (isCancelled) return;

        // Initialize Hand Landmarker (21 landmarks x 2 hands)
        const handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: handModelUrl,
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        if (isCancelled) return;

        poseLandmarkerRef.current = poseLandmarker;
        handLandmarkerRef.current = handLandmarker;
        setIsWasmReady(true);
        console.log("[useVisionPipeline] MediaPipe Tasks Vision (Pose + Hands) WASM initialized successfully!");
      } catch (err: any) {
        console.warn("[useVisionPipeline] MediaPipe WASM initialization error (falling back to vision bridge):", err);
        // Still mark as ready so synthetic / visual bridge can stream
        setIsWasmReady(true);
      }
    }

    initMediaPipeWasm();

    return () => {
      isCancelled = true;
      if (poseLandmarkerRef.current) {
        poseLandmarkerRef.current.close();
        poseLandmarkerRef.current = null;
      }
      if (handLandmarkerRef.current) {
        handLandmarkerRef.current.close();
        handLandmarkerRef.current = null;
      }
    };
  }, [wasmUrl, poseModelUrl, handModelUrl]);

  // 2. Flatten Pose & Hand landmarks into uniform 258-dim Float32Array
  const extractFlattened258Landmarks = useCallback(
    (
      poseResults: any,
      handResults: any
    ): Float32Array => {
      // Allocate exactly 258 float values (132 pose + 63 left hand + 63 right hand)
      const features = new Float32Array(258);

      // Section A: Pose Landmarks (33 points * 4 = 132 features)
      if (poseResults && poseResults.landmarks && poseResults.landmarks.length > 0) {
        const pose = poseResults.landmarks[0];
        for (let i = 0; i < 33 && i < pose.length; i++) {
          const pt = pose[i];
          const base = i * 4;
          features[base + 0] = pt.x ?? 0.0;
          features[base + 1] = pt.y ?? 0.0;
          features[base + 2] = pt.z ?? 0.0;
          features[base + 3] = pt.visibility ?? 1.0;
        }
      }

      // Section B: Left & Right Hands (21 points * 3 = 63 features each)
      const leftHandOffset = 132;
      const rightHandOffset = 132 + 63;

      if (handResults && handResults.landmarks && handResults.landmarks.length > 0) {
        const handednessList = handResults.handednesses || [];

        for (let h = 0; h < handResults.landmarks.length; h++) {
          const handLandmarks = handResults.landmarks[h];
          const handednessCategory = handednessList[h]?.[0]?.categoryName || (h === 0 ? "Left" : "Right");
          const isLeft = handednessCategory.toLowerCase().includes("left");
          const offset = isLeft ? leftHandOffset : rightHandOffset;

          for (let i = 0; i < 21 && i < handLandmarks.length; i++) {
            const pt = handLandmarks[i];
            const base = offset + i * 3;
            features[base + 0] = pt.x ?? 0.0;
            features[base + 1] = pt.y ?? 0.0;
            features[base + 2] = pt.z ?? 0.0;
          }
        }
      }

      return features;
    },
    []
  );

  // 3. Render Skeleton Feedback to Canvas
  const drawLandmarksOnCanvas = useCallback(
    (
      canvas: HTMLCanvasElement,
      video: HTMLVideoElement,
      poseResults: any,
      handResults: any
    ) => {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      if (canvas.width !== video.videoWidth && video.videoWidth > 0) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      ctx.save();
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const drawingUtils = new DrawingUtils(ctx);

      // Draw Pose Connections
      if (poseResults?.landmarks?.[0]) {
        drawingUtils.drawConnectors(
          poseResults.landmarks[0],
          PoseLandmarker.POSE_CONNECTIONS,
          { color: "#06b6d4", lineWidth: 2 }
        );
        drawingUtils.drawLandmarks(poseResults.landmarks[0], {
          color: "#38bdf8",
          lineWidth: 1,
          radius: 3,
        });
      }

      // Draw Hand Connections
      if (handResults?.landmarks) {
        for (const landmarks of handResults.landmarks) {
          drawingUtils.drawConnectors(
            landmarks,
            HandLandmarker.HAND_CONNECTIONS,
            { color: "#10b981", lineWidth: 2 }
          );
          drawingUtils.drawLandmarks(landmarks, {
            color: "#34d399",
            lineWidth: 1,
            radius: 3,
          });
        }
      }

      // Reticle Guide
      const cx = canvas.width / 2;
      const cy = canvas.height / 2;
      ctx.strokeStyle = "rgba(6, 182, 212, 0.4)";
      ctx.lineWidth = 2;
      ctx.strokeRect(cx - 140, cy - 140, 280, 280);

      ctx.restore();
    },
    []
  );

  // 4. Main RAF Detection Loop
  const runDetectionLoop = useCallback(() => {
    if (!isRunningRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const startTime = performance.now();

    if (video && video.readyState >= 2 && !video.paused) {
      const nowMs = performance.now();

      if (video.currentTime !== lastVideoTime.current) {
        lastVideoTime.current = video.currentTime;

        let poseResults = null;
        let handResults = null;

        // Perform Pose Detection
        if (poseLandmarkerRef.current) {
          try {
            poseResults = poseLandmarkerRef.current.detectForVideo(video, nowMs);
          } catch (e) {
            // Frame timing error handle
          }
        }

        // Perform Hand Detection
        if (handLandmarkerRef.current) {
          try {
            handResults = handLandmarkerRef.current.detectForVideo(video, nowMs);
          } catch (e) {
            // Frame timing error handle
          }
        }

        // Flatten features into exactly 258 float values
        const flattened258 = extractFlattened258Landmarks(poseResults, handResults);

        // Send landmark array to Zustand store / WebSocket
        sendLandmarkFrame(Array.from(flattened258));

        // Draw visual skeleton if canvas is attached
        if (canvas) {
          drawLandmarksOnCanvas(canvas, video, poseResults, handResults);
        }

        // Latency & FPS calculations
        const latency = performance.now() - startTime;
        frameCount.current += 1;
        if (nowMs - lastFpsTimestamp.current >= 1000) {
          const fps = frameCount.current;
          frameCount.current = 0;
          lastFpsTimestamp.current = nowMs;
          updatePerformance({
            fps,
            visionLatencyMs: Math.max(1.0, latency),
          });
        }
      }
    }

    if (isRunningRef.current) {
      animationFrameId.current = requestAnimationFrame(runDetectionLoop);
    }
  }, [extractFlattened258Landmarks, drawLandmarksOnCanvas, sendLandmarkFrame, updatePerformance]);

  // 5. Start Webcam Capture
  const startCamera = useCallback(
    async (video: HTMLVideoElement, canvas?: HTMLCanvasElement) => {
      try {
        setCameraError(null);
        videoRef.current = video;
        if (canvas) canvasRef.current = canvas;

        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            frameRate: { ideal: 30 },
            facingMode: "user",
          },
          audio: false,
        });

        streamRef.current = stream;
        video.srcObject = stream;
        await video.play();

        isRunningRef.current = true;
        setIsDetecting(true);
        animationFrameId.current = requestAnimationFrame(runDetectionLoop);
        console.log("[useVisionPipeline] Camera stream & detection loop active.");
      } catch (err: any) {
        console.error("[useVisionPipeline] Camera error:", err);
        setCameraError(
          err.name === "NotAllowedError"
            ? "Camera permission denied. Please allow camera access."
            : "No camera device detected. Running in synthetic mode."
        );
        setIsDetecting(false);
      }
    },
    [runDetectionLoop]
  );

  // 6. Stop Webcam Capture & Terminate Loop
  const stopCamera = useCallback(() => {
    isRunningRef.current = false;
    setIsDetecting(false);

    if (animationFrameId.current) {
      cancelAnimationFrame(animationFrameId.current);
      animationFrameId.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
    console.log("[useVisionPipeline] Camera stream stopped.");
  }, []);

  // 7. Toggle Camera
  const toggleCamera = useCallback(
    async (video: HTMLVideoElement, canvas?: HTMLCanvasElement) => {
      if (isDetecting) {
        stopCamera();
      } else {
        await startCamera(video, canvas);
      }
    },
    [isDetecting, startCamera, stopCamera]
  );

  // 8. Auto-cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  return {
    isWasmReady,
    isDetecting,
    cameraError,
    startCamera,
    stopCamera,
    toggleCamera,
  };
}
