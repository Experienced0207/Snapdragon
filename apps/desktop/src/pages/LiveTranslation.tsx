import React, { useRef, useEffect } from "react";
import {
  Camera,
  CameraOff,
  Volume2,
  Trash2,
  Sparkles,
  Zap,
  Activity,
  Layers,
  Cpu,
} from "lucide-react";
import { useSignStore } from "../store/useSignStore";
import { useVisionPipeline } from "../hooks/useVisionPipeline";

export const LiveTranslation: React.FC = () => {
  const {
    reconstructedSentence,
    sentenceAccumulator,
    currentGloss,
    confidence,
    landmarksDetected,
    isSpeaking,
    speakCurrentSentence,
    clearAccumulator,
  } = useSignStore();

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const {
    isWasmReady,
    isDetecting,
    cameraError,
    startCamera,
    stopCamera,
    toggleCamera,
  } = useVisionPipeline();

  // Auto-start webcam when element mounts
  useEffect(() => {
    if (videoRef.current && !isDetecting) {
      startCamera(videoRef.current, canvasRef.current || undefined);
    }
  }, [startCamera]);

  const confidencePercent = Math.round(confidence * 100);

  return (
    <div className="h-full flex flex-col gap-6 max-w-7xl mx-auto">
      {/* Top Banner / Guidance */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-slate-100 tracking-tight flex items-center gap-2">
            <span>Live Sign Translation</span>
            <span className="text-xs px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-mono">
              REAL-TIME ISL
            </span>
            {isWasmReady && (
              <span className="text-xs px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono flex items-center gap-1">
                <Cpu className="w-3 h-3" />
                <span>MediaPipe WASM</span>
              </span>
            )}
          </h2>
          <p className="text-sm text-slate-400">
            Sign naturally within the camera reticle. Gestures are tracked at 30 FPS, mapped to ISL gloss tokens, and reconstructed into fluent English.
          </p>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2.5">
          <button
            onClick={() => {
              if (videoRef.current) {
                toggleCamera(videoRef.current, canvasRef.current || undefined);
              }
            }}
            className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm font-medium border transition-all ${
              isDetecting
                ? "bg-slate-800 border-slate-700 text-slate-200 hover:bg-slate-700"
                : "bg-rose-500/10 border-rose-500/30 text-rose-300 hover:bg-rose-500/20"
            }`}
          >
            {isDetecting ? <Camera className="w-4 h-4 text-cyan-400" /> : <CameraOff className="w-4 h-4 text-rose-400" />}
            <span>{isDetecting ? "Pause Feed" : "Resume Camera"}</span>
          </button>

          <button
            onClick={clearAccumulator}
            className="flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm font-medium bg-slate-800 border border-slate-700 text-slate-300 hover:text-white hover:bg-slate-700 transition-all"
            title="Clear current transcription buffer"
          >
            <Trash2 className="w-4 h-4 text-slate-400" />
            <span>Clear</span>
          </button>

          <button
            onClick={speakCurrentSentence}
            disabled={isSpeaking || (!reconstructedSentence && !currentGloss)}
            className="flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-bold bg-cyan-500 hover:bg-cyan-400 text-slate-950 shadow-lg shadow-cyan-500/20 active:scale-95 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <Volume2 className={`w-4 h-4 ${isSpeaking ? "animate-spin" : ""}`} />
            <span>{isSpeaking ? "Speaking..." : "Speak Translation"}</span>
          </button>
        </div>
      </div>

      {/* Main Translation Split Viewport */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1 min-h-0">
        {/* Left Column: Camera Viewport (7 Cols) */}
        <div className="lg:col-span-7 flex flex-col bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl relative">
          <div className="relative flex-1 bg-black flex items-center justify-center min-h-[380px]">
            {/* HTML5 Video Element */}
            <video
              ref={videoRef}
              playsInline
              muted
              className={`w-full h-full object-cover transform -scale-x-100 ${
                !isDetecting ? "opacity-20 filter grayscale" : ""
              }`}
            />

            {/* Canvas Overlay for Skeleton / Bounding Reticles */}
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full pointer-events-none transform -scale-x-100"
            />

            {/* Camera Inactive / Error Overlay */}
            {(!isDetecting || cameraError) && (
              <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center bg-slate-950/80 backdrop-blur-sm">
                <CameraOff className="w-12 h-12 text-slate-500 mb-3" />
                <p className="text-base font-semibold text-slate-200">
                  {cameraError || "Camera feed paused"}
                </p>
                <p className="text-xs text-slate-400 mt-1 max-w-sm">
                  {cameraError
                    ? "Ensure camera permissions are granted in your browser or OS settings."
                    : "Click Resume Camera to activate live vision tracking."}
                </p>
                <button
                  onClick={() => {
                    if (videoRef.current) {
                      startCamera(videoRef.current, canvasRef.current || undefined);
                    }
                  }}
                  className="mt-4 px-4 py-2 bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs rounded-lg transition-all"
                >
                  Enable Camera
                </button>
              </div>
            )}

            {/* Live Detection Status Badge */}
            <div className="absolute top-4 left-4 flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-950/80 backdrop-blur-md border border-slate-800 text-xs text-slate-300">
              <span
                className={`w-2 h-2 rounded-full ${
                  landmarksDetected ? "bg-emerald-400 animate-ping" : "bg-cyan-400"
                }`}
              />
              <span className="font-mono text-[11px]">
                {landmarksDetected ? "Pose & Hands Extracted (258 features)" : "Scanning Vision Field..."}
              </span>
            </div>

            {/* Current Active Gloss Overlay Chip */}
            {currentGloss && (
              <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between px-4 py-3 rounded-xl bg-slate-950/90 backdrop-blur-md border border-cyan-500/40 shadow-xl">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-lg bg-cyan-500/20 border border-cyan-400 flex items-center justify-center text-cyan-400">
                    <Zap className="w-4 h-4" />
                  </div>
                  <div>
                    <div className="text-[10px] uppercase font-bold text-cyan-400 tracking-wider">
                      Current Predicted Sign
                    </div>
                    <div className="text-lg font-black uppercase text-white tracking-wide">
                      {currentGloss}
                    </div>
                  </div>
                </div>

                <div className="text-right">
                  <div className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">
                    Model Confidence
                  </div>
                  <div className="text-base font-mono font-bold text-cyan-300">
                    {confidencePercent}%
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: High-Contrast Transcription Deck (5 Cols) */}
        <div className="lg:col-span-5 flex flex-col gap-4">
          {/* Primary Reconstructed Sentence Card */}
          <div
            className="flex-1 bg-slate-900 border-2 border-slate-700/80 rounded-2xl p-6 shadow-xl flex flex-col justify-between"
            role="region"
            aria-label="Reconstructed Translation"
          >
            <div>
              <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-800">
                <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-400">
                  <Sparkles className="w-4 h-4 text-cyan-400" />
                  <span>Reconstructed English Sentence</span>
                </div>
                <span className="text-[11px] font-mono text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">
                  NLP Phase 2
                </span>
              </div>

              {/* Accessible High-Contrast Large Text Area */}
              <div
                className="min-h-[140px] flex items-center"
                aria-live="polite"
                aria-atomic="true"
              >
                {reconstructedSentence ? (
                  <p className="text-3xl font-extrabold text-white tracking-tight leading-snug">
                    "{reconstructedSentence}"
                  </p>
                ) : (
                  <div className="flex flex-col items-center justify-center w-full py-8 text-center text-slate-500">
                    <Layers className="w-8 h-8 mb-2 opacity-50" />
                    <p className="text-sm font-medium">Awaiting sign gesture sequence...</p>
                    <p className="text-xs text-slate-600 mt-0.5">
                      Signs will accumulate and automatically form complete sentences.
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Confidence Bar Meter */}
            <div className="pt-4 border-t border-slate-800/80 space-y-2">
              <div className="flex justify-between items-center text-xs">
                <span className="text-slate-400 font-medium">Prediction Certainty</span>
                <span className="font-mono font-bold text-slate-200">{confidencePercent}%</span>
              </div>
              <div className="w-full h-2.5 bg-slate-800 rounded-full overflow-hidden p-0.5 border border-slate-700/50">
                <div
                  className={`h-full rounded-full transition-all duration-300 ${
                    confidencePercent >= 70
                      ? "bg-emerald-400"
                      : confidencePercent >= 45
                      ? "bg-cyan-400"
                      : "bg-amber-400"
                  }`}
                  style={{ width: `${Math.max(4, confidencePercent)}%` }}
                />
              </div>
            </div>
          </div>

          {/* Raw ISL Gloss Token Stream Card */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
            <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-800">
              <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-400">
                <Activity className="w-3.5 h-3.5 text-cyan-400" />
                <span>Raw ISL Gloss Stream</span>
              </div>
              <span className="text-[11px] font-mono text-slate-400">
                {sentenceAccumulator.length} Tokens
              </span>
            </div>

            {/* Gloss Pills Stream */}
            <div className="flex flex-wrap gap-2 min-h-[52px] items-center p-2 rounded-xl bg-slate-950/60 border border-slate-800/60">
              {sentenceAccumulator.length > 0 ? (
                sentenceAccumulator.map((gloss, idx) => (
                  <span
                    key={`${gloss}-${idx}`}
                    className="inline-flex items-center gap-1 px-3 py-1 rounded-lg bg-slate-800 border border-cyan-500/30 text-xs font-mono font-bold text-cyan-300 uppercase tracking-wide shadow-sm"
                  >
                    <span>{gloss}</span>
                    <span className="text-[9px] text-slate-500 font-normal">#{idx + 1}</span>
                  </span>
                ))
              ) : (
                <span className="text-xs text-slate-500 italic px-1">
                  No active gloss tokens accumulated yet.
                </span>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
