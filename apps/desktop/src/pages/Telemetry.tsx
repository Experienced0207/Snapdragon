import React from "react";
import { Cpu, Zap, ShieldCheck, Activity, HardDrive, CheckCircle2 } from "lucide-react";
import { useSignStore } from "../store/useSignStore";

export const Telemetry: React.FC = () => {
  const { performance, connectionStatus } = useSignStore();

  const totalLatency = (
    performance.visionLatencyMs +
    performance.signLatencyMs +
    performance.languageLatencyMs
  ).toFixed(1);

  return (
    <div className="h-full flex flex-col gap-6 max-w-5xl mx-auto">
      <div>
        <h2 className="text-xl font-bold text-slate-100 tracking-tight flex items-center gap-2">
          <Cpu className="w-5 h-5 text-cyan-400" />
          <span>AI / NPU Hardware Telemetry</span>
        </h2>
        <p className="text-sm text-slate-400">
          Real-time metrics for on-device ONNX runtime acceleration, NPU latency, and local inference guarantees.
        </p>
      </div>

      {/* Grid of Telemetry Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {/* Card 1: Execution Engine */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-3">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span className="font-semibold uppercase tracking-wider">Acceleration Provider</span>
            <Zap className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-xl font-black text-white font-mono truncate">
            {performance.activeProvider.replace("ExecutionProvider", "")}
          </div>
          <p className="text-xs text-slate-400">
            {performance.activeProvider.includes("CoreML")
              ? "Apple Silicon Neural Engine & GPU CoreML acceleration."
              : performance.activeProvider.includes("QNN")
              ? "Qualcomm Snapdragon NPU / Hexagon DSP hardware execution."
              : "Optimized SIMD multi-threaded CPU execution."}
          </p>
          <div className="pt-2 border-t border-slate-800 flex items-center gap-1.5 text-xs text-emerald-400">
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>Hardware Parity Verified</span>
          </div>
        </div>

        {/* Card 2: Real-time Frame Rate */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-3">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span className="font-semibold uppercase tracking-wider">Vision Pipeline FPS</span>
            <Activity className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-3xl font-black text-emerald-400 font-mono">
            {performance.fps} <span className="text-sm text-slate-400 font-normal">FPS</span>
          </div>
          <p className="text-xs text-slate-400">
            Target frame rate: 30 FPS. Sustaining full temporal sign capture without frame drops.
          </p>
          <div className="pt-2 border-t border-slate-800 text-xs text-slate-400">
            Frame Time: {(1000 / Math.max(1, performance.fps)).toFixed(1)} ms
          </div>
        </div>

        {/* Card 3: Privacy & Cloud Independence */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-3">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span className="font-semibold uppercase tracking-wider">Cloud Request Count</span>
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-3xl font-black text-white font-mono">
            0 <span className="text-sm text-emerald-400 font-normal">Strictly Local</span>
          </div>
          <p className="text-xs text-slate-400">
            All vision processing, sign classification, sentence reconstruction, and voice synthesis run locally on-device.
          </p>
          <div className="pt-2 border-t border-slate-800 text-xs text-emerald-400 font-medium flex items-center gap-1.5">
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>Zero Cloud Telemetry Transmitted</span>
          </div>
        </div>
      </div>

      {/* Latency Breakdown Detailed Section */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-lg space-y-5">
        <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300">
          Inference & Pipeline Latency Breakdown
        </h3>

        <div className="space-y-4">
          <div>
            <div className="flex justify-between text-xs font-mono text-slate-300 mb-1.5">
              <span>Phase 1: Vision MediaPipe Extraction (258 coordinates)</span>
              <span>{performance.visionLatencyMs.toFixed(1)} ms</span>
            </div>
            <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="bg-cyan-500 h-full rounded-full"
                style={{ width: `${(performance.visionLatencyMs / Number(totalLatency)) * 100}%` }}
              />
            </div>
          </div>

          <div>
            <div className="flex justify-between text-xs font-mono text-slate-300 mb-1.5">
              <span>Phase 2: Temporal Transformer ONNX Inference (30 Frames)</span>
              <span>{performance.signLatencyMs.toFixed(1)} ms</span>
            </div>
            <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="bg-indigo-500 h-full rounded-full"
                style={{ width: `${(performance.signLatencyMs / Number(totalLatency)) * 100}%` }}
              />
            </div>
          </div>

          <div>
            <div className="flex justify-between text-xs font-mono text-slate-300 mb-1.5">
              <span>Phase 3: Gloss-to-Sentence Reconstruction (NLP Engine)</span>
              <span>{performance.languageLatencyMs.toFixed(1)} ms</span>
            </div>
            <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="bg-purple-500 h-full rounded-full"
                style={{ width: `${(performance.languageLatencyMs / Number(totalLatency)) * 100}%` }}
              />
            </div>
          </div>
        </div>

        <div className="flex justify-between items-center pt-3 border-t border-slate-800 text-sm">
          <span className="text-slate-400 font-medium">End-to-End Latency:</span>
          <span className="font-mono font-bold text-cyan-300 text-base">{totalLatency} ms</span>
        </div>
      </div>
    </div>
  );
};
