import React, { useState } from "react";
import { Settings as SettingsIcon, Server, Camera, Volume2, Save, Check, RefreshCw } from "lucide-react";
import { useSignStore } from "../store/useSignStore";

export const Settings: React.FC = () => {
  const { backendUrl, setBackendUrl, connectionStatus, connectWebSocket } = useSignStore();
  const [urlInput, setUrlInput] = useState<string>(backendUrl);
  const [saved, setSaved] = useState<boolean>(false);
  const [speechRate, setSpeechRate] = useState<number>(1.0);
  const [operationalMode, setOperationalMode] = useState<string>("DEV_MODE");

  const handleSave = () => {
    setBackendUrl(urlInput);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="h-full flex flex-col gap-6 max-w-4xl mx-auto">
      <div>
        <h2 className="text-xl font-bold text-slate-100 tracking-tight flex items-center gap-2">
          <SettingsIcon className="w-5 h-5 text-cyan-400" />
          <span>System Settings & Configuration</span>
        </h2>
        <p className="text-sm text-slate-400">
          Configure local backend endpoints, camera settings, and speech synthesis parameters.
        </p>
      </div>

      <div className="space-y-6">
        {/* Backend Connectivity Card */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-lg space-y-4">
          <div className="flex items-center gap-2 pb-3 border-b border-slate-800 text-sm font-bold uppercase tracking-wider text-slate-300">
            <Server className="w-4 h-4 text-cyan-400" />
            <span>FastAPI Backend Connection</span>
          </div>

          <div className="space-y-2">
            <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Backend REST & WebSocket Endpoint
            </label>
            <div className="flex gap-3">
              <input
                type="text"
                value={urlInput}
                onChange={(e) => setUrlInput(e.target.value)}
                className="flex-1 bg-slate-950 border border-slate-800 focus:border-cyan-500 rounded-lg px-4 py-2 text-sm font-mono text-slate-100 outline-none transition-all"
                placeholder="http://localhost:8000"
              />
              <button
                onClick={handleSave}
                className="flex items-center gap-2 px-5 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-sm transition-all"
              >
                {saved ? <Check className="w-4 h-4" /> : <Save className="w-4 h-4" />}
                <span>{saved ? "Saved" : "Save"}</span>
              </button>
            </div>
            <p className="text-xs text-slate-500">
              Default is http://localhost:8000. WebSocket endpoint is automatically mapped to /translate/live.
            </p>
          </div>

          <div className="flex items-center justify-between pt-2 text-xs">
            <span className="text-slate-400">Current Status:</span>
            <div className="flex items-center gap-2">
              <span className="font-semibold capitalize text-slate-200">{connectionStatus}</span>
              <button
                onClick={() => connectWebSocket()}
                className="p-1 text-slate-400 hover:text-white"
                title="Reconnect"
              >
                <RefreshCw className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>

        {/* Operational Mode Card */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-lg space-y-4">
          <div className="flex items-center gap-2 pb-3 border-b border-slate-800 text-sm font-bold uppercase tracking-wider text-slate-300">
            <Camera className="w-4 h-4 text-cyan-400" />
            <span>Operational Architecture</span>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <button
              onClick={() => setOperationalMode("DEV_MODE")}
              className={`p-4 rounded-xl border text-left transition-all ${
                operationalMode === "DEV_MODE"
                  ? "bg-cyan-500/10 border-cyan-500/50 text-white"
                  : "bg-slate-950/40 border-slate-800 text-slate-400 hover:border-slate-700"
              }`}
            >
              <div className="font-bold text-sm text-slate-200">DEV_MODE (Local Mac)</div>
              <p className="text-xs text-slate-400 mt-1">
                Apple Silicon CoreML execution, local MediaPipe capture, and local HuggingFace / rule heuristics.
              </p>
            </button>

            <button
              onClick={() => setOperationalMode("SNAPDRAGON_MODE")}
              className={`p-4 rounded-xl border text-left transition-all ${
                operationalMode === "SNAPDRAGON_MODE"
                  ? "bg-cyan-500/10 border-cyan-500/50 text-white"
                  : "bg-slate-950/40 border-slate-800 text-slate-400 hover:border-slate-700"
              }`}
            >
              <div className="font-bold text-sm text-slate-200">SNAPDRAGON_MODE (ARM64)</div>
              <p className="text-xs text-slate-400 mt-1">
                Qualcomm QNN execution provider on Snapdragon NPU / Hexagon DSP with GenieX runtime.
              </p>
            </button>
          </div>
        </div>

        {/* Speech Synthesis Voice Settings */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-lg space-y-4">
          <div className="flex items-center gap-2 pb-3 border-b border-slate-800 text-sm font-bold uppercase tracking-wider text-slate-300">
            <Volume2 className="w-4 h-4 text-cyan-400" />
            <span>Voice & Audio Settings</span>
          </div>

          <div className="space-y-3">
            <div className="flex justify-between items-center text-xs">
              <span className="text-slate-400">Speech Rate</span>
              <span className="font-mono font-bold text-slate-200">{speechRate}x</span>
            </div>
            <input
              type="range"
              min="0.5"
              max="2.0"
              step="0.1"
              value={speechRate}
              onChange={(e) => setSpeechRate(parseFloat(e.target.value))}
              className="w-full accent-cyan-500"
            />
          </div>
        </div>
      </div>
    </div>
  );
};
