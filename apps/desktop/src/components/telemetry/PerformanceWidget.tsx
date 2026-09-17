import React from "react";
import { Activity, Cpu, ShieldCheck, Zap } from "lucide-react";
import { useSignStore } from "../../store/useSignStore";

export const PerformanceWidget: React.FC = () => {
  const { performance, connectionStatus } = useSignStore();
  const totalLatency = (
    performance.visionLatencyMs +
    performance.signLatencyMs +
    performance.languageLatencyMs
  ).toFixed(1);

  return (
    <div className="bg-slate-900/95 border border-slate-800 rounded-xl p-3.5 shadow-lg backdrop-blur-md">
      <div className="flex items-center justify-between pb-2.5 mb-2.5 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-bold uppercase tracking-wider text-slate-300">
            Engine Telemetry
          </span>
        </div>
        <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-[11px] font-semibold text-emerald-400">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
          <span>{performance.fps} FPS</span>
        </div>
      </div>

      {/* Latency Breakdown Bar */}
      <div className="space-y-2 mb-3">
        <div className="flex justify-between text-[11px] text-slate-400 font-mono">
          <span>Vision: {performance.visionLatencyMs.toFixed(1)}ms</span>
          <span>Sign: {performance.signLatencyMs.toFixed(1)}ms</span>
          <span>NLP: {performance.languageLatencyMs.toFixed(1)}ms</span>
        </div>

        {/* Proportional Segmented Progress Bar */}
        <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden flex">
          <div
            className="bg-cyan-500 h-full transition-all duration-300"
            style={{ width: `${(performance.visionLatencyMs / Number(totalLatency)) * 100}%` }}
            title="Vision Preprocessing Latency"
          />
          <div
            className="bg-indigo-500 h-full transition-all duration-300"
            style={{ width: `${(performance.signLatencyMs / Number(totalLatency)) * 100}%` }}
            title="Transformer Sign Classifier Latency"
          />
          <div
            className="bg-purple-500 h-full transition-all duration-300"
            style={{ width: `${(performance.languageLatencyMs / Number(totalLatency)) * 100}%` }}
            title="Language Reconstruction Latency"
          />
        </div>

        <div className="flex justify-between items-center pt-0.5 text-xs">
          <span className="text-slate-400">Total Loop Latency</span>
          <span className="font-mono font-bold text-slate-200">{totalLatency} ms</span>
        </div>
      </div>

      {/* Execution Provider & Privacy Footer */}
      <div className="grid grid-cols-2 gap-2 pt-2 border-t border-slate-800/80 text-[11px]">
        <div className="flex items-center gap-1.5 text-slate-300 truncate" title={performance.activeProvider}>
          <Cpu className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
          <span className="truncate font-mono">
            {performance.activeProvider.replace("ExecutionProvider", "")}
          </span>
        </div>

        <div className="flex items-center justify-end gap-1.5 text-emerald-400 font-medium">
          <ShieldCheck className="w-3.5 h-3.5 shrink-0" />
          <span>Cloud Req: 0 (100% Local)</span>
        </div>
      </div>
    </div>
  );
};
