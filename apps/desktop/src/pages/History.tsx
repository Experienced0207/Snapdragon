import React from "react";
import { History as HistoryIcon, Volume2, Trash2, Clock, Sparkles } from "lucide-react";
import { useSignStore } from "../store/useSignStore";

export const History: React.FC = () => {
  const { history, speakText, isSpeaking } = useSignStore();

  return (
    <div className="h-full flex flex-col gap-6 max-w-5xl mx-auto">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-slate-100 tracking-tight flex items-center gap-2">
            <HistoryIcon className="w-5 h-5 text-cyan-400" />
            <span>Conversation History</span>
          </h2>
          <p className="text-sm text-slate-400">
            Log of previously recognized and reconstructed sign sequences in this session.
          </p>
        </div>
      </div>

      {/* History Feed */}
      <div className="flex-1 overflow-y-auto space-y-3 pr-2">
        {history.length > 0 ? (
          history.map((entry) => (
            <div
              key={entry.id}
              className="bg-slate-900 border border-slate-800 hover:border-slate-700 p-4 rounded-xl shadow-md transition-all flex items-center justify-between gap-4"
            >
              <div className="space-y-1.5 min-w-0">
                <div className="flex items-center gap-2 text-xs text-slate-400">
                  <Clock className="w-3.5 h-3.5 text-slate-500" />
                  <span className="font-mono">{entry.timestamp}</span>
                  <span className="text-slate-600">•</span>
                  <span className="text-emerald-400 font-medium">
                    {Math.round(entry.confidence * 100)}% Match
                  </span>
                </div>

                <p className="text-base font-bold text-white tracking-tight">
                  "{entry.sentence}"
                </p>

                <div className="flex flex-wrap gap-1.5 pt-0.5">
                  {entry.glosses.map((g, i) => (
                    <span
                      key={i}
                      className="px-2 py-0.5 rounded bg-slate-800 text-[11px] font-mono text-cyan-400 border border-slate-700/60 uppercase"
                    >
                      {g}
                    </span>
                  ))}
                </div>
              </div>

              <button
                onClick={() => speakText(entry.sentence)}
                disabled={isSpeaking}
                className="px-3.5 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-semibold flex items-center gap-2 transition-all shrink-0 active:scale-95"
                title="Re-speak this sentence"
              >
                <Volume2 className="w-4 h-4 text-cyan-400" />
                <span>Play</span>
              </button>
            </div>
          ))
        ) : (
          <div className="flex flex-col items-center justify-center py-20 text-center text-slate-500 bg-slate-900/40 border border-dashed border-slate-800 rounded-2xl">
            <Sparkles className="w-10 h-10 mb-3 text-slate-600" />
            <p className="text-base font-semibold text-slate-300">No conversation history yet</p>
            <p className="text-xs text-slate-500 mt-1 max-w-sm">
              Recognized ISL sentences will automatically be recorded here as you sign.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
