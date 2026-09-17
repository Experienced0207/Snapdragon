import React, { useEffect } from "react";
import { NavLink, Outlet } from "react-router-dom";
import {
  Video,
  History,
  Cpu,
  Settings,
  Radio,
  Volume2,
  RefreshCw,
  Shield,
} from "lucide-react";
import { useSignStore } from "../../store/useSignStore";
import { PerformanceWidget } from "../telemetry/PerformanceWidget";

export const Shell: React.FC = () => {
  const {
    connectionStatus,
    connectWebSocket,
    disconnectWebSocket,
    performance,
    isSpeaking,
    speakCurrentSentence,
    reconstructedSentence,
  } = useSignStore();

  useEffect(() => {
    connectWebSocket();
    return () => {
      disconnectWebSocket();
    };
  }, []);

  const navItems = [
    { to: "/", label: "Live Translation", icon: Video },
    { to: "/history", label: "Conversation Log", icon: History },
    { to: "/telemetry", label: "NPU / AI Engine", icon: Cpu },
    { to: "/settings", label: "Settings", icon: Settings },
  ];

  return (
    <div className="flex h-screen w-screen bg-slate-950 text-slate-100 overflow-hidden font-sans">
      {/* Accessible Fixed Sidebar */}
      <aside
        className="w-64 border-r border-slate-800/90 bg-slate-900/95 flex flex-col justify-between shrink-0 select-none"
        aria-label="Primary Navigation"
      >
        <div>
          {/* Brand Header */}
          <div className="p-5 border-b border-slate-800">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-cyan-500/20 border border-cyan-400 flex items-center justify-center text-cyan-400 font-black tracking-tighter">
                SB
              </div>
              <div>
                <h1 className="text-base font-black tracking-wider text-slate-100 uppercase">
                  SignBridge
                </h1>
                <p className="text-[10px] font-semibold text-cyan-400 tracking-wider uppercase">
                  ISL Accessibility AI
                </p>
              </div>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="p-3 space-y-1.5" aria-label="App routes">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  className={({ isActive }) =>
                    `flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                      isActive
                        ? "bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 font-semibold"
                        : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60 border border-transparent"
                    }`
                  }
                >
                  <Icon className="w-4 h-4 shrink-0" />
                  <span>{item.label}</span>
                </NavLink>
              );
            })}
          </nav>
        </div>

        {/* Telemetry Sidebar Card */}
        <div className="p-3 border-t border-slate-800/90 space-y-3">
          <PerformanceWidget />

          <div className="text-[11px] text-slate-500 flex items-center justify-between px-1">
            <span>SignBridge v1.0.0</span>
            <span className="flex items-center gap-1 text-slate-400">
              <Shield className="w-3 h-3 text-emerald-400" />
              Private & Local
            </span>
          </div>
        </div>
      </aside>

      {/* Main App Container */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Top Status Header */}
        <header
          className="h-16 border-b border-slate-800/90 bg-slate-900/60 backdrop-blur-md px-6 flex items-center justify-between shrink-0"
          role="banner"
        >
          <div className="flex items-center gap-3">
            <span className="text-xs uppercase font-bold tracking-widest text-slate-400">
              Active Environment
            </span>
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-800 border border-slate-700 text-xs font-mono text-slate-200">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              <span>
                {performance.activeProvider.includes("CoreML")
                  ? "CoreML / Apple Neural Engine"
                  : performance.activeProvider.includes("QNN")
                  ? "Qualcomm QNN / Snapdragon"
                  : "CPU Optimized Backend"}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* Quick Speak Trigger if sentence available */}
            {reconstructedSentence && (
              <button
                onClick={speakCurrentSentence}
                disabled={isSpeaking}
                className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-cyan-600 hover:bg-cyan-500 text-slate-950 font-semibold text-xs transition-all shadow-md active:scale-95 disabled:opacity-50"
                title="Speak current reconstructed sentence"
              >
                <Volume2 className={`w-3.5 h-3.5 ${isSpeaking ? "animate-bounce" : ""}`} />
                <span>{isSpeaking ? "Speaking..." : "Speak Transcript"}</span>
              </button>
            )}

            {/* Connection Status Badge */}
            <div
              className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium border ${
                connectionStatus === "connected"
                  ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-300"
                  : connectionStatus === "connecting"
                  ? "bg-amber-500/10 border-amber-500/30 text-amber-300"
                  : "bg-rose-500/10 border-rose-500/30 text-rose-300"
              }`}
              role="status"
              aria-live="polite"
            >
              <Radio
                className={`w-3.5 h-3.5 ${
                  connectionStatus === "connected"
                    ? "animate-pulse text-emerald-400"
                    : connectionStatus === "connecting"
                    ? "animate-spin text-amber-400"
                    : "text-rose-400"
                }`}
              />
              <span className="capitalize font-semibold">{connectionStatus}</span>

              {connectionStatus === "disconnected" && (
                <button
                  onClick={() => connectWebSocket()}
                  className="ml-1 text-slate-300 hover:text-white p-0.5"
                  title="Retry Connection"
                >
                  <RefreshCw className="w-3 h-3" />
                </button>
              )}
            </div>
          </div>
        </header>

        {/* Page Content Viewport */}
        <main className="flex-1 overflow-y-auto p-6 bg-slate-950/80">
          <Outlet />
        </main>
      </div>
    </div>
  );
};
