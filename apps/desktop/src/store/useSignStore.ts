import { create } from "zustand";

export interface PerformanceMetrics {
  fps: number;
  visionLatencyMs: number;
  signLatencyMs: number;
  languageLatencyMs: number;
  activeProvider: string;
}

export interface ConversationEntry {
  id: string;
  timestamp: string;
  glosses: string[];
  sentence: string;
  confidence: number;
}

interface SignState {
  // Connection & Backend
  backendUrl: string;
  wsUrl: string;
  connectionStatus: "connecting" | "connected" | "disconnected";
  
  // Real-time Recognition State
  currentGloss: string;
  confidence: number;
  sentenceAccumulator: string[];
  reconstructedSentence: string;
  landmarksDetected: boolean;

  // Speech
  isSpeaking: boolean;

  // History Log
  history: ConversationEntry[];

  // Hardware Telemetry
  performance: PerformanceMetrics;

  // Actions
  setBackendUrl: (url: string) => void;
  connectWebSocket: (customUrl?: string) => void;
  disconnectWebSocket: () => void;
  sendLandmarkFrame: (landmarks: number[]) => void;
  speakCurrentSentence: () => Promise<void>;
  speakText: (text: string) => Promise<void>;
  clearAccumulator: () => void;
  updatePerformance: (partial: Partial<PerformanceMetrics>) => void;
  checkHealth: () => Promise<void>;
}

let socket: WebSocket | null = null;
let reconnectTimer: any = null;
let audioContext: AudioContext | null = null;

export const useSignStore = create<SignState>((set, get) => ({
  backendUrl: "http://localhost:8000",
  wsUrl: "ws://localhost:8000/translate/live",
  connectionStatus: "disconnected",

  currentGloss: "",
  confidence: 0,
  sentenceAccumulator: [],
  reconstructedSentence: "",
  landmarksDetected: false,

  isSpeaking: false,
  history: [],

  performance: {
    fps: 30,
    visionLatencyMs: 3.8,
    signLatencyMs: 11.4,
    languageLatencyMs: 2.1,
    activeProvider: "CoreMLExecutionProvider",
  },

  setBackendUrl: (url: string) => {
    const cleanUrl = url.replace(/\/$/, "");
    const wsUrl = cleanUrl.replace(/^http/, "ws") + "/translate/live";
    set({ backendUrl: cleanUrl, wsUrl });
    get().disconnectWebSocket();
    get().connectWebSocket(wsUrl);
  },

  checkHealth: async () => {
    try {
      const res = await fetch(`${get().backendUrl}/health`);
      if (res.ok) {
        const data = await res.json();
        set((state) => ({
          performance: {
            ...state.performance,
            activeProvider: data.execution_provider || state.performance.activeProvider,
          },
        }));
      }
    } catch {
      // Backend may be starting up
    }
  },

  connectWebSocket: (customUrl?: string) => {
    const targetUrl = customUrl || get().wsUrl;

    if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
      return;
    }

    set({ connectionStatus: "connecting" });
    get().checkHealth();

    try {
      socket = new WebSocket(targetUrl);

      socket.onopen = () => {
        set({ connectionStatus: "connected" });
        if (reconnectTimer) {
          clearTimeout(reconnectTimer);
          reconnectTimer = null;
        }
      };

      socket.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);

          const newSentence = payload.reconstructed_sentence || get().reconstructedSentence;
          const prevSentence = get().reconstructedSentence;

          // If a new complete sentence was finalized, save it to history
          if (
            newSentence &&
            newSentence !== prevSentence &&
            payload.sentence_accumulator?.length > 0 &&
            payload.sentence_accumulator.length !== get().sentenceAccumulator.length
          ) {
            const entry: ConversationEntry = {
              id: Date.now().toString(),
              timestamp: new Date().toLocaleTimeString(),
              glosses: [...payload.sentence_accumulator],
              sentence: newSentence,
              confidence: payload.confidence || 0.85,
            };
            set((s) => ({ history: [entry, ...s.history.slice(0, 49)] }));
          }

          set({
            landmarksDetected: !!payload.current_frame_landmarks_detected,
            currentGloss: payload.latest_gloss || get().currentGloss,
            confidence: typeof payload.confidence === "number" ? payload.confidence : get().confidence,
            sentenceAccumulator: payload.sentence_accumulator || get().sentenceAccumulator,
            reconstructedSentence: newSentence,
          });
        } catch (e) {
          console.error("[useSignStore] Error parsing WebSocket message:", e);
        }
      };

      socket.onerror = () => {
        set({ connectionStatus: "disconnected" });
      };

      socket.onclose = () => {
        set({ connectionStatus: "disconnected" });
        // Auto-reconnect after 3 seconds
        if (!reconnectTimer) {
          reconnectTimer = setTimeout(() => {
            reconnectTimer = null;
            get().connectWebSocket(targetUrl);
          }, 3000);
        }
      };
    } catch (err) {
      set({ connectionStatus: "disconnected" });
      if (!reconnectTimer) {
        reconnectTimer = setTimeout(() => {
          reconnectTimer = null;
          get().connectWebSocket(targetUrl);
        }, 3000);
      }
    }
  },

  disconnectWebSocket: () => {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    if (socket) {
      socket.close();
      socket = null;
    }
    set({ connectionStatus: "disconnected" });
  },

  sendLandmarkFrame: (landmarks: number[]) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ landmarks }));
    }
  },

  speakCurrentSentence: async () => {
    const sentence = get().reconstructedSentence || get().currentGloss;
    if (!sentence) return;
    await get().speakText(sentence);
  },

  speakText: async (text: string) => {
    if (!text || get().isSpeaking) return;

    set({ isSpeaking: true });

    try {
      const res = await fetch(`${get().backendUrl}/conversation/speak`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      if (!res.ok) throw new Error(`TTS HTTP error: ${res.status}`);

      const data = await res.json();
      if (data.audio_base64) {
        const audioBytes = Uint8Array.from(atob(data.audio_base64), (c) => c.charCodeAt(0));
        
        if (!audioContext) {
          audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        }
        if (audioContext.state === "suspended") {
          await audioContext.resume();
        }

        const buffer = await audioContext.decodeAudioData(audioBytes.buffer);
        const source = audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(audioContext.destination);

        source.onended = () => {
          set({ isSpeaking: false });
        };

        source.start(0);
      } else {
        set({ isSpeaking: false });
      }
    } catch (err) {
      console.error("[useSignStore] Speak error:", err);
      // Fallback to browser Web Speech API
      if ("speechSynthesis" in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.onend = () => set({ isSpeaking: false });
        utterance.onerror = () => set({ isSpeaking: false });
        window.speechSynthesis.speak(utterance);
      } else {
        set({ isSpeaking: false });
      }
    }
  },

  clearAccumulator: () => {
    set({
      currentGloss: "",
      confidence: 0,
      sentenceAccumulator: [],
      reconstructedSentence: "",
    });
  },

  updatePerformance: (partial: Partial<PerformanceMetrics>) => {
    set((state) => ({
      performance: { ...state.performance, ...partial },
    }));
  },
}));
