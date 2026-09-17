/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        hp: {
          slate: {
            50: "#f8fafc",
            100: "#f1f5f9",
            200: "#e2e8f0",
            300: "#cbd5e1",
            400: "#94a3b8",
            500: "#64748b",
            600: "#475569",
            700: "#334155",
            800: "#1e293b",
            900: "#0f172a",
            950: "#020617",
          },
          accent: {
            blue: "#0096d6", // Accessible HP Blue
            cyan: "#06b6d4",
            emerald: "#10b981",
            amber: "#f59e0b",
            rose: "#f43f5e",
          },
          border: {
            DEFAULT: "#334155",
            high: "#64748b",
            accent: "#38bdf8",
          }
        },
      },
      fontFamily: {
        sans: [
          "Inter",
          "-apple-system",
          "BlinkMacSystemFont",
          "'Segoe UI'",
          "Roboto",
          "sans-serif",
        ],
        mono: [
          "'JetBrains Mono'",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "monospace",
        ],
      },
      boxShadow: {
        'high-contrast': '0 0 0 2px #38bdf8',
        'subtle-card': '0 4px 12px rgba(0, 0, 0, 0.25)',
      },
    },
  },
  plugins: [],
}
