import type { Config } from "tailwindcss";
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    container: { center: true, padding: "1.5rem" },
    extend: {
      fontFamily: { sans: ["var(--font-jakarta)", "Inter", "system-ui", "sans-serif"] },
      colors: {
        night: {
          900: "#0D1117",
          800: "#161B22",
          700: "#1E293B",
          glass: "rgba(255,255,255,0.08)",
          border: "rgba(255,255,255,0.125)",
        },
        accent: {
          100: "#C8E7FF",
          300: "#7DD3FC",
          500: "#38BDF8",
          700: "#0EA5E9",
        },
        gold: {
          400: "#FACC15",
          500: "#EAB308",
        },
      },
      boxShadow: {
        glass: "0 8px 32px rgba(0,0,0,0.35)",
        glow: "0 0 25px rgba(56,189,248,0.25)",
      },
      backgroundImage: {
        "prime-grid": "radial-gradient(circle at center, rgba(255,255,255,0.08) 0, transparent 60%)",
      },
      keyframes: {
        "grid-move": { "0%": { transform: "translateY(0)" }, "100%": { transform: "translateY(-8px)" } },
        hue: { "0%": { filter: "hue-rotate(0deg)" }, "100%": { filter: "hue-rotate(360deg)" } },
      },
      animation: {
        "grid-move": "grid-move 22s ease-in-out infinite alternate",
        hue: "hue 18s linear infinite",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};
export default config;
