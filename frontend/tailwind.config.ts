import type { Config } from "tailwindcss";
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    container: { center: true, padding: "1.25rem" },
    extend: {
      fontFamily: { sans: ["Inter", "system-ui", "-apple-system", "Segoe UI", "sans-serif"] },
      colors: {
        surface: {
          DEFAULT: "#f8fafc",
          muted: "#f1f5f9",
          line: "#e6e8ec",
          glass: "#ffffffb3",
        },
        accent: {
          start: "#6366f1",
          end: "#06b6d4",
          500: "#6366f1",
          600: "#4f46e5",
          700: "#4338ca",
        },
      },
      boxShadow: {
        glass: "0 1px 0 0 rgba(0,0,0,0.04), 0 8px 20px -12px rgba(0,0,0,0.25)",
      },
      keyframes: {
        "grid-move": { "0%": { transform: "translateY(0)" }, "100%": { transform: "translateY(-8px)" } },
        fade: { "0%": { opacity: "0" }, "100%": { opacity: "1" } },
        slide: {
          "0%": { opacity: "0", transform: "translateY(6px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "grid-move": "grid-move 14s ease-in-out infinite alternate",
        fade: "fade 200ms ease-out",
        slide: "slide 240ms ease-out",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};
export default config;
