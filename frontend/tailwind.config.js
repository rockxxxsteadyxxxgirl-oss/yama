/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ["'DM Sans'", "Inter", "system-ui", "sans-serif"],
        body: ["'Inter'", "system-ui", "sans-serif"],
      },
      colors: {
        midnight: "#0a1a2f",
        teal: "#3ac2c9",
        sky: "#7ed8ff",
      },
      boxShadow: {
        glow: "0 0 40px rgba(122, 227, 255, 0.35)",
      },
    },
  },
  plugins: [],
};
