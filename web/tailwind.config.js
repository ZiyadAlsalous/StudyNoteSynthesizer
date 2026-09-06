/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // One colour per source, used everywhere provenance is shown.
        slides: "#1f6feb",
        notes: "#1a7f37",
        textbook: "#d99b00",
      },
    },
  },
};
