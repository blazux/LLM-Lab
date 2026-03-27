/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        accent: '#22c55e',
        'accent-dim': '#16a34a',
        'accent-glow': 'rgba(34,197,94,0.15)',
      }
    },
  },
  plugins: [],
}
