/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Professional muted palette
        primary: '#3b82f6',    // Blue-500 - main accent
        secondary: '#64748b',  // Slate-500 - secondary
        // Category accent colors (muted)
        'cat-core': '#6366f1',      // Indigo - core components
        'cat-position': '#0ea5e9',  // Sky - positional encoding
        'cat-attention': '#14b8a6', // Teal - attention
        'cat-norm': '#f59e0b',      // Amber - normalization
        'cat-ffn': '#f97316',       // Orange - feed forward
        'cat-ssm': '#06b6d4',       // Cyan - state space
        'cat-data': '#3b82f6',      // Blue - datasets
        'cat-optim': '#ef4444',     // Red - optimizers
        'cat-sched': '#8b5cf6',     // Violet - schedulers
        'cat-config': '#22c55e',    // Green - configuration
        'cat-algo': '#ec4899',      // Pink - algorithms
      }
    },
  },
  plugins: [],
}
