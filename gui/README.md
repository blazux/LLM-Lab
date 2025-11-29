# LLM Lab GUI

A visual, drag-and-drop interface for building and configuring LLM models. Build your model like LEGO blocks!

## Features

- **Visual Model Builder**: Drag and drop components to design your LLM architecture
- **Interactive Canvas**: Connect blocks with visual flow connections
- **Real-time Validation**: Instant feedback on model configuration
- **Beautiful UI**: Modern, animated interface with smooth interactions
- **Component Library**:
  - 🔤 Tokenizer blocks
  - 📍 Positional encoding (RoPE, ALiBi, YARN)
  - 👁️ Attention mechanisms (MHA, GQA, MQA, MLA)
  - ⚡ Feed-forward networks with various activations
  - 🔷 Complete transformer layers

## Architecture

```
gui/
├── backend/          # FastAPI server
│   ├── main.py      # API server
│   └── api/         # API routes
└── frontend/         # React + TypeScript
    └── src/
        ├── components/    # UI components
        │   ├── Canvas.tsx         # React Flow canvas
        │   ├── Sidebar.tsx        # Component palette
        │   └── nodes/             # Custom node types
        └── styles/        # CSS styles
```

## Setup

### Backend Setup

1. Navigate to the backend directory:
```bash
cd gui/backend
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Start the FastAPI server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd gui/frontend
```

2. Install Node.js dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The GUI will be available at `http://localhost:5173`

## Usage

1. **Start both servers** (backend and frontend)
2. **Open your browser** to `http://localhost:5173`
3. **Drag blocks** from the left sidebar onto the canvas
4. **Connect blocks** by dragging from one node's handle to another
5. **Click any block** to configure its parameters (panel slides in from right)
6. **Create loops for repeated layers:**
   - Build ONE layer by connecting blocks from bottom-to-top (Norm → Attention → Norm → FFN)
   - Connect the top block back down to the bottom (creates a loop arrow!)
   - The loop arrow appears going around the RIGHT side
   - Click the glowing "↺ Repeat ×24" badge to configure layer count

**Note:** Models are built **bottom-to-top** like the original Transformer paper - tokenizer at bottom, output at top!
7. **Generate config** when ready (coming soon)

💡 **Pro Tip:** Instead of dragging 24 individual layers, draw a loop! Connect the end of your layer back to the beginning, and the system creates a beautiful loop arrow with a repetition counter.

## Component Blocks

### 🔤 Tokenizer Block (Purple)
- Starting point for any model
- Converts text to tokens
- Configurable tokenizer (HuggingFace)

### 📍 Positional Encoding (Blue)
- RoPE - Rotary Position Embeddings
- ALiBi - Attention with Linear Biases
- YARN - Yet Another RoPE Extension
- Sinusoidal - Original Transformer

### 🔄 Normalization (Yellow)
- RMSNorm (faster, modern)
- LayerNorm (traditional)
- Essential for training stability

### 👁️ Attention Block (Green)
- MHA - Multi-Head Attention
- GQA - Grouped-Query Attention
- MQA - Multi-Query Attention
- MLA - Multi-Head Latent Attention

### ⚡ Feed Forward (Orange)
- SwiGLU activation
- GELU, ReLU, SiLU
- Configurable hidden dimensions

### 🔁 Loop Arrows
- Connect backward to create loops
- Glowing violet arrows with badges
- Click badge to set repeat count (1-96)
- Curves around blocks for clarity

## Keyboard Shortcuts

- `Delete` / `Backspace` - Delete selected nodes
- `Ctrl/Cmd + C` - Copy selected nodes
- `Ctrl/Cmd + V` - Paste nodes
- `Ctrl/Cmd + Z` - Undo
- Mouse wheel - Zoom in/out
- Click + Drag - Pan canvas

## Coming Soon

- [ ] Config generation from visual layout
- [ ] Load existing configs into visual editor
- [ ] Training pipeline builder
- [ ] Real-time parameter count calculation
- [ ] Memory usage estimation
- [ ] Model presets (GPT-2, Llama-style, etc.)
- [ ] Export to JSON config
- [ ] Training job submission
- [ ] Real-time training logs
- [ ] Checkpoint browser

## Tech Stack

### Frontend
- **React** with TypeScript
- **React Flow** for node-based editor
- **Framer Motion** for animations
- **TailwindCSS** for styling
- **Vite** for fast development

### Backend
- **FastAPI** for REST API
- **Pydantic** for validation
- Integration with existing LLM-Lab configs

## Development

### Building for Production

Frontend:
```bash
cd gui/frontend
npm run build
```

Backend (with production server):
```bash
cd gui/backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Project Structure

```
LLM-Lab/
├── src/                    # Original CLI code
│   ├── config/            # ModelConfig, TrainingConfig
│   ├── model/             # Model implementations
│   └── training/          # Training loops
└── gui/                   # New GUI
    ├── backend/           # FastAPI server
    └── frontend/          # React app
```

## Contributing

The GUI is designed to integrate seamlessly with the existing LLM-Lab codebase. All model configurations are validated against the existing `ModelConfig` dataclass.

## License

Same as LLM-Lab main project
