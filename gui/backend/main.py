import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api.routes import router
from api.training import router as training_router
from api.inference import router as inference_router
from api.merge import router as merge_router

app = FastAPI(
    title="LLM Lab GUI API",
    description="API for visual LLM model building and training",
    version="1.0.0"
)

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8000"],  # Vite dev + Docker
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(router, prefix="/api")
app.include_router(training_router, prefix="/api/training", tags=["training"])
app.include_router(inference_router, prefix="/api/inference", tags=["inference"])
app.include_router(merge_router, prefix="/api/merge", tags=["merge"])

# Check if built frontend exists (for Docker deployment)
frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    # Mount static files
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

    # Serve index.html for all non-API routes (SPA routing)
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # Don't intercept API routes or docs
        if full_path.startswith("api") or full_path.startswith("docs"):
            return {"error": "Not found"}

        # Serve static files if they exist
        file_path = frontend_dist / full_path
        if file_path.is_file():
            return FileResponse(file_path)

        # Otherwise serve index.html (SPA)
        return FileResponse(frontend_dist / "index.html")
else:
    # Development mode - API only
    @app.get("/")
    async def root():
        return {
            "message": "LLM Lab GUI API",
            "docs": "/docs",
            "version": "1.0.0",
            "note": "Frontend not built. Run from Docker or build frontend separately."
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
