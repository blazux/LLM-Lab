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
from api.export import router as export_router

app = FastAPI(
    title="LLM Lab GUI API",
    description="API for visual LLM model building and training",
    version="1.0.0"
)

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (safe for local development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(router, prefix="/api")
app.include_router(training_router, prefix="/api/training", tags=["training"])
app.include_router(inference_router, prefix="/api/inference", tags=["inference"])
app.include_router(merge_router, prefix="/api/merge", tags=["merge"])
app.include_router(export_router, prefix="/api/export", tags=["export"])

# Built frontend paths
frontend_v1_dist = Path(__file__).parent.parent / "frontend" / "dist"
frontend_v2_dist = Path(__file__).parent.parent.parent / "gui-v2" / "frontend" / "dist"

# Mount v1 static assets
if frontend_v1_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_v1_dist / "assets")), name="assets")

# Mount v2 static assets (built with --base=/v2/, so assets live at /v2/assets/...)
if frontend_v2_dist.exists():
    app.mount("/v2/assets", StaticFiles(directory=str(frontend_v2_dist / "assets")), name="assets-v2")

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    # Don't intercept API routes or docs
    if full_path.startswith("api") or full_path.startswith("docs"):
        return {"error": "Not found"}

    # Serve v2 SPA for /v2/* routes
    if full_path == "v2" or full_path.startswith("v2/"):
        if frontend_v2_dist.exists():
            # Strip leading "v2/" to resolve actual file (e.g. v2/favicon.ico)
            relative = full_path[3:].lstrip("/") if full_path.startswith("v2/") else ""
            if relative:
                candidate = frontend_v2_dist / relative
                if candidate.is_file():
                    return FileResponse(candidate)
            return FileResponse(frontend_v2_dist / "index.html")

    # Serve v1 SPA for all other routes
    if frontend_v1_dist.exists():
        file_path = frontend_v1_dist / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(frontend_v1_dist / "index.html")

    return {
        "message": "LLM Lab GUI API",
        "docs": "/docs",
        "version": "1.0.0",
        "note": "Frontend not built. Run from Docker or build frontend separately."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
