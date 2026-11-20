import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from api.routes import router
from api.training import router as training_router
from api.inference import router as inference_router
from api.merge import router as merge_router
from api.export import router as export_router
from api.assembly import router as assembly_router
from api.learning import router as learning_router

@asynccontextmanager
async def lifespan(app):
    from api.training import restore_state
    restore_state()
    yield
    from api.learning import shutdown
    shutdown()


VERSION = (Path(__file__).resolve().parents[2] / 'VERSION').read_text().strip()

app = FastAPI(
    lifespan=lifespan,
    title="LLM Lab GUI API",
    description="API for visual LLM model building and training",
    version=VERSION
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
app.include_router(assembly_router, prefix="/api/assembly", tags=["assembly"])
app.include_router(learning_router, prefix="/api/learning", tags=["learning"])

# Built frontend path
frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"

if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    # Don't intercept API routes or docs
    if full_path in ("api", "docs") or full_path.startswith("api/") or full_path.startswith("docs/"):
        return JSONResponse(status_code=404, content={"detail": "Not found"})

    # Legacy /v2 URLs now live at the root
    if full_path == "v2" or full_path.startswith("v2/"):
        full_path = full_path[3:].lstrip("/")

    if frontend_dist.exists():
        file_path = (frontend_dist / full_path).resolve()
        if not file_path.is_relative_to(frontend_dist.resolve()):
            return JSONResponse(status_code=404, content={'detail': 'Not found'})
        if full_path and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(frontend_dist / "index.html")

    return {
        "message": "LLM Lab GUI API",
        "docs": "/docs",
        "version": VERSION,
        "note": "Frontend not built. Run from Docker or build frontend separately."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
