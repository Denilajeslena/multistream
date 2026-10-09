import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from server.backend.config import settings
from server.backend.db.database import init_db
from server.backend.routers.health import router as health_router
from server.backend.routers.cameras import router as cameras_router
from server.backend.routers.query import router as query_router
from server.backend.routers.evidence import router as evidence_router
from server.backend.routers.simulation import router as simulation_router
from server.backend.routers.uploads import router as uploads_router
from server.backend.routers.analysis import router as analysis_router
from server.backend.ingestion.realtime_analysis import realtime_analysis_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure directories and database initialized
    settings.ensure_directories()
    init_db()
    yield
    realtime_analysis_service.shutdown()

app = FastAPI(
    title="Central CCTV Intelligence Server",
    description="Central AI server for multi-camera video intelligence, grounded search, and evidence retrieval",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for LAN/Wi-Fi and Web UI clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static media directories
app.mount("/data/frames", StaticFiles(directory=str(settings.FRAMES_DIR)), name="frames")
app.mount("/data/evidence", StaticFiles(directory=str(settings.EVIDENCE_DIR)), name="evidence")

# Include routers
app.include_router(health_router)
app.include_router(cameras_router)
app.include_router(query_router)
app.include_router(evidence_router)
app.include_router(simulation_router)
app.include_router(uploads_router)
app.include_router(analysis_router)

# Mount frontend if built
frontend_dist = settings.BASE_DIR / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(
        "server.backend.main:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=False
    )
