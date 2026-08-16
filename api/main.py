"""
FastAPI application entry point.

Usage (development):
    uvicorn api.main:app --reload

Then open http://127.0.0.1:8000/docs — FastAPI auto-generates an
interactive API explorer from the type hints and Pydantic schemas below,
which is exactly how we'll verify every endpoint works BEFORE any React
code touches it.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import matches, performance, betting


app = FastAPI(
    title="PitchSense API",
    description="Football match predictions and betting analysis",
    version="0.1.0",
)

# CORS: the Next.js dashboard (Phase 5's frontend) runs on a different
# port (typically 3000) than this API (8000) during development — browsers
# block cross-origin requests by default unless the server explicitly
# allows it. Wide open for local dev; this would need tightening to a
# specific origin before ever deploying this publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(matches.router, prefix="/matches", tags=["matches"])
app.include_router(performance.router, prefix="/model", tags=["model"])
app.include_router(betting.router, prefix="/betting", tags=["betting"])


@app.get("/health")
def health_check():
    """Simple liveness check — useful for confirming the server is up
    before debugging anything more complex."""
    return {"status": "ok"}