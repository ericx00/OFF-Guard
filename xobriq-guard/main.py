import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import assess
from schema import Case, RiskReport
from sanctions_check import check_name

app = FastAPI(title="Off-Guard")

static_dir = Path(__file__).parent / "static"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.on_event("startup")
async def startup_event() -> None:
    if not os.getenv("FIREWORKS_API_KEY"):
        raise RuntimeError(
            "FIREWORKS_API_KEY is not set. Add it to your environment or .env file before starting the server."
        )


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.post("/screen", response_model=RiskReport)
async def screen_case(case: Case) -> RiskReport:
    try:
        return await assess(case)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class SanctionsRequest(BaseModel):
    name: str


@app.post("/sanctions-check")
async def sanctions_check(payload: SanctionsRequest) -> dict:
    """
    Real OFAC SDN sanctions-list screening on a submitted name.
    Uses live, public government data — not synthetic test data.
    """
    try:
        matches = check_name(payload.name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "name": payload.name,
        "flagged": len(matches) > 0,
        "matches": [m.as_dict() for m in matches],
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
