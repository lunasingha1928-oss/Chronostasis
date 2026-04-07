"""
server/app.py — Chronostasis OpenEnv Server Entry Point
========================================================
Standalone FastAPI app for openenv validate multi-mode deployment.
"""

import json
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

# Add parent dir to path so we can import tasks
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field

from tasks import TASK_REGISTRY, REGIONS, DEFAULT_REGION, BaseTask

# ── Config ──
GEE_PROJECT  = os.getenv("GEE_PROJECT", "your-gee-project-id")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
AGENT_MODEL  = "llama-3.3-70b-versatile"

# ── GEE (optional) ──
GEE_AVAILABLE = False
try:
    import ee
    sa_json = os.getenv("GEE_SERVICE_ACCOUNT_JSON")
    if sa_json:
        key_data = sa_json if isinstance(sa_json, dict) else json.loads(sa_json)
        credentials = ee.ServiceAccountCredentials(
            email=key_data.get("client_email"), key_data=key_data)
        ee.Initialize(credentials, project=GEE_PROJECT)
    else:
        ee.Initialize(project=GEE_PROJECT)
    GEE_AVAILABLE = True
except Exception as exc:
    print(f"[WARN] GEE mock mode: {exc}")


# ── Pydantic models ──
class FloodObservation(BaseModel):
    task_id:            str
    task_description:   str
    step:               int
    max_steps:          int
    available_data:     List[str]
    last_action_result: Optional[str] = None
    last_action_error:  Optional[str] = None
    context:            Dict[str, Any] = Field(default_factory=dict)
    echoed_message:     str = ""
    region_id:          str = DEFAULT_REGION
    region_name:        str = "Brahmaputra Valley"

class FloodAction(BaseModel):
    message: str

class StepResult(BaseModel):
    observation:       FloodObservation
    reward:            float
    done:              bool
    info:              Dict[str, Any] = Field(default_factory=dict)
    last_action_error: Optional[str] = None

class FloodState(BaseModel):
    episode_id:    str
    task_id:       str
    region_id:     str
    step:          int
    max_steps:     int
    total_reward:  float
    done:          bool
    history:       List[Dict[str, Any]]
    gee_available: bool
    started_at:    float

class ResetRequest(BaseModel):
    task_id:   Optional[str] = None
    region_id: Optional[str] = None

class RegionInfo(BaseModel):
    id:           str
    name:         str
    state:        str
    river:        str
    peak_year:    int
    accuracy_pct: float
    flood_areas:  Dict[str, float]

class TaskInfo(BaseModel):
    id:          str
    name:        str
    description: str
    difficulty:  str
    max_steps:   int
    region_id:   str


# ── Episode state ──
class EpisodeState:
    def __init__(self, task: BaseTask, region_id: str):
        self.episode_id   = str(uuid.uuid4())
        self.task         = task
        self.region_id    = region_id
        self.step         = 0
        self.done         = False
        self.total_reward = 0.0
        self.history: List[Dict[str, Any]] = []
        self.started_at   = time.time()

_episode: Optional[EpisodeState] = None


# ── App ──
app = FastAPI(title="Chronostasis OpenEnv", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_static = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "static")
if os.path.isdir(_static):
    app.mount("/static", StaticFiles(directory=_static), name="static")

@app.get("/", include_in_schema=False)
async def root():
    idx = os.path.join(_static, "index.html")
    if os.path.isfile(idx):
        return FileResponse(idx)
    return {"name": "Chronostasis", "status": "running", "docs": "/docs"}


# ── OpenEnv endpoints ──
@app.post("/reset", response_model=FloodObservation)
async def reset(req: ResetRequest = ResetRequest()):
    global _episode
    task_id   = req.task_id   or "flood_year_comparison"
    region_id = req.region_id or DEFAULT_REGION
    if task_id not in TASK_REGISTRY:
        raise HTTPException(400, f"Unknown task: {task_id}")
    if region_id not in REGIONS:
        raise HTTPException(400, f"Unknown region: {region_id}")
    task = TASK_REGISTRY[task_id](gee_available=GEE_AVAILABLE, region=region_id)
    _episode = EpisodeState(task, region_id)
    region = REGIONS[region_id]
    return FloodObservation(
        task_id=task_id, task_description=task.description,
        step=0, max_steps=task.max_steps, available_data=task.available_data,
        context=task.get_context(), echoed_message=f"Task: {task.description}",
        region_id=region_id, region_name=region["name"])


@app.post("/step", response_model=StepResult)
async def step(action: FloodAction):
    global _episode
    if _episode is None:
        raise HTTPException(400, "No active episode. Call /reset first.")
    if _episode.done:
        raise HTTPException(400, "Episode done. Call /reset.")
    ep = _episode
    ep.step += 1
    result = ep.task.step(action.message, ep.step)
    reward = float(result.get("reward", 0) or 0)
    done   = bool(result.get("done", False)) or ep.step >= ep.task.max_steps
    ep.total_reward = round(ep.total_reward + reward, 4)
    ep.done = done
    ep.history.append({"step": ep.step, "action": action.message[:200],
                       "reward": reward, "done": done})
    region = REGIONS[ep.region_id]
    obs = FloodObservation(
        task_id=ep.task.task_id, task_description=ep.task.description,
        step=ep.step, max_steps=ep.task.max_steps,
        available_data=ep.task.available_data,
        last_action_result=result.get("result", ""),
        last_action_error=result.get("error"),
        context=ep.task.get_context(), echoed_message=action.message,
        region_id=ep.region_id, region_name=region["name"])
    return StepResult(observation=obs, reward=reward, done=done,
                      info={"total_reward": ep.total_reward,
                            "episode_id": ep.episode_id},
                      last_action_error=result.get("error"))


@app.get("/state", response_model=FloodState)
async def state():
    if _episode is None:
        raise HTTPException(400, "No active episode.")
    ep = _episode
    return FloodState(episode_id=ep.episode_id, task_id=ep.task.task_id,
                      region_id=ep.region_id, step=ep.step,
                      max_steps=ep.task.max_steps, total_reward=ep.total_reward,
                      done=ep.done, history=ep.history,
                      gee_available=GEE_AVAILABLE, started_at=ep.started_at)


@app.get("/regions", response_model=List[RegionInfo])
async def list_regions():
    return [RegionInfo(id=rid, name=r["name"], state=r["state"],
                       river=r["river"], peak_year=r["peak_year"],
                       accuracy_pct=r["accuracy_pct"],
                       flood_areas={str(k): v for k, v in r["flood_areas"].items()})
            for rid, r in REGIONS.items()]


@app.get("/tasks", response_model=List[TaskInfo])
async def list_tasks():
    return [TaskInfo(id=tid, name=tcls(gee_available=False).name,
                     description=tcls(gee_available=False).description,
                     difficulty=tcls(gee_available=False).difficulty,
                     max_steps=tcls(gee_available=False).max_steps,
                     region_id=DEFAULT_REGION)
            for tid, tcls in TASK_REGISTRY.items()]


@app.get("/health")
async def health():
    return {"status": "ok", "gee_available": GEE_AVAILABLE,
            "llm_configured": bool(GROQ_API_KEY),
            "agent_model": AGENT_MODEL,
            "regions": list(REGIONS.keys()),
            "tasks": list(TASK_REGISTRY.keys())}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()