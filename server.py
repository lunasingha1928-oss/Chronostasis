"""
server.py — Chronostasis OpenEnv Environment Server
====================================================
FastAPI server exposing the OpenEnv interface for the
Brahmaputra Valley Flood Intelligence environment.

Endpoints:
  POST /reset       → FloodObservation (initial state)
  POST /step        → StepResult (obs, reward, done, info)
  GET  /state       → FloodState (current full state)
  GET  /tasks       → List of available tasks
  GET  /health      → Health check

Run locally:
  uvicorn server:app --host 0.0.0.0 --port 7860

Environment variables:
  GEE_PROJECT               Your GEE project ID
  GEE_SERVICE_ACCOUNT_JSON  Service account JSON (for Docker/HF Space)
"""

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import ee
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from tasks import TASK_REGISTRY, BaseTask

# ──────────────────────────────────────────────
# GEE INIT
# ──────────────────────────────────────────────
GEE_PROJECT = os.getenv("GEE_PROJECT", "your-gee-project-id")

def init_gee():
    sa_json = os.getenv("GEE_SERVICE_ACCOUNT_JSON")
    try:
        if sa_json:
            credentials = ee.ServiceAccountCredentials(
                email=None, key_data=json.loads(sa_json)
            )
            ee.Initialize(credentials, project=GEE_PROJECT)
        else:
            ee.Initialize(project=GEE_PROJECT)
        return True
    except Exception as exc:
        print(f"[WARN] GEE init failed: {exc} — running in mock mode")
        return False

GEE_AVAILABLE = init_gee()


# ──────────────────────────────────────────────
# PYDANTIC MODELS
# ──────────────────────────────────────────────

class FloodObservation(BaseModel):
    """Observation returned by reset() and step()."""
    task_id: str                            = Field(..., description="Active task identifier")
    task_description: str                   = Field(..., description="Natural language task description")
    step: int                               = Field(..., description="Current step number (0 = just reset)")
    max_steps: int                          = Field(..., description="Maximum steps before episode ends")
    available_data: List[str]               = Field(..., description="Data sources the agent can query")
    last_action_result: Optional[str]       = Field(None, description="Result of the agent's last action, if any")
    last_action_error: Optional[str]        = Field(None, description="Error from last action, or null")
    context: Dict[str, Any]                 = Field(default_factory=dict, description="Task-specific context data")
    echoed_message: str                     = Field("", description="Echo of last agent message (OpenEnv compat)")


class FloodAction(BaseModel):
    """Action submitted by the agent."""
    message: str = Field(..., description="Agent's analysis, query, or answer string")


class FloodReward(BaseModel):
    """Reward signal for the completed step."""
    value: float    = Field(..., ge=0.0, le=1.0, description="Reward this step [0.0, 1.0]")
    rationale: str  = Field(..., description="Human-readable explanation of the reward")
    partial: bool   = Field(False, description="True if this is partial progress reward")


class StepResult(BaseModel):
    """Full result returned by step()."""
    observation:    FloodObservation
    reward:         float   = Field(..., ge=0.0, le=1.0)
    done:           bool
    info:           Dict[str, Any] = Field(default_factory=dict)
    last_action_error: Optional[str] = None


class FloodState(BaseModel):
    """Full episode state (returned by state())."""
    episode_id:     str
    task_id:        str
    step:           int
    max_steps:      int
    total_reward:   float
    done:           bool
    history:        List[Dict[str, Any]]
    gee_available:  bool
    started_at:     float


class TaskInfo(BaseModel):
    id:             str
    name:           str
    description:    str
    difficulty:     str   # easy | medium | hard
    max_steps:      int
    reward_range:   List[float]


class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(None, description="Task to initialise. Defaults to 'flood_year_comparison'.")


# ──────────────────────────────────────────────
# EPISODE STATE
# ──────────────────────────────────────────────

class EpisodeState:
    def __init__(self, task: BaseTask):
        self.episode_id  = str(uuid.uuid4())
        self.task        = task
        self.step        = 0
        self.done        = False
        self.total_reward = 0.0
        self.history: List[Dict[str, Any]] = []
        self.started_at  = time.time()

_current_episode: Optional[EpisodeState] = None


# ──────────────────────────────────────────────
# APP
# ──────────────────────────────────────────────

app = FastAPI(
    title="Chronostasis OpenEnv",
    description="Brahmaputra Valley Flood Intelligence — OpenEnv-compatible environment",
    version="1.0.0",
    tags_metadata=[
        {"name": "openenv", "description": "Core OpenEnv interface endpoints"},
        {"name": "meta",    "description": "Metadata and health endpoints"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────

@app.post("/reset", response_model=FloodObservation, tags=["openenv"])
async def reset(request: ResetRequest = ResetRequest()):
    """Reset the environment. Returns initial observation."""
    global _current_episode

    task_id = request.task_id or "flood_year_comparison"
    if task_id not in TASK_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{task_id}'. Available: {list(TASK_REGISTRY.keys())}"
        )

    task = TASK_REGISTRY[task_id](gee_available=GEE_AVAILABLE)
    _current_episode = EpisodeState(task)

    obs = FloodObservation(
        task_id=task_id,
        task_description=task.description,
        step=0,
        max_steps=task.max_steps,
        available_data=task.available_data,
        last_action_result=None,
        last_action_error=None,
        context=task.get_context(),
        echoed_message=f"Task started: {task.description}",
    )
    return obs


@app.post("/step", response_model=StepResult, tags=["openenv"])
async def step(action: FloodAction):
    """Advance the environment one step with the agent's action."""
    global _current_episode

    if _current_episode is None:
        raise HTTPException(status_code=400, detail="Episode not started. Call /reset first.")
    if _current_episode.done:
        raise HTTPException(status_code=400, detail="Episode is finished. Call /reset to start a new one.")

    ep = _current_episode
    ep.step += 1

    # Grade the action
    result = ep.task.step(action.message, ep.step)
    reward      = float(result["reward"])
    done        = bool(result["done"]) or ep.step >= ep.task.max_steps
    error       = result.get("error")
    action_result = result.get("result", "")

    ep.total_reward += reward
    ep.done = done
    ep.history.append({
        "step":   ep.step,
        "action": action.message[:200],
        "reward": reward,
        "done":   done,
        "error":  error,
    })

    obs = FloodObservation(
        task_id=ep.task.task_id,
        task_description=ep.task.description,
        step=ep.step,
        max_steps=ep.task.max_steps,
        available_data=ep.task.available_data,
        last_action_result=action_result,
        last_action_error=error,
        context=ep.task.get_context(),
        echoed_message=action.message,
    )

    return StepResult(
        observation=obs,
        reward=reward,
        done=done,
        info={"total_reward": ep.total_reward, "episode_id": ep.episode_id},
        last_action_error=error,
    )


@app.get("/state", response_model=FloodState, tags=["openenv"])
async def state():
    """Return the current full episode state."""
    if _current_episode is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    ep = _current_episode
    return FloodState(
        episode_id=ep.episode_id,
        task_id=ep.task.task_id,
        step=ep.step,
        max_steps=ep.task.max_steps,
        total_reward=ep.total_reward,
        done=ep.done,
        history=ep.history,
        gee_available=GEE_AVAILABLE,
        started_at=ep.started_at,
    )


@app.get("/tasks", response_model=List[TaskInfo], tags=["meta"])
async def list_tasks():
    """List all available tasks with metadata."""
    tasks = []
    for tid, tcls in TASK_REGISTRY.items():
        t = tcls(gee_available=GEE_AVAILABLE)
        tasks.append(TaskInfo(
            id=tid,
            name=t.name,
            description=t.description,
            difficulty=t.difficulty,
            max_steps=t.max_steps,
            reward_range=[0.0, 1.0],
        ))
    return tasks


@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok", "gee_available": GEE_AVAILABLE, "tasks": list(TASK_REGISTRY.keys())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
