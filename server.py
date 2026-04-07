"""
server.py — Chronostasis OpenEnv Environment Server
====================================================
Multi-region flood intelligence environment for Indian river basins.
Real LLM agent via HuggingFace router + GEE satellite data.
"""

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import ee
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from openai import OpenAI

from tasks import TASK_REGISTRY, REGIONS, DEFAULT_REGION, BaseTask

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
GEE_PROJECT = os.getenv("GEE_PROJECT", "your-gee-project-id")
HF_TOKEN    = os.getenv("HF_TOKEN", "")
AGENT_MODEL = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
AGENT_BASE  = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")


def init_gee():
    sa_json = os.getenv("GEE_SERVICE_ACCOUNT_JSON")
    try:
        if sa_json:
            key_data = sa_json if isinstance(sa_json, dict) else json.loads(sa_json)
            credentials = ee.ServiceAccountCredentials(
                email=key_data.get("client_email"), key_data=key_data)
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
    observation:        FloodObservation
    reward:             float
    done:               bool
    info:               Dict[str, Any] = Field(default_factory=dict)
    last_action_error:  Optional[str] = None

class FloodState(BaseModel):
    episode_id:   str
    task_id:      str
    region_id:    str
    step:         int
    max_steps:    int
    total_reward: float
    done:         bool
    history:      List[Dict[str, Any]]
    gee_available: bool
    started_at:   float

class ResetRequest(BaseModel):
    task_id:   Optional[str] = None
    region_id: Optional[str] = None

class AgentRunRequest(BaseModel):
    max_steps: Optional[int] = None

class TaskInfo(BaseModel):
    id:         str
    name:       str
    description: str
    difficulty:  str
    max_steps:   int
    region_id:   str

class RegionInfo(BaseModel):
    id:         str
    name:       str
    state:      str
    river:      str
    peak_year:  int
    accuracy_pct: float
    flood_areas: Dict[str, float]


# ──────────────────────────────────────────────
# EPISODE STATE
# ──────────────────────────────────────────────

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

_current_episode: Optional[EpisodeState] = None


# ──────────────────────────────────────────────
# AGENT LLM CLIENT
# ──────────────────────────────────────────────

def get_llm_client() -> Optional[OpenAI]:
    api_key = HF_TOKEN or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None
    return OpenAI(base_url=AGENT_BASE, api_key=api_key)


def build_agent_prompt(episode: EpisodeState) -> str:
    task = episode.task
    region = REGIONS[episode.region_id]
    ctx = task.get_context()
    history_txt = ""
    if episode.history:
        lines = [f"Step {h['step']}: reward={h['reward']:.2f} | {h['action'][:100]}" 
                 for h in episode.history[-4:]]
        history_txt = "\n".join(lines)

    return f"""You are a GIS flood analysis agent for the {region['name']} ({region['state']}).

TASK: {task.description}

AVAILABLE DATA: {', '.join(task.available_data)}

CONTEXT:
- Region: {region['name']}, {region['state']}
- River: {region['river']}
- SAR threshold: {region['sar_threshold_db']} dB
- Model accuracy: {region['accuracy_pct']}%
- Flood areas km2: 2022={region['flood_areas'][2022]}, 2023={region['flood_areas'][2023]}, 2024={region['flood_areas'][2024]}
- Peak year: {region['peak_year']}
- Chronic area: {region['chronic_km2']} km2
- Population at risk: {region['chronic_pop']:,}
- High-risk zones: {', '.join(region['high_risk_zones'])}
- Risk zones km2: high={region['risk_zones_km2']['high']}, moderate={region['risk_zones_km2']['moderate']}, low={region['risk_zones_km2']['low']}

STEP {episode.step + 1} of {task.max_steps}
PREVIOUS STEPS:
{history_txt if history_txt else 'None yet'}

Provide a specific, data-backed analysis. Include exact numbers. Be concise but precise.
Cite specific districts, zone names, and km2 figures from the context above."""


# ──────────────────────────────────────────────
# APP
# ──────────────────────────────────────────────

app = FastAPI(title="Chronostasis OpenEnv", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

import os as _os
_static = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "static")
if _os.path.isdir(_static):
    app.mount("/static", StaticFiles(directory=_static), name="static")

@app.get("/", include_in_schema=False)
async def root():
    idx = _os.path.join(_static, "index.html")
    if _os.path.isfile(idx):
        return FileResponse(idx)
    return {"name": "Chronostasis", "status": "running", "docs": "/docs"}


# ──────────────────────────────────────────────
# CORE OPENENV ENDPOINTS
# ──────────────────────────────────────────────

@app.post("/reset", response_model=FloodObservation)
async def reset(request: ResetRequest = ResetRequest()):
    global _current_episode
    task_id   = request.task_id   or "flood_year_comparison"
    region_id = request.region_id or DEFAULT_REGION

    if task_id not in TASK_REGISTRY:
        raise HTTPException(400, f"Unknown task '{task_id}'. Available: {list(TASK_REGISTRY.keys())}")
    if region_id not in REGIONS:
        raise HTTPException(400, f"Unknown region '{region_id}'. Available: {list(REGIONS.keys())}")

    task = TASK_REGISTRY[task_id](gee_available=GEE_AVAILABLE, region=region_id)
    _current_episode = EpisodeState(task, region_id)
    region = REGIONS[region_id]

    return FloodObservation(
        task_id=task_id, task_description=task.description,
        step=0, max_steps=task.max_steps, available_data=task.available_data,
        context=task.get_context(), echoed_message=f"Task started: {task.description}",
        region_id=region_id, region_name=region["name"]
    )


@app.post("/step", response_model=StepResult)
async def step(action: FloodAction):
    global _current_episode
    if _current_episode is None:
        raise HTTPException(400, "No active episode. Call /reset first.")
    if _current_episode.done:
        raise HTTPException(400, "Episode finished. Call /reset to start a new one.")

    ep = _current_episode
    ep.step += 1
    result = ep.task.step(action.message, ep.step)
    reward = float(result["reward"])
    done   = bool(result["done"]) or ep.step >= ep.task.max_steps
    error  = result.get("error")

    ep.total_reward += reward
    ep.done = done
    ep.history.append({"step": ep.step, "action": action.message[:200], "reward": reward, "done": done})

    region = REGIONS[ep.region_id]
    obs = FloodObservation(
        task_id=ep.task.task_id, task_description=ep.task.description,
        step=ep.step, max_steps=ep.task.max_steps, available_data=ep.task.available_data,
        last_action_result=result.get("result",""), last_action_error=error,
        context=ep.task.get_context(), echoed_message=action.message,
        region_id=ep.region_id, region_name=region["name"]
    )
    return StepResult(observation=obs, reward=reward, done=done,
                      info={"total_reward": ep.total_reward, "episode_id": ep.episode_id},
                      last_action_error=error)


@app.get("/state", response_model=FloodState)
async def state():
    if _current_episode is None:
        raise HTTPException(400, "No active episode.")
    ep = _current_episode
    return FloodState(episode_id=ep.episode_id, task_id=ep.task.task_id,
                      region_id=ep.region_id, step=ep.step, max_steps=ep.task.max_steps,
                      total_reward=ep.total_reward, done=ep.done, history=ep.history,
                      gee_available=GEE_AVAILABLE, started_at=ep.started_at)


# ──────────────────────────────────────────────
# REAL AGENT ENDPOINT
# ──────────────────────────────────────────────

@app.post("/agent/step")
async def agent_step():
    """Run ONE real LLM agent step against the current episode."""
    global _current_episode
    if _current_episode is None:
        raise HTTPException(400, "No active episode. Call /reset first.")
    if _current_episode.done:
        raise HTTPException(400, "Episode is done.")

    client = get_llm_client()
    if not client:
        raise HTTPException(503, "No LLM API key configured (HF_TOKEN or OPENAI_API_KEY).")

    ep = _current_episode
    prompt = build_agent_prompt(ep)

    try:
        completion = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise GIS flood analysis agent. Always include specific numbers, district names, and km2 figures in your responses."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.3,
        )
        message = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        raise HTTPException(502, f"LLM call failed: {exc}")

    # Now step the environment with the LLM response
    ep.step += 1
    result = ep.task.step(message, ep.step)
    reward = float(result["reward"])
    done   = bool(result["done"]) or ep.step >= ep.task.max_steps
    ep.total_reward += reward
    ep.done = done
    ep.history.append({"step": ep.step, "action": message[:200], "reward": reward, "done": done})

    return {
        "step":          ep.step,
        "agent_message": message,
        "reward":        reward,
        "done":          done,
        "result":        result.get("result", ""),
        "total_reward":  ep.total_reward,
        "model":         AGENT_MODEL,
    }


# ──────────────────────────────────────────────
# META ENDPOINTS
# ──────────────────────────────────────────────

@app.get("/regions", response_model=List[RegionInfo])
async def list_regions():
    return [
        RegionInfo(id=rid, name=r["name"], state=r["state"], river=r["river"],
                   peak_year=r["peak_year"], accuracy_pct=r["accuracy_pct"],
                   flood_areas={str(k): v for k, v in r["flood_areas"].items()})
        for rid, r in REGIONS.items()
    ]

@app.get("/tasks", response_model=List[TaskInfo])
async def list_tasks():
    tasks = []
    for tid, tcls in TASK_REGISTRY.items():
        t = tcls(gee_available=GEE_AVAILABLE, region=DEFAULT_REGION)
        tasks.append(TaskInfo(id=tid, name=t.name, description=t.description,
                              difficulty=t.difficulty, max_steps=t.max_steps,
                              region_id=DEFAULT_REGION))
    return tasks

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "gee_available": GEE_AVAILABLE,
        "llm_configured": bool(HF_TOKEN or os.getenv("OPENAI_API_KEY")),
        "agent_model": AGENT_MODEL,
        "regions": list(REGIONS.keys()),
        "tasks": list(TASK_REGISTRY.keys()),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
