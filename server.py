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

# ── Load .env for local dev (HF Space uses its own secrets) ──────────────
from dotenv import load_dotenv
load_dotenv()

import ee
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from openai import OpenAI

from tasks import TASK_REGISTRY, REGIONS, DEFAULT_REGION, BaseTask
from renderer import render_flood_report

# Safe imports for GEE helpers (added in later commits)
try:
    from gee_client import (
        init_gee as init_gee_client,
        gee_available as gee_client_available,
        get_stats_or_mock,
        get_flood_tile_url,
        query_any_location,
    )
    HAS_GEE_CLIENT = True
except ImportError:
    HAS_GEE_CLIENT = False
    def get_stats_or_mock(lat, lon, r=80):
        return {"mock": True, "lat": lat, "lon": lon,
                "flood_areas_km2": {2022: 4812.3, 2023: 3601.7, 2024: 4102.8},
                "peak_year": 2022, "chronic_km2": 1823.4,
                "risk_zones_km2": {"high": 3218.4, "moderate": 5901.2, "low": 8234.7},
                "peak_rainfall_mm": 1587, "tiles": {}}
    def get_flood_tile_url(*a, **k): return {"mock": True, "tiles": {}}
    def query_any_location(lat, lon, r=80): return get_stats_or_mock(lat, lon, r)

try:
    from gee_codegen import generate_gee_code, generate_multi_basin_comparison_code
    HAS_GEE_CODEGEN = True
except ImportError:
    HAS_GEE_CODEGEN = False
    def generate_gee_code(region_id, region_data, year=2022):
        return f"// GEE code for {region_id} {year} — gee_codegen.py not found"
    def generate_multi_basin_comparison_code(regions):
        return "// All-India GEE comparison — gee_codegen.py not found"

# ──────────────────────────────────────────────
# CONFIG — all values read from .env / HF secrets
# ──────────────────────────────────────────────

GEE_PROJECT      = os.getenv("GEE_PROJECT", "chronostasis-gee")

# ── LLM: trained RL model (primary) → HF router (fallback) ───────────────
HF_TOKEN         = os.getenv("HF_TOKEN", "")
TRAINED_MODEL    = os.getenv("TRAINED_MODEL", "LunaAmagi/chronostasis-3b-grpo-medium")
BASE_MODEL       = os.getenv("BASE_MODEL",    "Qwen/Qwen2.5-72B-Instruct")
USE_TRAINED      = os.getenv("USE_TRAINED_MODEL", "true").lower() != "false"
MODEL_NAME       = TRAINED_MODEL if USE_TRAINED else BASE_MODEL
HF_ROUTER_URL    = "https://router.huggingface.co/v1"
HF_INFERENCE_URL = "https://api-inference.huggingface.co/v1"

# ──────────────────────────────────────────────
# GEE INIT
# ──────────────────────────────────────────────

def init_gee():
    """Initialize GEE — uses gee_client if available, else inline."""
    if HAS_GEE_CLIENT:
        return init_gee_client(
            project=GEE_PROJECT,
            sa_json=os.getenv("GEE_SERVICE_ACCOUNT_JSON")
        )
    # Inline fallback
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
        print(f"[WARN] GEE init failed: {exc} — running in mock mode", flush=True)
        return False

GEE_AVAILABLE = init_gee()

# ──────────────────────────────────────────────
# LLM HELPERS
# ──────────────────────────────────────────────

def call_trained_model(messages: list, max_tokens: int = 350) -> str:
    """Call the trained RL model directly via HF Inference API."""
    client = OpenAI(base_url=HF_INFERENCE_URL, api_key=HF_TOKEN)
    completion = client.chat.completions.create(
        model=TRAINED_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return (completion.choices[0].message.content or "").strip()


def call_base_model(messages: list, max_tokens: int = 350) -> str:
    """Call base model via HF router (OpenAI-compatible)."""
    client = OpenAI(base_url=HF_ROUTER_URL, api_key=HF_TOKEN)
    completion = client.chat.completions.create(
        model=BASE_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return (completion.choices[0].message.content or "").strip()


def call_llm(messages: list, max_tokens: int = 350) -> str:
    """Smart dispatcher: trained model first, base model fallback."""
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not set — add it to .env or HF Space secrets")
    if USE_TRAINED:
        try:
            result = call_trained_model(messages, max_tokens)
            if result:
                return result
        except Exception as e:
            print(f"[WARN] Trained model failed ({e}), falling back to base", flush=True)
    return call_base_model(messages, max_tokens)


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
    observation:       FloodObservation
    reward:            float
    done:              bool
    info:              Dict[str, Any] = Field(default_factory=dict)
    last_action_error: Optional[str] = None

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
    season:    Optional[str] = "kharif"

class AgentStepRequest(BaseModel):
    task_id:   Optional[str] = None
    region_id: Optional[str] = None

class LocationRequest(BaseModel):
    lat:       float
    lon:       float
    radius_km: Optional[float] = 80.0
    year:      Optional[int]   = 2022

class RegionInfo(BaseModel):
    id:          str
    name:        str
    state:       str
    river:       str
    peak_year:   int
    accuracy_pct: float
    flood_areas: Dict[str, float]

class TaskInfo(BaseModel):
    id:          str
    name:        str
    description: str
    difficulty:  str
    max_steps:   int
    region_id:   str

# ──────────────────────────────────────────────
# EPISODE STATE
# ──────────────────────────────────────────────

class EpisodeState:
    def __init__(self, task: BaseTask, region_id: str):
        self.episode_id  = str(uuid.uuid4())
        self.task        = task
        self.region_id   = region_id
        self.step        = 0
        self.done        = False
        self.total_reward = 0.0
        self.history: List[Dict[str, Any]] = []
        self.started_at  = time.time()

_current_episode: Optional[EpisodeState] = None

# ──────────────────────────────────────────────
# APP
# ──────────────────────────────────────────────

app = FastAPI(title="Chronostasis OpenEnv", version="2.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_static = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(_static):
    app.mount("/static", StaticFiles(directory=_static), name="static")


@app.get("/", include_in_schema=False)
async def root():
    idx = os.path.join(_static, "index.html")
    if os.path.isfile(idx):
        return FileResponse(idx)
    return {"name": "Chronostasis", "version": "2.1.0", "docs": "/docs"}


@app.get("/map", include_in_schema=False)
async def map_page():
    path = os.path.join(_static, "map.html")
    if os.path.isfile(path):
        return FileResponse(path)
    raise HTTPException(404, "map.html not found in static/")


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
        context=task.get_context(),
        echoed_message=f"Task started: {task.description}",
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
    reward = float(result.get("reward", 0) or 0)
    done   = bool(result.get("done", False)) or ep.step >= ep.task.max_steps
    error  = result.get("error")

    ep.total_reward = round(ep.total_reward + reward, 4)
    ep.done = done
    ep.history.append({
        "step": ep.step, "action": action.message[:200],
        "reward": reward, "done": done
    })

    region = REGIONS[ep.region_id]
    obs = FloodObservation(
        task_id=ep.task.task_id, task_description=ep.task.description,
        step=ep.step, max_steps=ep.task.max_steps,
        available_data=ep.task.available_data,
        last_action_result=result.get("result", ""),
        last_action_error=error,
        context=ep.task.get_context(),
        echoed_message=action.message,
        region_id=ep.region_id, region_name=region["name"]
    )
    return StepResult(
        observation=obs, reward=reward, done=done,
        info={"total_reward": ep.total_reward, "episode_id": ep.episode_id},
        last_action_error=error
    )


@app.get("/state", response_model=FloodState)
async def state():
    if _current_episode is None:
        raise HTTPException(400, "No active episode.")
    ep = _current_episode
    return FloodState(
        episode_id=ep.episode_id, task_id=ep.task.task_id,
        region_id=ep.region_id, step=ep.step, max_steps=ep.task.max_steps,
        total_reward=ep.total_reward, done=ep.done, history=ep.history,
        gee_available=GEE_AVAILABLE, started_at=ep.started_at
    )


# ──────────────────────────────────────────────
# TRAINED RL AGENT ENDPOINT (4-agent pipeline)
# ──────────────────────────────────────────────

@app.post("/agent/step")
async def agent_step(request: AgentStepRequest = AgentStepRequest()):
    """
    4-agent debate pipeline using trained RL model.
    Agents: Data Analyst → Domain Expert → Critic → Aggregator.
    """
    global _current_episode

    if not HF_TOKEN:
        raise HTTPException(503, "HF_TOKEN not configured. Add to .env or HF Space secrets.")

    task_id   = request.task_id   or "flood_year_comparison"
    region_id = request.region_id or DEFAULT_REGION

    if task_id not in TASK_REGISTRY:
        raise HTTPException(400, f"Unknown task: {task_id}")
    if region_id not in REGIONS:
        raise HTTPException(400, f"Unknown region: {region_id}")

    # Reuse or create episode
    if (_current_episode is None
            or _current_episode.done
            or _current_episode.task.task_id != task_id
            or _current_episode.region_id != region_id):
        task = TASK_REGISTRY[task_id](gee_available=GEE_AVAILABLE, region=region_id)
        _current_episode = EpisodeState(task, region_id)

    ep     = _current_episode
    region = REGIONS[region_id]

    fa    = region["flood_areas"]
    fa_str = ", ".join(f"{yr}={fa.get(yr, fa.get(str(yr), 0))}" for yr in [2022, 2023, 2024])
    rz    = region["risk_zones_km2"]
    history_txt = "\n".join(
        f"Step {h['step']}: reward={h['reward']:.2f} | {h['action'][:80]}"
        for h in ep.history[-3:]
    ) if ep.history else "None yet"

    context_block = "\n".join([
        f"River: {region['river']}",
        f"Model accuracy: {region['accuracy_pct']}%",
        f"Flood areas km²: {fa_str}",
        f"Peak flood year: {region['peak_year']}",
        f"Chronic area: {region['chronic_km2']} km²",
        f"Population at risk: {region['chronic_pop']:,}",
        f"Chronic districts: {', '.join(region['chronic_districts'])}",
        f"High-risk zones: {', '.join(region['high_risk_zones'])}",
        f"Risk zones km²: high={rz['high']}, moderate={rz['moderate']}, low={rz['low']}",
        f"Peak rainfall: {region['peak_rainfall_mm']}mm",
    ])

    sys_msg = ("You are a precise GIS flood analyst. "
               "Always cite exact km² figures, district names, and causal factors. "
               "Respond in clear prose only — never write code.")

    task_prompt = (
        f"Region: {region['name']} ({region['state']})\n"
        f"Task: {ep.task.description}\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"Step {ep.step + 1} of {ep.task.max_steps}\n"
        f"Previous steps: {history_txt}\n\n"
    )

    try:
        # Agent 1 — Data Analyst
        a1 = call_llm([
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content": task_prompt +
             "Extract and report exact flood extent figures (km²) for each year. "
             "Identify the peak year with justification."}
        ], max_tokens=200)

        # Agent 2 — Domain Expert
        a2 = call_llm([
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content": task_prompt +
             f"Data Analyst found: {a1[:300]}\n\n"
             "Add district names, CHIRPS rainfall values, DEM elevation context, "
             "and HydroSHEDS flow accumulation analysis."}
        ], max_tokens=200)

        # Agent 3 — Critic
        a3 = call_llm([
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content":
             f"Review these flood analysis responses and identify any missing data, "
             f"vague claims, or unsupported statements:\n\nAnalysis 1: {a1[:300]}\n"
             f"Analysis 2: {a2[:300]}\n\nList specific gaps that need addressing."}
        ], max_tokens=150)

        # Agent 4 — Aggregator
        final = call_llm([
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content":
             f"Synthesise these analyses into one comprehensive response. "
             f"Address the critic's concerns. Include all exact numbers.\n\n"
             f"Analysis 1: {a1[:300]}\nAnalysis 2: {a2[:300]}\n"
             f"Critic: {a3[:200]}\n\n"
             f"Write the final integrated flood analysis."}
        ], max_tokens=350)

    except Exception as exc:
        raise HTTPException(502, f"LLM error: {type(exc).__name__}: {str(exc)[:250]}")

    # Grade
    ep.step += 1
    result = ep.task.step(final, ep.step)
    reward = float(result.get("reward", 0) or 0)
    done   = bool(result.get("done", False)) or ep.step >= ep.task.max_steps

    ep.total_reward = round(ep.total_reward + reward, 4)
    ep.done = done
    ep.history.append({"step": ep.step, "action": final[:200], "reward": reward, "done": done})

    return {
        "step":          ep.step,
        "agent_message": final,
        "reward":        reward,
        "done":          done,
        "result":        result.get("result", ""),
        "total_reward":  ep.total_reward,
        "model":         MODEL_NAME,
        "using_trained": USE_TRAINED,
        "task_id":       task_id,
        "region_id":     region_id,
        "agents": {
            "data_analyst":  a1[:200],
            "domain_expert": a2[:200],
            "critic":        a3[:150],
            "aggregator":    final[:200],
        }
    }


@app.post("/agent/compare")
async def agent_compare(request: AgentStepRequest = AgentStepRequest()):
    """Compare trained model vs vague baseline side-by-side."""
    task_id   = request.task_id   or "flood_year_comparison"
    region_id = request.region_id or DEFAULT_REGION

    if task_id not in TASK_REGISTRY or region_id not in REGIONS:
        raise HTTPException(400, "Invalid task_id or region_id")

    region = REGIONS[region_id]
    fa = region["flood_areas"]
    fa_str = ", ".join(f"{yr}={fa.get(yr, fa.get(str(yr), 0))}" for yr in [2022, 2023, 2024])
    prompt_ctx = (
        f"Region: {region['name']}, River: {region['river']}\n"
        f"Task: {TASK_REGISTRY[task_id](gee_available=GEE_AVAILABLE, region=region_id).description}\n"
        f"Flood areas km²: {fa_str}, Peak year: {region['peak_year']}\n"
        f"Answer in prose with exact figures."
    )

    sys_msg = "You are a precise GIS flood analyst. Cite exact km² figures, district names, causal factors."

    baseline = "Floods in Indian river basins vary by year during monsoon season."

    try:
        trained = call_llm([
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content": prompt_ctx}
        ], max_tokens=250)
    except Exception as e:
        trained = f"Error: {e}"

    # Score both
    task = TASK_REGISTRY[task_id](gee_available=GEE_AVAILABLE, region=region_id)
    baseline_score = float(task.step(baseline,  1).get("reward", 0))
    task2          = TASK_REGISTRY[task_id](gee_available=GEE_AVAILABLE, region=region_id)
    trained_score  = float(task2.step(trained, 1).get("reward", 0))

    return {
        "task_id":        task_id,
        "region_id":      region_id,
        "baseline":       {"response": baseline, "reward": baseline_score},
        "trained":        {"response": trained,  "reward": trained_score},
        "improvement":    round(trained_score - baseline_score, 3),
        "model":          MODEL_NAME,
    }


# ──────────────────────────────────────────────
# GEE / MAP ENDPOINTS
# ──────────────────────────────────────────────

@app.post("/query/location")
async def query_location(req: LocationRequest):
    """Query real GEE data for any lat/lon point in India."""
    result = query_any_location(req.lat, req.lon, req.radius_km)
    tiles  = get_flood_tile_url(req.lat, req.lon, req.year, req.radius_km * 2)
    result["tiles"] = tiles.get("tiles", {})
    return result


@app.get("/query/tiles")
async def query_tiles(lat: float, lon: float, year: int = 2022, radius_km: float = 200):
    """Get Leaflet tile URLs for flood risk visualization."""
    return get_flood_tile_url(lat, lon, year, radius_km)


@app.get("/gee/code")
async def gee_code(region_id: str = "brahmaputra", year: int = 2022):
    """Download GEE JavaScript flood analysis script for a basin."""
    if region_id not in REGIONS:
        raise HTTPException(400, f"Unknown region: {region_id}")
    code = generate_gee_code(region_id, REGIONS[region_id], year)
    from fastapi.responses import Response
    return Response(
        content=code,
        media_type="application/javascript",
        headers={"Content-Disposition": f'attachment; filename="chronostasis_{region_id}_{year}.js"'}
    )


@app.get("/gee/code/all")
async def gee_code_all():
    """Download GEE script covering all 15 Indian river basins."""
    code = generate_multi_basin_comparison_code(REGIONS)
    from fastapi.responses import Response
    return Response(
        content=code,
        media_type="application/javascript",
        headers={"Content-Disposition": 'attachment; filename="chronostasis_all_india.js"'}
    )


@app.get("/india_risk_map")
async def india_risk_map(season: str = "kharif"):
    """Returns flood risk data for all regions — for map visualization."""
    season_multipliers = {
        "kharif":      {"brahmaputra": 0.95, "ganga": 0.88, "mahanadi": 0.82, "krishna": 0.75, "godavari": 0.79},
        "pre-monsoon": {"brahmaputra": 0.45, "ganga": 0.38, "mahanadi": 0.35, "krishna": 0.28, "godavari": 0.32},
        "post-monsoon":{"brahmaputra": 0.60, "ganga": 0.55, "mahanadi": 0.50, "krishna": 0.42, "godavari": 0.48},
        "rabi":        {"brahmaputra": 0.10, "ganga": 0.12, "mahanadi": 0.08, "krishna": 0.05, "godavari": 0.07},
    }
    mults = season_multipliers.get(season, season_multipliers["kharif"])
    return {
        rid: {
            "name":         r["name"],
            "state":        r["state"],
            "river":        r["river"],
            "lat":          r.get("lat", 26.0),
            "lon":          r.get("lon", 90.0),
            "risk_level":   "high" if r["risk_zones_km2"]["high"] > 3000 else "moderate",
            "seasonal_risk": mults.get(rid, 0.5),
            "peak_year":    r["peak_year"],
            "accuracy_pct": r["accuracy_pct"],
            "chronic_pop":  r["chronic_pop"],
            "flood_areas":  {str(k): v for k, v in r["flood_areas"].items()},
        }
        for rid, r in REGIONS.items()
    }


@app.get("/seasons")
async def seasons():
    return {
        "seasons": [
            {"id": "pre-monsoon",  "name": "Pre-Monsoon",   "months": "Mar–May",  "risk": "moderate"},
            {"id": "kharif",       "name": "Kharif Monsoon","months": "Jun–Sep",  "risk": "high"},
            {"id": "post-monsoon", "name": "Post-Monsoon",  "months": "Oct–Nov",  "risk": "moderate-low"},
            {"id": "rabi",         "name": "Rabi / Winter", "months": "Dec–Feb",  "risk": "low"},
        ]
    }


@app.get("/model/info")
async def model_info():
    return {
        "trained_model":   TRAINED_MODEL,
        "base_model":      BASE_MODEL,
        "active_model":    MODEL_NAME,
        "using_trained":   USE_TRAINED,
        "hf_token_set":    bool(HF_TOKEN),
        "trained_models": [
            "LunaAmagi/chronostasis-sft-trained",
            "LunaAmagi/chronostasis-3b-grpo-medium",
            "LunaAmagi/chronostasis-3b-grpo-hard",
        ]
    }


# ──────────────────────────────────────────────
# META ENDPOINTS
# ──────────────────────────────────────────────

@app.get("/regions")
async def list_regions():
    return [
        {
            "id": rid, "name": r["name"], "state": r["state"],
            "river": r["river"], "peak_year": r["peak_year"],
            "accuracy_pct": r["accuracy_pct"],
            "flood_areas": {str(k): v for k, v in r["flood_areas"].items()},
            "lat": r.get("lat", 26.0), "lon": r.get("lon", 90.0),
        }
        for rid, r in REGIONS.items()
    ]


@app.get("/tasks")
async def list_tasks():
    tasks = []
    for tid, tcls in TASK_REGISTRY.items():
        t = tcls(gee_available=GEE_AVAILABLE, region=DEFAULT_REGION)
        tasks.append({
            "id": tid, "name": t.name, "description": t.description,
            "difficulty": t.difficulty, "max_steps": t.max_steps,
        })
    return tasks


@app.get("/report")
async def report():
    ep = _current_episode
    region_id = ep.region_id if ep else DEFAULT_REGION
    r = REGIONS[region_id]
    return {
        "region_id":       region_id,
        "region_name":     r["name"],
        "state":           r["state"],
        "river":           r["river"],
        "flood_areas":     {str(k): v for k, v in r["flood_areas"].items()},
        "peak_year":       r["peak_year"],
        "chronic_km2":     r["chronic_km2"],
        "chronic_pop":     r["chronic_pop"],
        "chronic_districts": r["chronic_districts"],
        "high_risk_zones": r["high_risk_zones"],
        "accuracy_pct":    r["accuracy_pct"],
        "risk_zones_km2":  r["risk_zones_km2"],
        "peak_rainfall_mm": r["peak_rainfall_mm"],
        "episode": {
            "task_id":      ep.task.task_id,
            "total_reward": ep.total_reward,
            "steps":        ep.step,
            "done":         ep.done,
            "history":      ep.history,
        } if ep else None,
        "all_regions_summary": [
            {
                "id":            rid,
                "name":          rv["name"],
                "peak_year":     rv["peak_year"],
                "peak_flood_km2": rv["flood_areas"][rv["peak_year"]],
                "chronic_km2":   rv["chronic_km2"],
                "accuracy_pct":  rv["accuracy_pct"],
            }
            for rid, rv in REGIONS.items()
        ]
    }


@app.post("/render")
async def render(request: ResetRequest = ResetRequest()):
    region_id = request.region_id or DEFAULT_REGION
    if region_id not in REGIONS:
        raise HTTPException(400, f"Unknown region: {region_id}")
    region  = REGIONS[region_id]
    history = _current_episode.history if _current_episode else []
    task_id = _current_episode.task.task_id if _current_episode else "flood_year_comparison"
    try:
        charts = render_flood_report(region, history, task_id)
        return {"region_id": region_id, "region_name": region["name"],
                "charts": charts, "chart_names": list(charts.keys())}
    except Exception as e:
        raise HTTPException(500, f"Render failed: {str(e)[:200]}")


@app.get("/health")
async def health():
    return {
        "status":             "ok",
        "version":            "2.1.0",
        "gee_available":      GEE_AVAILABLE,
        "llm_configured":     bool(HF_TOKEN),
        "using_trained_model": USE_TRAINED,
        "agent_model":        MODEL_NAME,
        "trained_model":      TRAINED_MODEL,
        "regions":            list(REGIONS.keys()),
        "tasks":              list(TASK_REGISTRY.keys()),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))