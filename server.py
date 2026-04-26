"""
server.py — Chronostasis OpenEnv Environment Server v2.0
=========================================================
Multi-region flood intelligence environment for 15 Indian river basins.
Uses trained RL model (chronostasis-3b-grpo-medium) via HF Inference API.
Falls back to Qwen2.5-72B via HF router if trained model unavailable.
"""

import json
import os
import time
import uuid
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

import ee
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from openai import OpenAI

from tasks import TASK_REGISTRY, REGIONS, DEFAULT_REGION, BaseTask
from renderer import render_flood_report


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
GEE_PROJECT    = os.getenv("GEE_PROJECT", "your-gee-project-id")
HF_TOKEN       = os.getenv("HF_TOKEN", "") or os.getenv("API_KEY", "")

TRAINED_MODEL  = os.getenv("TRAINED_MODEL",       "LunaAmagi/chronostasis-3b-grpo-medium")
BASE_MODEL     = os.getenv("BASE_MODEL",           "Qwen/Qwen2.5-72B-Instruct")
USE_TRAINED    = os.getenv("USE_TRAINED_MODEL",    "true").lower() != "false"
MODEL_NAME     = TRAINED_MODEL if USE_TRAINED else BASE_MODEL

HF_INFERENCE_URL = f"https://api-inference.huggingface.co/models/{TRAINED_MODEL}"
HF_ROUTER_URL    = "https://router.huggingface.co/v1"

SYSTEM_PROMPT = (
    "You are a precise GIS flood analyst for Indian river basins. "
    "Always cite exact km² figures, district names, CHIRPS rainfall totals, "
    "DEM elevation values, HydroSHEDS flow accumulation, and causal factors. "
    "Never be vague. Respond in clear prose only."
)


# ──────────────────────────────────────────────
# GEE INIT
# ──────────────────────────────────────────────
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
    season:    Optional[str] = "kharif"

class AgentStepRequest(BaseModel):
    task_id:   Optional[str] = None
    region_id: Optional[str] = None
    season:    Optional[str] = "kharif"

class CompareRequest(BaseModel):
    task_id:   str = "flood_year_comparison"
    region_id: str = "brahmaputra"

class TaskInfo(BaseModel):
    id:          str
    name:        str
    description: str
    difficulty:  str
    max_steps:   int
    region_id:   str

class RegionInfo(BaseModel):
    id:           str
    name:         str
    state:        str
    river:        str
    peak_year:    int
    accuracy_pct: float
    flood_areas:  Dict[str, float]


# ──────────────────────────────────────────────
# EPISODE STATE
# ──────────────────────────────────────────────
class EpisodeState:
    def __init__(self, task: BaseTask, region_id: str):
        self.episode_id    = str(uuid.uuid4())
        self.task          = task
        self.region_id     = region_id
        self.step          = 0
        self.done          = False
        self.total_reward  = 0.0
        self.history: List[Dict[str, Any]] = []
        self.started_at    = time.time()

_current_episode: Optional[EpisodeState] = None


# ──────────────────────────────────────────────
# LLM CLIENTS
# ──────────────────────────────────────────────
def get_router_client() -> OpenAI:
    """HF router — OpenAI-compatible, used for base model fallback."""
    return OpenAI(base_url=HF_ROUTER_URL, api_key=HF_TOKEN)


def call_trained_model(prompt: str, max_retries: int = 3) -> str:
    """
    Calls the fine-tuned RL model via HF Inference API.
    Returns generated text or raises on failure.
    """
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type":  "application/json",
    }
    payload = json.dumps({
        "inputs": prompt,
        "parameters": {
            "max_new_tokens":   300,
            "temperature":      0.2,
            "do_sample":        True,
            "return_full_text": False,
        },
    }).encode()

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                HF_INFERENCE_URL, data=payload, method="POST", headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                result = json.loads(r.read().decode())
                if isinstance(result, list) and result:
                    text = result[0].get("generated_text", "").strip()
                elif isinstance(result, dict) and "generated_text" in result:
                    text = result["generated_text"].strip()
                elif isinstance(result, dict) and "error" in result:
                    if "loading" in result["error"].lower() and attempt < max_retries - 1:
                        wait = min(result.get("estimated_time", 20), 30)
                        print(f"[INFO] Model loading, waiting {wait}s...", flush=True)
                        time.sleep(wait)
                        continue
                    raise ValueError(result["error"])
                else:
                    raise ValueError(f"Unexpected response: {str(result)[:100]}")
                # Strip chat template tokens
                return text.replace("<|im_end|>", "").strip()
        except Exception as e:
            print(f"[DEBUG] Trained model attempt {attempt+1} failed: {e}", flush=True)
            if attempt == max_retries - 1:
                raise
            time.sleep(2)
    return ""


def call_base_model(client: OpenAI, user_prompt: str) -> str:
    """Calls base Qwen2.5-72B via HF router as fallback."""
    try:
        completion = client.chat.completions.create(
            model=BASE_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=300,
            temperature=0.2,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] Base model failed: {exc}", flush=True)
        return ""


def call_llm(user_prompt: str) -> str:
    """
    Routes to trained RL model first, falls back to base model.
    For trained model, wraps prompt in Qwen2.5 chat template.
    """
    if USE_TRAINED and HF_TOKEN:
        formatted = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        try:
            result = call_trained_model(formatted)
            if result:
                print(f"[INFO] Trained model used: {TRAINED_MODEL}", flush=True)
                return result
        except Exception as e:
            print(f"[INFO] Trained model unavailable ({e}), using base model", flush=True)

    if HF_TOKEN:
        client = get_router_client()
        result = call_base_model(client, user_prompt)
        if result:
            print(f"[INFO] Base model used: {BASE_MODEL}", flush=True)
            return result

    return "Flood data analysis unavailable — no LLM configured."


# ──────────────────────────────────────────────
# PROMPT BUILDER
# ──────────────────────────────────────────────
def build_agent_prompt(ep: EpisodeState) -> str:
    region = REGIONS[ep.region_id]
    fa  = region["flood_areas"]
    rz  = region["risk_zones_km2"]
    fa_str = ", ".join(f"{yr}={fa.get(yr, fa.get(str(yr), 0))} km²" for yr in [2022, 2023, 2024])

    history_txt = ""
    if ep.history:
        lines = [f"Step {h['step']}: reward={h['reward']:.3f} | {h['action'][:80]}"
                 for h in ep.history[-3:]]
        history_txt = "\n".join(lines)

    return "\n".join([
        f"Task: {ep.task.description}",
        f"",
        f"Region: {region['name']}, State: {region['state']}, River: {region['river']}",
        f"SAR flood areas: {fa_str}",
        f"Peak flood year: {region['peak_year']}",
        f"Model accuracy: {region['accuracy_pct']}%",
        f"Chronic inundation: {region['chronic_km2']} km² | Population: {region['chronic_pop']:,}",
        f"Chronic districts: {', '.join(region['chronic_districts'])}",
        f"High-risk zones: {', '.join(region['high_risk_zones'])}",
        f"Risk zones km²: high={rz['high']}, moderate={rz['moderate']}, low={rz['low']}",
        f"Peak rainfall: {region['peak_rainfall_mm']}mm",
        f"",
        f"Step {ep.step + 1} of {ep.task.max_steps}",
        f"Previous steps:\n{history_txt}" if history_txt else "No previous steps.",
        f"",
        f"Provide a specific, data-backed analysis with exact km² figures, district names, "
        f"and causal factors (CHIRPS rainfall, DEM elevation, HydroSHEDS flow accumulation).",
    ])


# ──────────────────────────────────────────────
# APP
# ──────────────────────────────────────────────
app = FastAPI(title="Chronostasis OpenEnv", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

import os as _os
_static = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "static")
if _os.path.isdir(_static):
    app.mount("/static", StaticFiles(directory=_static), name="static")

@app.get("/", include_in_schema=False)
async def root():
    idx = _os.path.join(_static, "index.html")
    if _os.path.isfile(idx):
        return FileResponse(idx)
    return {"name": "Chronostasis", "version": "2.0.0", "docs": "/docs"}


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

    season = request.season or "kharif"
    task   = TASK_REGISTRY[task_id](gee_available=GEE_AVAILABLE, region=region_id, season=season)
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
        step=ep.step, max_steps=ep.task.max_steps, available_data=ep.task.available_data,
        last_action_result=result.get("result", ""), last_action_error=error,
        context=ep.task.get_context(), echoed_message=action.message,
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
# AGENT ENDPOINT — TRAINED RL MODEL
# ──────────────────────────────────────────────

@app.post("/agent/step")
async def agent_step(request: AgentStepRequest = AgentStepRequest()):
    """
    Runs the trained RL model (or base model fallback) on the current task.
    Self-contained: resets episode internally to handle multi-replica deployments.
    """
    global _current_episode
    try:
        if not HF_TOKEN:
            raise HTTPException(503, "No HF_TOKEN configured in Space secrets.")

        task_id   = request.task_id   or "flood_year_comparison"
        region_id = request.region_id or DEFAULT_REGION
        season    = request.season    or "kharif"

        if task_id not in TASK_REGISTRY:
            raise HTTPException(400, f"Unknown task: {task_id}")
        if region_id not in REGIONS:
            raise HTTPException(400, f"Unknown region: {region_id}")

        # Reset if needed (handles multi-replica state isolation)
        if (_current_episode is None
                or _current_episode.done
                or _current_episode.task.task_id != task_id
                or _current_episode.region_id != region_id):
            task = TASK_REGISTRY[task_id](
                gee_available=GEE_AVAILABLE, region=region_id, season=season)
            _current_episode = EpisodeState(task, region_id)

        ep     = _current_episode
        prompt = build_agent_prompt(ep)

        # Call trained model (or fallback)
        message = call_llm(prompt)

        if not message:
            raise HTTPException(502, "LLM returned empty response")

        # Grade the response
        ep.step += 1
        result  = ep.task.step(message, ep.step)
        reward  = float(result.get("reward", 0) or 0)
        done    = bool(result.get("done", False)) or ep.step >= ep.task.max_steps
        ep.total_reward = round(ep.total_reward + reward, 4)
        ep.done = done
        ep.history.append({
            "step": ep.step, "action": message[:200],
            "reward": reward, "done": done
        })

        return {
            "step":              ep.step,
            "agent_message":     message,
            "reward":            reward,
            "done":              done,
            "result":            result.get("result", ""),
            "total_reward":      ep.total_reward,
            "model":             MODEL_NAME,
            "using_trained":     USE_TRAINED,
            "task_id":           task_id,
            "region_id":         region_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Agent error: {type(e).__name__}: {str(e)[:300]}")


# ──────────────────────────────────────────────
# COMPARE ENDPOINT — TRAINED vs BASELINE
# ──────────────────────────────────────────────

@app.post("/agent/compare")
async def agent_compare(request: CompareRequest):
    """
    Runs the trained RL model AND a vague baseline on the same task/region.
    Returns side-by-side responses + reward scores.
    """
    task_id   = request.task_id
    region_id = request.region_id

    if task_id not in TASK_REGISTRY:
        raise HTTPException(400, f"Unknown task: {task_id}")
    if region_id not in REGIONS:
        raise HTTPException(400, f"Unknown region: {region_id}")

    task   = TASK_REGISTRY[task_id](gee_available=GEE_AVAILABLE, region=region_id)
    region = REGIONS[region_id]
    fa     = region["flood_areas"]
    rz     = region["risk_zones_km2"]
    fa_str = ", ".join(f"{yr}={fa.get(yr, fa.get(str(yr), 0))} km²" for yr in [2022, 2023, 2024])

    prompt = "\n".join([
        f"Task: {task.description}",
        f"Region: {region['name']}, River: {region['river']}",
        f"Flood areas: {fa_str}",
        f"Peak year: {region['peak_year']}",
        f"Model accuracy: {region['accuracy_pct']}%",
        f"Chronic area: {region['chronic_km2']} km²",
        f"Chronic districts: {', '.join(region['chronic_districts'])}",
        f"High-risk zones: {', '.join(region['high_risk_zones'])}",
        f"Risk zones km²: high={rz['high']}, moderate={rz['moderate']}, low={rz['low']}",
        f"Answer in prose with exact km² figures, district names, and causal factors.",
    ])

    # ── Trained model ──────────────────────────
    trained_response = ""
    trained_score    = 0.01
    try:
        trained_response = call_llm(prompt)
        if trained_response:
            trained_score = float(task.step(trained_response, step_num=1).get("reward", 0.01))
    except Exception as e:
        trained_response = f"[Model error: {str(e)[:80]}]"

    # ── Vague baseline ─────────────────────────
    baseline_map = {
        "flood_year_comparison":     "Floods in Indian river basins vary by year during monsoon season. Some years are more severe than others.",
        "district_inundation_report": "Several districts experience recurring flooding. Populations in low-lying areas are affected annually.",
        "flood_risk_forecast":        "Certain areas face higher flood risk. Early warning systems may help reduce flood impact.",
    }
    baseline_response = baseline_map.get(task_id, "Flooding occurs in this region.")
    try:
        baseline_score = float(task.step(baseline_response, step_num=1).get("reward", 0.01))
    except:
        baseline_score = 0.01

    return {
        "task_id":     task_id,
        "region_id":   region_id,
        "trained": {
            "model":    MODEL_NAME,
            "response": trained_response,
            "reward":   round(trained_score, 3),
        },
        "baseline": {
            "model":    "vague_baseline",
            "response": baseline_response,
            "reward":   round(baseline_score, 3),
        },
        "improvement":       round(trained_score - baseline_score, 3),
        "using_trained_model": USE_TRAINED,
    }


# ──────────────────────────────────────────────
# MODEL INFO
# ──────────────────────────────────────────────

@app.get("/model/info")
async def model_info():
    return {
        "active_model":         MODEL_NAME,
        "trained_model":        TRAINED_MODEL,
        "base_model":           BASE_MODEL,
        "using_trained_model":  USE_TRAINED,
        "hf_token_configured":  bool(HF_TOKEN),
        "inference_api":        HF_INFERENCE_URL,
        "hf_model_url":         f"https://huggingface.co/{TRAINED_MODEL}",
    }


# ──────────────────────────────────────────────
# META ENDPOINTS
# ──────────────────────────────────────────────

@app.get("/regions")
async def list_regions():
    return [
        {
            "id":           rid,
            "name":         r["name"],
            "state":        r["state"],
            "river":        r["river"],
            "peak_year":    r["peak_year"],
            "accuracy_pct": r["accuracy_pct"],
            "flood_areas":  {str(k): v for k, v in r["flood_areas"].items()},
            "lat":          r.get("lat"),
            "lon":          r.get("lon"),
        }
        for rid, r in REGIONS.items()
    ]


@app.get("/tasks")
async def list_tasks():
    tasks = []
    for tid, tcls in TASK_REGISTRY.items():
        t = tcls(gee_available=GEE_AVAILABLE, region=DEFAULT_REGION)
        tasks.append({
            "id":          tid,
            "name":        t.name,
            "description": t.description,
            "difficulty":  t.difficulty,
            "max_steps":   t.max_steps,
        })
    return tasks


@app.get("/report")
async def report():
    ep        = _current_episode
    region_id = ep.region_id if ep else DEFAULT_REGION
    r         = REGIONS[region_id]
    return {
        "region_id":         region_id,
        "region_name":       r["name"],
        "state":             r["state"],
        "river":             r["river"],
        "flood_areas":       {str(k): v for k, v in r["flood_areas"].items()},
        "peak_year":         r["peak_year"],
        "chronic_km2":       r["chronic_km2"],
        "chronic_pop":       r["chronic_pop"],
        "chronic_districts": r["chronic_districts"],
        "high_risk_zones":   r["high_risk_zones"],
        "accuracy_pct":      r["accuracy_pct"],
        "risk_zones_km2":    r["risk_zones_km2"],
        "peak_rainfall_mm":  r["peak_rainfall_mm"],
        "episode": {
            "task_id":      ep.task.task_id if ep else None,
            "total_reward": ep.total_reward if ep else 0,
            "steps":        ep.step if ep else 0,
            "done":         ep.done if ep else False,
            "history":      ep.history if ep else [],
        },
        "all_regions_summary": [
            {
                "id":             rid,
                "name":           rv["name"],
                "peak_year":      rv["peak_year"],
                "peak_flood_km2": rv["flood_areas"][rv["peak_year"]],
                "chronic_km2":    rv["chronic_km2"],
                "accuracy_pct":   rv["accuracy_pct"],
            }
            for rid, rv in REGIONS.items()
        ],
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
        return {
            "region_id":   region_id,
            "region_name": region["name"],
            "charts":      charts,
            "chart_names": list(charts.keys()),
        }
    except Exception as e:
        raise HTTPException(500, f"Render failed: {str(e)[:200]}")


@app.get("/health")
async def health():
    return {
        "status":               "ok",
        "gee_available":        GEE_AVAILABLE,
        "llm_configured":       bool(HF_TOKEN),
        "using_trained_model":  USE_TRAINED,
        "agent_model":          MODEL_NAME,
        "trained_model":        TRAINED_MODEL,
        "regions":              list(REGIONS.keys()),
        "tasks":                list(TASK_REGISTRY.keys()),
        "version":              "2.0.0",
    }


@app.get("/india_risk_map")
async def india_risk_map(season: str = "kharif"):
    valid = ["pre_monsoon", "kharif", "post_monsoon", "rabi"]
    if season not in valid:
        season = "kharif"

    result = []
    for rid, r in REGIONS.items():
        sr       = r.get("seasonal_risk", {}).get(season, 0.5)
        high_km2 = r["risk_zones_km2"]["high"]

        if sr >= 0.80 and high_km2 >= 2000:
            risk_level = "critical"
        elif sr >= 0.60 and high_km2 >= 1000:
            risk_level = "high"
        elif sr >= 0.30:
            risk_level = "moderate"
        else:
            risk_level = "low"

        result.append({
            "id":              rid,
            "name":            r["name"],
            "state":           r["state"],
            "river":           r["river"],
            "lat":             r.get("lat"),
            "lon":             r.get("lon"),
            "risk_level":      risk_level,
            "seasonal_risk":   sr,
            "season":          season,
            "peak_year":       r["peak_year"],
            "accuracy_pct":    r["accuracy_pct"],
            "chronic_pop":     r["chronic_pop"],
            "chronic_km2":     r["chronic_km2"],
            "high_risk_km2":   high_km2,
            "peak_flood_km2":  r["flood_areas"][r["peak_year"]],
            "high_risk_zones": r["high_risk_zones"],
        })

    return {
        "season":        season,
        "season_desc":   f"Flood risk for {season.replace('_', ' ')} season",
        "total_regions": len(result),
        "regions":       result,
    }


@app.get("/seasons")
async def list_seasons():
    return {
        "seasons": [
            {"id": "pre_monsoon",  "label": "Pre-Monsoon",  "months": "March–May",        "desc": "Dry season, localised storm risk"},
            {"id": "kharif",       "label": "Kharif",        "months": "June–September",   "desc": "Peak monsoon, maximum flood risk"},
            {"id": "post_monsoon", "label": "Post-Monsoon",  "months": "October–November", "desc": "Receding waters, secondary flood risk"},
            {"id": "rabi",         "label": "Rabi",          "months": "December–February","desc": "Winter season, minimal flood risk"},
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)