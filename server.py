"""
server.py — Chronostasis OpenEnv Environment Server v2.0
=========================================================
Multi-region flood intelligence environment for 15 Indian river basins.
4-Agent pipeline (Data Analyst → Domain Expert → Critic → Aggregator)
all running on the trained RL model (chronostasis-3b-grpo-medium).
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
from gee_codegen import generate_gee_code, generate_multi_basin_comparison_code
from gee_client import (init_gee as init_gee_client, gee_available,
                         get_stats_or_mock, get_flood_tile_url,
                         query_any_location)
from renderer import render_flood_report
from dotenv import load_dotenv
load_dotenv()


# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
GEE_PROJECT   = os.getenv("GEE_PROJECT", "your-gee-project-id")
HF_TOKEN      = os.getenv("HF_TOKEN", "") or os.getenv("API_KEY", "")

TRAINED_MODEL = os.getenv("TRAINED_MODEL",      "LunaAmagi/chronostasis-3b-grpo-medium")
BASE_MODEL    = os.getenv("BASE_MODEL",          "Qwen/Qwen2.5-72B-Instruct")
USE_TRAINED   = os.getenv("USE_TRAINED_MODEL",   "true").lower() != "false"
MODEL_NAME    = TRAINED_MODEL if USE_TRAINED else BASE_MODEL

HF_INFERENCE_URL = f"https://api-inference.huggingface.co/models/{TRAINED_MODEL}"
HF_ROUTER_URL    = "https://router.huggingface.co/v1"

# Agent temperature — low = deterministic, good for trained model
AGENT_TEMPERATURE = 0.2
MAX_TOKENS        = 300


# ─────────────────────────────────────────────────────────
# GEE INIT
# ─────────────────────────────────────────────────────────
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
# Also init the gee_client module
if GEE_AVAILABLE:
    init_gee_client(
        project=GEE_PROJECT,
        sa_json=os.getenv("GEE_SERVICE_ACCOUNT_JSON")
    )


# ─────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────
# EPISODE STATE
# ─────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────
# LLM — TRAINED MODEL VIA HF INFERENCE API
# ─────────────────────────────────────────────────────────
def call_trained_model(prompt: str, max_retries: int = 3) -> str:
    """
    Calls trained RL model via HF Inference API.
    Prompt should already be formatted with Qwen2.5 chat template.
    """
    if not HF_TOKEN:
        raise ValueError("No HF_TOKEN configured")

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type":  "application/json",
    }
    payload = json.dumps({
        "inputs": prompt,
        "parameters": {
            "max_new_tokens":   MAX_TOKENS,
            "temperature":      AGENT_TEMPERATURE,
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
                raise ValueError(f"Unexpected: {str(result)[:100]}")

            # Strip Qwen chat template end token
            return text.replace("<|im_end|>", "").strip()

        except Exception as e:
            print(f"[DEBUG] Trained model attempt {attempt+1}: {e}", flush=True)
            if attempt == max_retries - 1:
                raise
            time.sleep(2)
    return ""


def call_base_model(system: str, user: str) -> str:
    """Calls Qwen2.5-72B via HF router as fallback."""
    if not HF_TOKEN:
        return ""
    try:
        client = OpenAI(base_url=HF_ROUTER_URL, api_key=HF_TOKEN)
        completion = client.chat.completions.create(
            model=BASE_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=MAX_TOKENS,
            temperature=AGENT_TEMPERATURE,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] Base model failed: {exc}", flush=True)
        return ""


def call_llm(system: str, user: str) -> str:
    """
    Smart dispatcher:
    1. Try trained RL model (HF Inference API)
    2. Fall back to base Qwen2.5-72B (HF router)
    """
    if USE_TRAINED and HF_TOKEN:
        # Qwen2.5 chat template format
        formatted = (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        try:
            result = call_trained_model(formatted)
            if result:
                return result
        except Exception as e:
            print(f"[INFO] Trained model unavailable ({e}), using base model", flush=True)

    if HF_TOKEN:
        result = call_base_model(system, user)
        if result:
            return result

    return ""


# ─────────────────────────────────────────────────────────
# 4-AGENT PIPELINE
# ─────────────────────────────────────────────────────────
def _ctx_str(region: dict) -> str:
    """Build compact context string from region data."""
    fa  = region["flood_areas"]
    rz  = region["risk_zones_km2"]
    fa_str = ", ".join(
        f"{yr}: {fa.get(yr, fa.get(str(yr), 0))} km²" for yr in [2022, 2023, 2024])
    return (
        f"River: {region['river']} | State: {region['state']}\n"
        f"SAR flood areas: {fa_str}\n"
        f"Peak year: {region['peak_year']} | Model accuracy: {region['accuracy_pct']}%\n"
        f"Chronic area: {region['chronic_km2']} km² | Population: {region['chronic_pop']:,}\n"
        f"Chronic districts: {', '.join(region['chronic_districts'])}\n"
        f"High-risk zones: {', '.join(region['high_risk_zones'])}\n"
        f"Risk zones km²: high={rz['high']}, moderate={rz['moderate']}, low={rz['low']}\n"
        f"Peak rainfall: {region['peak_rainfall_mm']} mm"
    )


def agent_1_data_analyst(task_desc: str, ctx: str, history: str) -> str:
    """Agent 1 — Extracts exact numbers: km², rainfall, population."""
    system = (
        "You are a GIS Data Analyst specialising in Sentinel-1 SAR flood data. "
        "Your ONLY job: report EXACT numbers from the context. "
        "Include flood extent km² for each year, CHIRPS rainfall mm, "
        "population counts, accuracy percentages. "
        "Never use vague language. Every figure must be cited."
    )
    user = (
        f"Task: {task_desc}\n\n"
        f"Data context:\n{ctx}\n\n"
        f"Previous steps: {history or 'None'}\n\n"
        "Report ONLY exact numeric data: km² flood extents, rainfall totals, "
        "population numbers, accuracy metrics. One paragraph, prose only."
    )
    return call_llm(system, user)


def agent_2_domain_expert(task_desc: str, ctx: str, history: str) -> str:
    """Agent 2 — District names and GIS causal factors."""
    system = (
        "You are a Senior GIS and Hydrology Expert for South Asian river systems. "
        "Name specific districts affected by flooding. "
        "Explain causal factors: DEM elevation, HydroSHEDS flow accumulation, "
        "CHIRPS rainfall anomalies, slope. "
        "Identify high-risk geographic zones by name. "
        "Always cite at least 3 districts and 2 causal factors."
    )
    user = (
        f"Task: {task_desc}\n\n"
        f"Context:\n{ctx}\n\n"
        f"Previous steps: {history or 'None'}\n\n"
        "Name specific districts, explain causal GIS factors, "
        "identify flood-prone zones. One paragraph, prose only."
    )
    return call_llm(system, user)


def agent_3_critic(task_desc: str, analyst: str, expert: str) -> str:
    """Agent 3 — Finds gaps before the answer is submitted."""
    system = (
        "You are a strict scientific peer reviewer for flood intelligence reports. "
        "Identify MISSING information in the two answers below. "
        "Look for: missing km² figures, missing district names, "
        "vague unsupported claims, missing causal factors. "
        "Be specific. Under 80 words."
    )
    user = (
        f"Task: {task_desc}\n\n"
        f"DATA ANALYST:\n{analyst}\n\n"
        f"DOMAIN EXPERT:\n{expert}\n\n"
        "What is MISSING? List specific gaps only."
    )
    return call_llm(system, user)


def agent_4_aggregator(task_desc: str, ctx: str,
                        analyst: str, expert: str, critic: str) -> str:
    """Agent 4 — Final answer combining all three inputs."""
    system = (
        "You are a Chief Flood Intelligence Officer writing the official final report. "
        "Combine the Data Analyst's exact numbers and the Domain Expert's GIS knowledge. "
        "Fix every gap the Critic identified. "
        "Your answer MUST include: "
        "(1) Exact km² figures for flood extents, "
        "(2) At least 3 specific district names, "
        "(3) Causal factors: CHIRPS rainfall, DEM elevation, HydroSHEDS flow accumulation, "
        "(4) No vague claims — every statement backed by data. "
        "Under 280 words. Prose only, no bullet points."
    )
    user = (
        f"Task: {task_desc}\n\n"
        f"Context:\n{ctx}\n\n"
        f"DATA ANALYST:\n{analyst}\n\n"
        f"DOMAIN EXPERT:\n{expert}\n\n"
        f"CRITIC GAPS:\n{critic}\n\n"
        "Write the final comprehensive answer. Fix ALL gaps. "
        "Include exact km² figures, district names, causal factors."
    )
    return call_llm(system, user)


def run_four_agent_pipeline(ep: EpisodeState) -> str:
    """
    Full 4-agent debate pipeline.

    Agent 1 (numbers) ──┐
                         ├──▶ Agent 3 (critic) ──▶ Agent 4 (final) ──▶ /step
    Agent 2 (GIS)    ──┘

    Each agent calls the same trained RL model with different system prompts.
    """
    region   = REGIONS[ep.region_id]
    ctx      = _ctx_str(region)
    task_desc = ep.task.description
    history  = ""
    if ep.history:
        history = " | ".join(
            f"step {h['step']}: reward={h['reward']:.3f}"
            for h in ep.history[-3:]
        )

    print(f"[AGENT] Starting 4-agent pipeline | step={ep.step+1} | "
          f"task={ep.task.task_id} | region={ep.region_id} | "
          f"model={'trained' if USE_TRAINED else 'base'}", flush=True)

    # Agent 1
    analyst = agent_1_data_analyst(task_desc, ctx, history)
    print(f"[AGENT 1] Data Analyst: {len(analyst)} chars", flush=True)

    # Agent 2
    expert = agent_2_domain_expert(task_desc, ctx, history)
    print(f"[AGENT 2] Domain Expert: {len(expert)} chars", flush=True)

    # Agent 3 — only if we have both inputs
    if analyst and expert:
        critic = agent_3_critic(task_desc, analyst, expert)
        print(f"[AGENT 3] Critic: {len(critic)} chars", flush=True)
    else:
        critic = "No gaps identified — insufficient input from previous agents."

    # Agent 4 — final aggregation
    if analyst or expert:
        final = agent_4_aggregator(task_desc, ctx, analyst, expert, critic)
        print(f"[AGENT 4] Aggregator: {len(final)} chars", flush=True)
    else:
        final = ""

    # Fallback if all agents failed
    if not final:
        fallbacks = {
            "flood_year_comparison":     "Floods vary by year during monsoon season.",
            "district_inundation_report": "Several districts experience recurring flooding.",
            "flood_risk_forecast":        "Certain areas face higher flood risk.",
        }
        final = fallbacks.get(ep.task.task_id, "Flood analysis unavailable.")
        print("[AGENT] All agents failed — using fallback", flush=True)

    return final


# ─────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────
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

@app.get("/map", include_in_schema=False)
async def flood_map():
    """Serves the interactive Leaflet flood risk map."""
    mp = _os.path.join(_static, "map.html")
    if _os.path.isfile(mp):
        return FileResponse(mp)
    raise HTTPException(404, "map.html not found in static/")


# ─────────────────────────────────────────────────────────
# CORE OPENENV ENDPOINTS
# ─────────────────────────────────────────────────────────

@app.post("/reset", response_model=FloodObservation)
async def reset(request: ResetRequest = ResetRequest()):
    global _current_episode
    task_id   = request.task_id   or "flood_year_comparison"
    region_id = request.region_id or DEFAULT_REGION
    season    = request.season    or "kharif"

    if task_id not in TASK_REGISTRY:
        raise HTTPException(400, f"Unknown task '{task_id}'. "
                            f"Available: {list(TASK_REGISTRY.keys())}")
    if region_id not in REGIONS:
        raise HTTPException(400, f"Unknown region '{region_id}'. "
                            f"Available: {list(REGIONS.keys())}")

    task = TASK_REGISTRY[task_id](
        gee_available=GEE_AVAILABLE, region=region_id, season=season)
    _current_episode = EpisodeState(task, region_id)
    region = REGIONS[region_id]

    return FloodObservation(
        task_id=task_id, task_description=task.description,
        step=0, max_steps=task.max_steps, available_data=task.available_data,
        context=task.get_context(),
        echoed_message=f"Episode started: {task.description}",
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
        "step":   ep.step,
        "action": action.message[:200],
        "reward": reward,
        "done":   done,
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


# ─────────────────────────────────────────────────────────
# 4-AGENT ENDPOINT — FULL PIPELINE
# ─────────────────────────────────────────────────────────

@app.post("/agent/step")
async def agent_step(request: AgentStepRequest = AgentStepRequest()):
    """
    Runs the full 4-agent debate pipeline using the trained RL model.

    Pipeline:
      Agent 1 (Data Analyst)  — exact km², rainfall, population
      Agent 2 (Domain Expert) — districts, GIS causal factors
      Agent 3 (Critic)        — identifies gaps
      Agent 4 (Aggregator)    — final answer → submitted to /step grader

    Falls back to base Qwen2.5-72B if trained model unavailable.
    """
    global _current_episode
    try:
        if not HF_TOKEN:
            raise HTTPException(503, "No HF_TOKEN in Space secrets.")

        task_id   = request.task_id   or "flood_year_comparison"
        region_id = request.region_id or DEFAULT_REGION
        season    = request.season    or "kharif"

        if task_id not in TASK_REGISTRY:
            raise HTTPException(400, f"Unknown task: {task_id}")
        if region_id not in REGIONS:
            raise HTTPException(400, f"Unknown region: {region_id}")

        # Reset if needed — handles multi-replica state isolation
        if (_current_episode is None
                or _current_episode.done
                or _current_episode.task.task_id != task_id
                or _current_episode.region_id != region_id):
            task = TASK_REGISTRY[task_id](
                gee_available=GEE_AVAILABLE, region=region_id, season=season)
            _current_episode = EpisodeState(task, region_id)

        ep = _current_episode

        # Run 4-agent pipeline
        final_answer = run_four_agent_pipeline(ep)

        if not final_answer:
            raise HTTPException(502, "4-agent pipeline returned empty response")

        # Grade the aggregated answer
        ep.step += 1
        result  = ep.task.step(final_answer, ep.step)
        reward  = float(result.get("reward", 0) or 0)
        done    = bool(result.get("done", False)) or ep.step >= ep.task.max_steps
        ep.total_reward = round(ep.total_reward + reward, 4)
        ep.done = done
        ep.history.append({
            "step":   ep.step,
            "action": final_answer[:200],
            "reward": reward,
            "done":   done,
        })

        return {
            "step":            ep.step,
            "agent_message":   final_answer,
            "reward":          reward,
            "done":            done,
            "result":          result.get("result", ""),
            "total_reward":    ep.total_reward,
            "model":           MODEL_NAME,
            "using_trained":   USE_TRAINED,
            "pipeline":        "4-agent",
            "task_id":         task_id,
            "region_id":       region_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Agent error: {type(e).__name__}: {str(e)[:300]}")


# ─────────────────────────────────────────────────────────
# COMPARE ENDPOINT — TRAINED vs BASELINE
# ─────────────────────────────────────────────────────────

@app.post("/agent/compare")
async def agent_compare(request: CompareRequest):
    """
    Runs the 4-agent system AND a vague baseline on the same task/region.
    Returns side-by-side responses + reward scores for demo purposes.
    """
    task_id   = request.task_id
    region_id = request.region_id

    if task_id not in TASK_REGISTRY:
        raise HTTPException(400, f"Unknown task: {task_id}")
    if region_id not in REGIONS:
        raise HTTPException(400, f"Unknown region: {region_id}")

    task   = TASK_REGISTRY[task_id](gee_available=GEE_AVAILABLE, region=region_id)
    region = REGIONS[region_id]

    # ── 4-agent response ───────────────────────
    trained_response = ""
    trained_score    = 0.01
    try:
        # Create a temporary episode for the comparison
        tmp_ep = EpisodeState(task, region_id)
        trained_response = run_four_agent_pipeline(tmp_ep)
        if trained_response:
            trained_score = float(
                task.step(trained_response, step_num=1).get("reward", 0.01))
    except Exception as e:
        trained_response = f"[Pipeline error: {str(e)[:80]}]"

    # ── Vague baseline ─────────────────────────
    baseline_map = {
        "flood_year_comparison":      "Floods in Indian river basins vary by year during monsoon season. Some years are more severe than others.",
        "district_inundation_report": "Several districts experience recurring flooding. Populations in low-lying areas are affected annually.",
        "flood_risk_forecast":        "Certain areas face higher flood risk. Early warning systems may help reduce flood impact.",
    }
    baseline_response = baseline_map.get(task_id, "Flooding occurs in this region.")
    try:
        baseline_score = float(
            task.step(baseline_response, step_num=1).get("reward", 0.01))
    except:
        baseline_score = 0.01

    return {
        "task_id":    task_id,
        "region_id":  region_id,
        "pipeline":   "4-agent",
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
        "improvement":         round(trained_score - baseline_score, 3),
        "using_trained_model": USE_TRAINED,
    }


# ─────────────────────────────────────────────────────────
# MODEL INFO
# ─────────────────────────────────────────────────────────

@app.get("/model/info")
async def model_info():
    return {
        "active_model":        MODEL_NAME,
        "trained_model":       TRAINED_MODEL,
        "base_model":          BASE_MODEL,
        "using_trained_model": USE_TRAINED,
        "hf_token_configured": bool(HF_TOKEN),
        "pipeline":            "4-agent",
        "agents": [
            "Agent 1: Data Analyst — exact km², rainfall, population",
            "Agent 2: Domain Expert — districts, GIS causal factors",
            "Agent 3: Critic — identifies gaps",
            "Agent 4: Aggregator — final graded answer",
        ],
        "inference_api":  HF_INFERENCE_URL,
        "hf_model_url":   f"https://huggingface.co/{TRAINED_MODEL}",
    }


# ─────────────────────────────────────────────────────────
# META ENDPOINTS
# ─────────────────────────────────────────────────────────

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
    return [
        {
            "id":          tid,
            "name":        tcls(gee_available=GEE_AVAILABLE, region=DEFAULT_REGION).name,
            "description": tcls(gee_available=GEE_AVAILABLE, region=DEFAULT_REGION).description,
            "difficulty":  tcls(gee_available=GEE_AVAILABLE, region=DEFAULT_REGION).difficulty,
            "max_steps":   tcls(gee_available=GEE_AVAILABLE, region=DEFAULT_REGION).max_steps,
        }
        for tid, tcls in TASK_REGISTRY.items()
    ]


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
    task_id = (_current_episode.task.task_id
               if _current_episode else "flood_year_comparison")
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
        "status":              "ok",
        "gee_available":       GEE_AVAILABLE,
        "llm_configured":      bool(HF_TOKEN),
        "using_trained_model": USE_TRAINED,
        "agent_model":         MODEL_NAME,
        "trained_model":       TRAINED_MODEL,
        "pipeline":            "4-agent",
        "regions":             list(REGIONS.keys()),
        "tasks":               list(TASK_REGISTRY.keys()),
        "version":             "2.0.0",
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
            {"id": "pre_monsoon",  "label": "Pre-Monsoon",
             "months": "March–May",        "desc": "Dry season, localised storm risk"},
            {"id": "kharif",       "label": "Kharif",
             "months": "June–September",   "desc": "Peak monsoon, maximum flood risk"},
            {"id": "post_monsoon", "label": "Post-Monsoon",
             "months": "October–November", "desc": "Receding waters, secondary flood risk"},
            {"id": "rabi",         "label": "Rabi",
             "months": "December–February","desc": "Winter season, minimal flood risk"},
        ]
    }


# ─────────────────────────────────────────────────────────
# DYNAMIC GEE QUERY — ANY LOCATION IN INDIA
# ─────────────────────────────────────────────────────────

class LocationQuery(BaseModel):
    lat:       float
    lon:       float
    radius_km: float = 80.0
    year:      int   = 2022

@app.post("/query/location")
async def query_location(req: LocationQuery):
    """
    Query real SAR flood data for any lat/lon in India.
    Returns flood extent, risk zones, rainfall, and tile URLs for Leaflet.
    """
    if req.lat < 6 or req.lat > 38 or req.lon < 66 or req.lon > 98:
        raise HTTPException(400, "Coordinates outside India bounds")
    if req.radius_km < 10 or req.radius_km > 500:
        raise HTTPException(400, "radius_km must be 10–500")
    if req.year not in [2022, 2023, 2024]:
        raise HTTPException(400, "year must be 2022, 2023, or 2024")

    result = get_stats_or_mock(req.lat, req.lon, req.radius_km)
    result["year"] = req.year

    # Add tile URLs for the requested year
    if gee_available():
        tiles = get_flood_tile_url(req.lat, req.lon, req.year, req.radius_km * 2)
        result["tiles"] = tiles.get("tiles", {})

    return result


@app.get("/query/tiles")
async def query_tiles(lat: float, lon: float, year: int = 2022,
                      radius_km: float = 150):
    """
    Returns Leaflet-compatible tile URLs for flood risk visualization.
    Use in frontend: L.tileLayer(url).addTo(map)
    """
    if not gee_available():
        return {
            "mock": True,
            "message": "GEE not configured — tile URLs unavailable",
            "tiles": {},
        }
    result = get_flood_tile_url(lat, lon, year, radius_km)
    return result


@app.post("/query/reset")
async def query_reset(req: LocationQuery):
    """
    Resets episode using real GEE data for any lat/lon.
    Dynamically builds region context from satellite data.
    """
    global _current_episode

    # Get real data
    stats = get_stats_or_mock(req.lat, req.lon, req.radius_km)

    # Build dynamic region data
    dynamic_region = {
        "name":             f"Custom Region ({req.lat:.2f}°N, {req.lon:.2f}°E)",
        "state":            "India",
        "river":            "Dynamic Query",
        "lat":              req.lat,
        "lon":              req.lon,
        "flood_areas":      stats.get("flood_areas_km2", {2022:0,2023:0,2024:0}),
        "peak_year":        stats.get("peak_year", 2022),
        "chronic_km2":      stats.get("chronic_km2", 0),
        "chronic_pop":      0,
        "chronic_districts": [],
        "high_risk_zones":  ["Query Area"],
        "accuracy_pct":     91.0,
        "risk_zones_km2":   stats.get("risk_zones_km2", {"high":0,"moderate":0,"low":0}),
        "peak_rainfall_mm": stats.get("peak_rainfall_mm", 0),
        "seasonal_risk":    {"pre_monsoon":0.3,"kharif":0.9,"post_monsoon":0.5,"rabi":0.1},
        "sar_threshold_db": -16,
    }

    return {
        "status":       "ok",
        "region":       dynamic_region,
        "tiles":        stats.get("tiles", {}),
        "mock":         stats.get("mock", True),
        "gee_available": gee_available(),
    }


# ─────────────────────────────────────────────────────────
# GEE CODE DOWNLOAD ENDPOINTS
# ─────────────────────────────────────────────────────────

@app.get("/gee/code")
async def gee_code(region_id: str = "brahmaputra", year: int = 2022):
    """
    Returns downloadable GEE JavaScript for SAR flood analysis.
    Paste into code.earthengine.google.com to run.
    """
    if region_id not in REGIONS:
        raise HTTPException(400, f"Unknown region: {region_id}. "
                            f"Available: {list(REGIONS.keys())}")
    if year not in [2022, 2023, 2024]:
        raise HTTPException(400, "Year must be 2022, 2023, or 2024")

    region = REGIONS[region_id]
    code   = generate_gee_code(region_id, region, year)

    from fastapi.responses import Response
    filename = f"chronostasis_{region_id}_flood_{year}.js"
    return Response(
        content=code,
        media_type="application/javascript",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Region":  region_id,
            "X-Year":    str(year),
            "X-Basin":   region["name"],
        }
    )


@app.get("/gee/code/all")
async def gee_code_all():
    """
    Returns downloadable GEE JavaScript for all 15 basins comparison.
    Single script showing all-India risk map.
    """
    code = generate_multi_basin_comparison_code(REGIONS)

    from fastapi.responses import Response
    return Response(
        content=code,
        media_type="application/javascript",
        headers={
            "Content-Disposition": 'attachment; filename="chronostasis_all_india_flood.js"',
        }
    )


@app.get("/gee/info")
async def gee_info():
    """Returns info about available GEE code downloads."""
    return {
        "description": "Download GEE JavaScript for SAR flood analysis",
        "usage": "Paste downloaded .js file into code.earthengine.google.com",
        "endpoints": {
            "single_basin": "/gee/code?region_id={region_id}&year={year}",
            "all_basins":   "/gee/code/all",
        },
        "available_regions": list(REGIONS.keys()),
        "available_years":   [2022, 2023, 2024],
        "layers_included": [
            "Sentinel-1 SAR VV monsoon composite",
            "Flood extent mask (per year)",
            "Chronic inundation (all 3 years)",
            "CHIRPS rainfall overlay",
            "SRTM DEM + slope",
            "HydroSHEDS flow accumulation",
            "Multi-factor risk zones (high/moderate/low)",
            "Export-ready GeoTIFF scripts",
        ],
        "example": "/gee/code?region_id=brahmaputra&year=2022",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)