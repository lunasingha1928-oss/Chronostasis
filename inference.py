"""
inference.py — Chronostasis 4-Agent Flood Intelligence System
=============================================================
Uses the trained RL model (chronostasis-3b-grpo-medium) via HF Inference API.
Falls back to base Qwen2.5-72B if trained model is unavailable.

Agent pipeline:
  Agent 1 (Data Analyst)   → exact km², rainfall, population figures
  Agent 2 (Domain Expert)  → district names, causal GIS factors
  Agent 3 (Critic)         → finds gaps before submission
  Agent 4 (Aggregator)     → combines all into final answer → /step
"""

import asyncio
import json
import os
import time
import urllib.request
import urllib.error
from typing import List, Optional
from openai import OpenAI


# ── Configuration ──────────────────────────────────────────────────────────
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://LunaAmagi-chronostasis.hf.space")
BENCHMARK    = os.getenv("CHRONOSTASIS_BENCH", "chronostasis")
REGION_ID    = os.getenv("CHRONOSTASIS_REGION", "brahmaputra")

# Model selection
# Priority: trained RL model → base model fallback
TRAINED_MODEL   = os.getenv("TRAINED_MODEL",  "LunaAmagi/chronostasis-3b-grpo-medium")
BASE_MODEL      = os.getenv("BASE_MODEL",     "Qwen/Qwen2.5-72B-Instruct")
USE_TRAINED     = os.getenv("USE_TRAINED_MODEL", "true").lower() != "false"
MODEL_NAME      = TRAINED_MODEL if USE_TRAINED else BASE_MODEL

# HF Inference API for trained model
HF_INFERENCE_URL = f"https://api-inference.huggingface.co/models/{TRAINED_MODEL}"

# HF Router for base model (OpenAI-compatible)
HF_ROUTER_URL   = "https://router.huggingface.co/v1"

ALL_TASKS = [
    "flood_year_comparison",
    "district_inundation_report",
    "flood_risk_forecast",
]
TASK_NAME             = os.getenv("MY_ENV_V4_TASK", ALL_TASKS[0])
MAX_STEPS             = 8
TEMPERATURE           = 0.2        # lower = more deterministic for trained model
MAX_TOKENS            = 350
SUCCESS_SCORE_THRESHOLD = 0.5


# ── Logging ────────────────────────────────────────────────────────────────
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    action_clean = action.replace("\n", " ").replace("\r", "").strip()[:200]
    print(f"[STEP] step={step} action={action_clean!r} "
          f"reward={reward:.2f} done={str(done).lower()} "
          f"error={error if error else 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"score={score:.3f} rewards={rewards_str}", flush=True)


# ── HTTP Environment Client ────────────────────────────────────────────────
def env_request(path, method="GET", body=None):
    url  = ENV_BASE_URL.rstrip("/") + path
    data = json.dumps(body or {}).encode()
    req  = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:100]}"}
    except Exception as ex:
        return {"error": str(ex)}

def env_reset(task_id):
    return env_request("/reset", "POST", {"task_id": task_id, "region_id": REGION_ID})

def env_step(message):
    return env_request("/step", "POST", {"message": message})


# ── Vague fallbacks (baseline — low reward by design) ─────────────────────
FALLBACKS = {
    "flood_year_comparison": [
        "Floods in Indian river basins vary by year during monsoon season.",
        "Based on available data, some years appear to have more flooding.",
    ],
    "district_inundation_report": [
        "Several districts experience recurring flooding.",
        "Populations in low-lying areas are affected annually.",
    ],
    "flood_risk_forecast": [
        "Certain areas face higher flood risk based on historical patterns.",
        "Early warning systems may help reduce flood impact.",
    ],
}


# ── Trained model inference (HF Inference API) ────────────────────────────
def call_trained_model(prompt: str, max_retries: int = 3) -> str:
    """
    Calls the fine-tuned RL model via HF Inference API.
    The model is hosted at LunaAmagi/chronostasis-3b-grpo-medium.
    Uses direct text-generation endpoint (not OpenAI-compatible).
    """
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens":  MAX_TOKENS,
            "temperature":     TEMPERATURE,
            "do_sample":       True,
            "return_full_text": False,
        },
    }

    for attempt in range(max_retries):
        try:
            data = json.dumps(payload).encode()
            req  = urllib.request.Request(
                HF_INFERENCE_URL, data=data, method="POST",
                headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                result = json.loads(r.read().decode())
                # HF Inference API returns list: [{"generated_text": "..."}]
                if isinstance(result, list) and result:
                    return result[0].get("generated_text", "").strip()
                elif isinstance(result, dict) and "generated_text" in result:
                    return result["generated_text"].strip()
                elif isinstance(result, dict) and "error" in result:
                    # Model loading — retry
                    if "loading" in result["error"].lower() and attempt < max_retries - 1:
                        wait = result.get("estimated_time", 20)
                        print(f"[INFO] Model loading, waiting {wait}s...", flush=True)
                        time.sleep(min(wait, 30))
                        continue
                    raise ValueError(result["error"])
        except Exception as e:
            print(f"[DEBUG] Trained model attempt {attempt+1} failed: {e}", flush=True)
            if attempt == max_retries - 1:
                raise
            time.sleep(3)
    return ""


# ── Base model inference (HF Router, OpenAI-compatible) ───────────────────
def call_base_model(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    """Calls Qwen2.5-72B via HF router as OpenAI-compatible fallback."""
    try:
        completion = client.chat.completions.create(
            model=BASE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] Base model call failed: {exc}", flush=True)
        return ""


# ── Smart dispatcher ───────────────────────────────────────────────────────
def call_llm(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    """
    Routes to trained model first, falls back to base model.
    For the trained model, formats the chat template manually since
    HF Inference API doesn't use OpenAI format.
    """
    if USE_TRAINED:
        # Format as Qwen2.5 chat template
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        try:
            result = call_trained_model(prompt)
            # Strip any trailing im_end tokens
            result = result.replace("<|im_end|>", "").strip()
            if result:
                print(f"[INFO] Used trained model: {TRAINED_MODEL}", flush=True)
                return result
        except Exception as e:
            print(f"[INFO] Trained model unavailable ({e}), falling back to base", flush=True)

    # Fallback to base model
    return call_base_model(client, system_prompt, user_prompt)


# ── 4-Agent Pipeline ───────────────────────────────────────────────────────
def build_context(obs: dict) -> str:
    """Extract useful context from observation for agent prompts."""
    ctx = obs.get("context", {})
    if isinstance(ctx, str):
        return ctx[:400]
    if isinstance(ctx, dict):
        return json.dumps(ctx)[:400]
    return str(ctx)[:400]


def agent_data_analyst(client, obs, step, task_id):
    """Agent 1: Focus on exact numbers — km², rainfall, population."""
    system = (
        "You are a GIS Data Analyst specialising in Sentinel-1 SAR flood data for Indian river basins. "
        "Your ONLY job is to report EXACT numbers from the context: "
        "flood extent in km² for each year, CHIRPS rainfall totals in mm, "
        "population counts, district areas, accuracy percentages. "
        "Be specific. Never use vague language. Cite every figure you state."
    )
    user = (
        f"Task: {obs.get('task_description', task_id)}\n"
        f"Step {step} of {obs.get('max_steps', MAX_STEPS)}\n"
        f"Data context: {build_context(obs)}\n"
        f"Last result: {obs.get('last_action_result') or 'None'}\n\n"
        "Report ONLY the exact numeric data from the context. "
        "Include km² flood extents, rainfall totals, population numbers, accuracy metrics."
    )
    return call_llm(client, system, user)


def agent_domain_expert(client, obs, step, task_id):
    """Agent 2: Focus on district names and causal GIS factors."""
    system = (
        "You are a Senior GIS and Hydrology Expert for South Asian river systems. "
        "Your job: name specific districts affected by flooding, explain causal factors "
        "(DEM elevation, HydroSHEDS flow accumulation, CHIRPS rainfall anomalies, slope), "
        "identify high-risk geographic zones, and provide data-backed flood risk assessments. "
        "Always name at least 3 specific districts and cite 2+ causal factors."
    )
    user = (
        f"Task: {obs.get('task_description', task_id)}\n"
        f"Step {step} of {obs.get('max_steps', MAX_STEPS)}\n"
        f"Context: {build_context(obs)}\n"
        f"Last result: {obs.get('last_action_result') or 'None'}\n\n"
        "Provide expert GIS analysis: name specific districts, explain causal factors "
        "(CHIRPS rainfall, DEM elevation, HydroSHEDS flow accumulation, SRTM slope), "
        "identify flood-prone zones by name."
    )
    return call_llm(client, system, user)


def agent_critic(client, analyst_answer, expert_answer, task_id):
    """Agent 3: Find gaps before the answer is submitted to the environment."""
    system = (
        "You are a strict scientific peer reviewer for flood intelligence reports. "
        "Your job: identify MISSING information in two flood analysis answers. "
        "Look for: missing km² figures, missing district names, vague unsupported claims, "
        "missing causal factors, missing accuracy metrics. "
        "Be specific about each gap. Under 120 words."
    )
    user = (
        f"Task: {task_id}\n\n"
        f"DATA ANALYST ANSWER:\n{analyst_answer}\n\n"
        f"DOMAIN EXPERT ANSWER:\n{expert_answer}\n\n"
        "What is MISSING? List specific gaps: missing km² values, missing districts, "
        "vague claims needing numbers, absent causal factors."
    )
    return call_llm(client, system, user)


def agent_aggregator(client, analyst_answer, expert_answer, critic_feedback, obs, task_id):
    """Agent 4: Final answer combining all inputs — this gets sent to /step."""
    system = (
        "You are a Chief Flood Intelligence Officer writing the official final report. "
        "Combine the Data Analyst's numbers, the Domain Expert's GIS knowledge, "
        "and fix every gap the Critic identified. "
        "Your answer MUST include: "
        "(1) All exact km² figures for flood extents, "
        "(2) At least 3 specific district names, "
        "(3) Causal factors: CHIRPS rainfall, DEM elevation, HydroSHEDS flow accumulation, "
        "(4) No vague unsupported claims — every statement backed by data. "
        "Under 300 words. Prose only, no bullet points."
    )
    user = (
        f"Task: {obs.get('task_description', task_id)}\n"
        f"Context data: {build_context(obs)[:300]}\n\n"
        f"DATA ANALYST:\n{analyst_answer}\n\n"
        f"DOMAIN EXPERT:\n{expert_answer}\n\n"
        f"CRITIC IDENTIFIED GAPS:\n{critic_feedback}\n\n"
        "Write the final comprehensive answer. "
        "Fix ALL gaps. Include exact km² figures, district names, causal factors."
    )
    return call_llm(client, system, user)


def get_agent_response(client, obs, step, history, task_id):
    """
    Full 4-agent pipeline → returns best possible answer for /step.
    
    Flow:
      Agent 1 (numbers) ──┐
                           ├──▶ Agent 3 (critic) ──▶ Agent 4 (aggregator) ──▶ FINAL
      Agent 2 (GIS)    ──┘
    """
    print(f"[DEBUG] 4-agent pipeline | step={step} | model={'trained' if USE_TRAINED else 'base'}", flush=True)

    analyst = agent_data_analyst(client, obs, step, task_id)
    print(f"[DEBUG] Agent 1 (Data Analyst): {len(analyst)} chars", flush=True)

    expert = agent_domain_expert(client, obs, step, task_id)
    print(f"[DEBUG] Agent 2 (Domain Expert): {len(expert)} chars", flush=True)

    if analyst and expert:
        critic = agent_critic(client, analyst, expert, task_id)
        print(f"[DEBUG] Agent 3 (Critic): {len(critic)} chars", flush=True)
    else:
        critic = "No specific gaps identified."

    if analyst or expert:
        final = agent_aggregator(client, analyst, expert, critic, obs, task_id)
        print(f"[DEBUG] Agent 4 (Aggregator): {len(final)} chars", flush=True)
    else:
        final = ""

    if not final:
        fallback_list = FALLBACKS.get(task_id, FALLBACKS["flood_year_comparison"])
        return fallback_list[min(step - 1, len(fallback_list) - 1)]

    return final


# ── Run one task episode ────────────────────────────────────────────────────
async def run_task(client: OpenAI, task_id: str) -> float:
    history: List[str]   = []
    rewards: List[float] = []
    steps_taken = 0
    score   = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env_reset(task_id)
        if "error" in obs:
            print(f"[DEBUG] Reset error: {obs['error']}", flush=True)
            obs = {"task_description": task_id, "max_steps": MAX_STEPS,
                   "context": {}, "last_action_result": None, "done": False}

        max_s = obs.get("max_steps", MAX_STEPS)

        for step in range(1, max_s + 1):
            if obs.get("done", False):
                break

            action = get_agent_response(client, obs, step, history, task_id)
            result = env_step(action)

            if "error" in result:
                reward   = 0.0
                done     = False
                error    = result["error"][:80]
                obs_next = obs
            else:
                reward   = float(result.get("reward", 0) or 0)
                done     = bool(result.get("done", False))
                error    = result.get("last_action_error")
                obs_next = result.get("observation", obs)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action, reward=reward, done=done, error=error)
            history.append(f"Step {step}: reward={reward:+.2f}")
            obs = obs_next

            if done or step >= max_s:
                break

        raw_score = sum(rewards)
        score     = max(0.01, min(raw_score, 0.99))
        success   = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task error: {exc}", flush=True)
        score = 0.01
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Main ────────────────────────────────────────────────────────────────────
async def main() -> None:
    # OpenAI client points to HF router (used for base model fallback)
    client = OpenAI(base_url=HF_ROUTER_URL, api_key=HF_TOKEN)

    ALL_REGIONS = [
        "brahmaputra", "ganga", "mahanadi", "krishna", "godavari",
        "narmada", "tapti", "cauvery", "damodar", "sabarmati",
        "mahi", "baitarani", "subarnarekha", "indus", "luni",
    ]

    # If specific task set via env, run only that
    if os.getenv("MY_ENV_V4_TASK"):
        await run_task(client, TASK_NAME)
        return

    # Otherwise run all tasks across all regions
    for region in ALL_REGIONS:
        os.environ["CHRONOSTASIS_REGION"] = region
        print(f"\n{'='*50}\nREGION: {region.upper()}\n{'='*50}", flush=True)
        for task_id in ALL_TASKS:
            await run_task(client, task_id)
            print("", flush=True)


if __name__ == "__main__":
    asyncio.run(main())