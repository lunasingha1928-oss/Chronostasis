"""
inference.py — Chronostasis Flood Intelligence Agent
OpenEnv Submission | India River Basin SAR Flood Detection
===========================================================

STDOUT format (strict):
[START] task=<task> env=<benchmark> model=<model>
[STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import asyncio
import json
import os
import urllib.request
import urllib.error
from typing import List, Optional

# ── Load .env for local dev (HF Space uses its own secrets) ──────────────
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

# ──────────────────────────────────────────────────────────
# ENVIRONMENT CONFIGURATION — all from .env / HF secrets
# ──────────────────────────────────────────────────────────

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "LunaAmagi/chronostasis-3b-grpo-medium")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://LunaAmagi-chronostasis.hf.space")
BENCHMARK    = os.getenv("CHRONOSTASIS_BENCH",  "chronostasis")
REGION_ID    = os.getenv("CHRONOSTASIS_REGION", "brahmaputra")

ALL_TASKS = [
    "flood_year_comparison",
    "district_inundation_report",
    "flood_risk_forecast",
]
TASK_NAME              = os.getenv("MY_ENV_V4_TASK", ALL_TASKS[0])
MAX_STEPS              = 8
TEMPERATURE            = 0.3
MAX_TOKENS             = 400
SUCCESS_SCORE_THRESHOLD = 0.5

# ──────────────────────────────────────────────────────────
# STDOUT LOGGING
# ──────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    action_clean = action.replace("\n", " ").replace("\r", "").strip()[:200]
    error_val    = error if error else "null"
    print(f"[STEP] step={step} action={action_clean!r} "
          f"reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"score={score:.3f} rewards={rewards_str}", flush=True)

# ──────────────────────────────────────────────────────────
# HTTP CLIENT
# ──────────────────────────────────────────────────────────

def env_request(path: str, method: str = "GET", body: dict = None) -> dict:
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

def env_reset(task_id: str) -> dict:
    return env_request("/reset", "POST", {"task_id": task_id, "region_id": REGION_ID})

def env_step(message: str) -> dict:
    return env_request("/step", "POST", {"message": message})

# ──────────────────────────────────────────────────────────
# FALLBACK RESPONSES (used when LLM unavailable)
# ──────────────────────────────────────────────────────────

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

# ──────────────────────────────────────────────────────────
# AGENT
# ──────────────────────────────────────────────────────────

def get_agent_response(client: OpenAI, obs: dict, step: int,
                       history: List[str], task_id: str) -> str:
    try:
        ctx    = obs.get("context", {})
        prompt = (
            f"Task: {obs.get('task_description', task_id)}\n"
            f"Step {step} of {obs.get('max_steps', MAX_STEPS)}\n"
            f"Context: {json.dumps(ctx)[:400]}\n"
            f"Last result: {obs.get('last_action_result') or 'None'}\n"
            f"Provide a specific data-backed response with exact km² figures and district names."
        )
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": (
                    "You are a precise GIS flood analyst. "
                    "Always cite exact km² figures, district names, and percentages."
                )},
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        msg = (completion.choices[0].message.content or "").strip()
        if msg:
            return msg
    except Exception as exc:
        print(f"[DEBUG] LLM failed: {exc}", flush=True)

    # Vague fallback — deliberately weak for baseline comparison
    fallback_steps = FALLBACKS.get(task_id, FALLBACKS["flood_year_comparison"])
    idx = min(step - 1, len(fallback_steps) - 1)
    return fallback_steps[idx]

# ──────────────────────────────────────────────────────────
# RUN ONE TASK EPISODE
# ──────────────────────────────────────────────────────────

async def run_task(client: OpenAI, task_id: str) -> float:
    history:     List[str]  = []
    rewards:     List[float] = []
    steps_taken: int        = 0
    score:       float      = 0.0
    success:     bool       = False

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
                print(f"[DEBUG] Step error: {result['error']}", flush=True)
                reward  = 0.0
                done    = False
                error   = result["error"][:80]
                obs_next = obs
            else:
                reward   = float(result.get("reward", 0) or 0)
                done     = bool(result.get("done", False))
                error    = result.get("last_action_error")
                obs_next = result.get("observation", obs)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {reward:+.2f}")
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

# ──────────────────────────────────────────────────────────
# MAIN — runs all 3 tasks
# ──────────────────────────────────────────────────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    tasks_to_run = [TASK_NAME] if os.getenv("MY_ENV_V4_TASK") else ALL_TASKS

    for task_id in tasks_to_run:
        await run_task(client, task_id)
        print("", flush=True)

if __name__ == "__main__":
    asyncio.run(main())