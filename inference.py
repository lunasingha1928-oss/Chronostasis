"""
multi_agent_inference.py — Chronostasis Multi-Agent Flood Intelligence System
==============================================================================

HOW THIS WORKS (beginner explanation):
---------------------------------------
Instead of ONE AI answering each flood question, we use FOUR specialized agents
that work together like a team of experts:

  Agent 1 - DATA ANALYST   → focuses on exact numbers (km², rainfall stats)
  Agent 2 - DOMAIN EXPERT  → focuses on district names, causes, GIS knowledge
  Agent 3 - CRITIC         → reads both answers and finds missing pieces
  Agent 4 - AGGREGATOR     → combines everything into one perfect final answer

Only the AGGREGATOR's answer gets sent to the environment (/step).
This maximizes reward because all criteria get covered.

STDOUT format (same as original inference.py — don't change):
[START] task=<task> env=<benchmark> model=<model>
[STEP] step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import asyncio
import json
import os
import urllib.request
import urllib.error
from typing import List, Optional
from openai import OpenAI


# ──────────────────────────────────────────────────────────
# CONFIGURATION (same as original inference.py)
# ──────────────────────────────────────────────────────────

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "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")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://LunaAmagi-chronostasis.hf.space")
BENCHMARK    = os.getenv("CHRONOSTASIS_BENCH", "chronostasis")
REGION_ID    = os.getenv("CHRONOSTASIS_REGION", "brahmaputra")

ALL_TASKS = [
    "flood_year_comparison",
    "district_inundation_report",
    "flood_risk_forecast",
]

TASK_NAME             = os.getenv("MY_ENV_V4_TASK", ALL_TASKS[0])
MAX_STEPS             = 8
TEMPERATURE           = 0.3
MAX_TOKENS            = 400
SUCCESS_SCORE_THRESHOLD = 0.5


# ──────────────────────────────────────────────────────────
# STDOUT LOGGING (identical to original — required format)
# ──────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    action_clean = action.replace("\n", " ").replace("\r", "").strip()[:200]
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action_clean!r} "
          f"reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"score={score:.3f} rewards={rewards_str}", flush=True)


# ──────────────────────────────────────────────────────────
# HTTP CLIENT (identical to original)
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
# FALLBACK RESPONSES
# (intentionally vague — used only when LLM is unavailable)
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
# THE FOUR AGENTS
# Each agent is just a different instruction (system prompt)
# sent to the same AI model. Think of it like asking a friend
# to "wear a different hat" each time.
# ──────────────────────────────────────────────────────────

def call_llm(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    """
    Helper: sends one message to the AI and gets a response back.
    Used by all 4 agents — only the system_prompt changes per agent.
    """
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return ""


def agent_data_analyst(client: OpenAI, obs: dict, step: int, task_id: str) -> str:
    """
    AGENT 1 — DATA ANALYST
    Role: Focus ONLY on numbers. km² figures, rainfall mm, percentages,
    population counts. Cite exact values from the context provided.
    Why: The reward rubric requires exact numeric figures — this agent
    makes sure none are missing.
    """
    system = (
        "You are a GIS Data Analyst specializing in SAR satellite flood data. "
        "Your ONLY job is to extract and report EXACT numbers: "
        "flood extent in km², CHIRPS rainfall in mm, population counts, "
        "percentages, and accuracy metrics. "
        "Never be vague. Always give specific figures."
    )
    ctx = obs.get("context", {})
    user = (
        f"Task: {obs.get('task_description', task_id)}\n"
        f"Step {step} of {obs.get('max_steps', MAX_STEPS)}\n"
        f"Data context: {json.dumps(ctx)[:400]}\n"
        f"Last result: {obs.get('last_action_result') or 'None'}\n\n"
        f"Report ONLY the exact numeric data relevant to this task. "
        f"Include km² figures for flood extents, rainfall totals, population numbers."
    )
    return call_llm(client, system, user)


def agent_domain_expert(client: OpenAI, obs: dict, step: int, task_id: str) -> str:
    """
    AGENT 2 — DOMAIN EXPERT
    Role: Focus on GIS knowledge — district names, causal factors
    (why floods happen), geographic zones, risk explanations.
    Why: The reward rubric requires district names and causal terminology.
    This agent covers what Agent 1 misses.
    """
    system = (
        "You are a Senior GIS and Hydrology Expert for South Asian river systems. "
        "Your job is to provide expert analysis: name specific districts, "
        "explain causal factors (DEM elevation, HydroSHEDS flow accumulation, "
        "CHIRPS rainfall, slope), identify geographic zones, and give "
        "data-backed flood risk assessments. "
        "Always name at least 3 specific districts and 2 causal factors."
    )
    ctx = obs.get("context", {})
    user = (
        f"Task: {obs.get('task_description', task_id)}\n"
        f"Step {step} of {obs.get('max_steps', MAX_STEPS)}\n"
        f"Context: {json.dumps(ctx)[:400]}\n"
        f"Last result: {obs.get('last_action_result') or 'None'}\n\n"
        f"Provide your expert GIS analysis. Name specific districts "
        f"(Morigaon, Dhubri, Barpeta, Goalpara, Kamrup) and causal factors."
    )
    return call_llm(client, system, user)


def agent_critic(client: OpenAI, analyst_answer: str, expert_answer: str,
                 task_id: str) -> str:
    """
    AGENT 3 — CRITIC
    Role: Read what Agent 1 and Agent 2 said, and find what's MISSING.
    Point out: missing numbers, vague claims, missing district names,
    missing causal factors.
    Why: This is the 'debate' step. The critic forces the final answer
    to be complete. Vague claims cost -0.10 in the reward — the critic
    catches those before submission.
    """
    system = (
        "You are a strict scientific reviewer. Your job is to identify GAPS "
        "in two flood analysis answers. Look for: missing km² figures, "
        "missing district names, vague unsupported claims, missing causal factors, "
        "missing accuracy metrics. "
        "Be specific about what each answer is missing. Keep it under 150 words."
    )
    user = (
        f"Task: {task_id}\n\n"
        f"--- DATA ANALYST ANSWER ---\n{analyst_answer}\n\n"
        f"--- DOMAIN EXPERT ANSWER ---\n{expert_answer}\n\n"
        f"What is MISSING from these answers? What vague claims need specific data? "
        f"What district names or km² figures are absent? List the gaps clearly."
    )
    return call_llm(client, system, user)


def agent_aggregator(client: OpenAI, analyst_answer: str, expert_answer: str,
                     critic_feedback: str, obs: dict, task_id: str) -> str:
    """
    AGENT 4 — AGGREGATOR
    Role: Read all three inputs and write ONE perfect final answer.
    This is the answer that actually gets sent to /step in the environment.
    Why: Combines the best of both Agent 1 and Agent 2, while fixing
    everything the Critic flagged. Maximizes reward score.
    """
    system = (
        "You are a Chief Flood Intelligence Officer writing the final official report. "
        "You have inputs from a Data Analyst (numbers), a Domain Expert (districts/causes), "
        "and a Critic (what's missing). "
        "Write ONE comprehensive answer that: "
        "1) Includes ALL exact km² figures "
        "2) Names ALL relevant districts "
        "3) Cites causal factors (rainfall, DEM, flow accumulation) "
        "4) Fixes every gap the Critic identified "
        "5) Makes NO vague unsupported claims "
        "Keep it under 300 words but make every sentence count."
    )
    ctx = obs.get("context", {})
    user = (
        f"Task: {obs.get('task_description', task_id)}\n"
        f"Context data: {json.dumps(ctx)[:300]}\n\n"
        f"--- DATA ANALYST SAID ---\n{analyst_answer}\n\n"
        f"--- DOMAIN EXPERT SAID ---\n{expert_answer}\n\n"
        f"--- CRITIC IDENTIFIED THESE GAPS ---\n{critic_feedback}\n\n"
        f"Write the final complete answer addressing ALL gaps. "
        f"Include specific km² figures, district names, and causal factors."
    )
    return call_llm(client, system, user)


# ──────────────────────────────────────────────────────────
# MULTI-AGENT ORCHESTRATOR
# This is the main function that coordinates all 4 agents
# and returns the final answer to send to the environment.
# ──────────────────────────────────────────────────────────

def get_multi_agent_response(client: OpenAI, obs: dict, step: int,
                              history: List[str], task_id: str) -> str:
    """
    Runs the full 4-agent pipeline and returns the best possible answer.

    Flow:
      Agent 1 (Data Analyst)  ──┐
                                 ├──► Agent 3 (Critic) ──► Agent 4 (Aggregator) ──► FINAL ANSWER
      Agent 2 (Domain Expert) ──┘

    Falls back to FALLBACKS dict if all LLM calls fail.
    """
    print(f"[DEBUG] Running multi-agent pipeline for step {step}...", flush=True)

    # ── Step 1: Data Analyst gives numbers ──────────────────
    analyst_answer = agent_data_analyst(client, obs, step, task_id)
    print(f"[DEBUG] Agent 1 (Data Analyst) done.", flush=True)

    # ── Step 2: Domain Expert gives GIS knowledge ───────────
    expert_answer = agent_domain_expert(client, obs, step, task_id)
    print(f"[DEBUG] Agent 2 (Domain Expert) done.", flush=True)

    # ── Step 3: Critic finds gaps ────────────────────────────
    # Only run critic if both agents produced something useful
    if analyst_answer and expert_answer:
        critic_feedback = agent_critic(client, analyst_answer, expert_answer, task_id)
        print(f"[DEBUG] Agent 3 (Critic) done.", flush=True)
    else:
        critic_feedback = "No specific gaps identified."

    # ── Step 4: Aggregator writes the final answer ───────────
    final_answer = ""
    if analyst_answer or expert_answer:
        final_answer = agent_aggregator(
            client, analyst_answer, expert_answer, critic_feedback, obs, task_id
        )
        print(f"[DEBUG] Agent 4 (Aggregator) done.", flush=True)

    # ── Fallback: if all agents failed, use safe fallback ────
    if not final_answer:
        print(f"[DEBUG] All agents failed, using fallback.", flush=True)
        fallback_steps = FALLBACKS.get(task_id, FALLBACKS["flood_year_comparison"])
        idx = min(step - 1, len(fallback_steps) - 1)
        return fallback_steps[idx]

    return final_answer


# ──────────────────────────────────────────────────────────
# RUN ONE TASK EPISODE (same structure as original)
# ──────────────────────────────────────────────────────────

async def run_task(client: OpenAI, task_id: str) -> float:
    """
    Runs one complete task episode using the multi-agent system.
    Structure is identical to original inference.py — only the
    'get agent response' call is replaced with multi-agent pipeline.
    """
    history: List[str]  = []
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

            # ← THIS IS THE KEY CHANGE: multi-agent instead of single agent
            action = get_multi_agent_response(client, obs, step, history, task_id)

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
# MAIN — runs all 3 tasks (same as original)
# ──────────────────────────────────────────────────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    tasks_to_run = [TASK_NAME] if os.getenv("MY_ENV_V4_TASK") else ALL_TASKS

    for task_id in tasks_to_run:
        await run_task(client, task_id)
        print("", flush=True)  # blank line between tasks


if __name__ == "__main__":
    asyncio.run(main())
