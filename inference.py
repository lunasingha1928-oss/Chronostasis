"""
inference.py — Chronostasis Flood Intelligence Agent
OpenEnv Submission | Brahmaputra Valley SAR Flood Detection
============================================================

Environment variables required:
    API_BASE_URL        LLM endpoint  (default: HuggingFace router)
    MODEL_NAME          Model ID      (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN            HuggingFace / API key
    IMAGE_NAME          Docker image name for MyEnvV4

STDOUT format (strict):
    [START] task=<task> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

import ee
from openai import OpenAI

from my_env_v4 import MyEnvV4Action, MyEnvV4Env

# ──────────────────────────────────────────────────────────
# ENVIRONMENT CONFIGURATION
# ──────────────────────────────────────────────────────────
IMAGE_NAME   = os.getenv("IMAGE_NAME")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
GEE_PROJECT  = os.getenv("GEE_PROJECT",  "your-gee-project-id")  # ← set in env

TASK_NAME  = os.getenv("MY_ENV_V4_TASK",      "brahmaputra-flood-detection")
BENCHMARK  = os.getenv("MY_ENV_V4_BENCHMARK", "chronostasis")

MAX_STEPS               = 8
TEMPERATURE             = 0.3   # Lower = more deterministic for GIS analysis
MAX_TOKENS              = 512
SUCCESS_SCORE_THRESHOLD = 0.5
SAR_THRESHOLD           = -16   # dB — standardised across all years


# ──────────────────────────────────────────────────────────
# STDOUT LOGGING (strict OpenEnv format)
# ──────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    # Sanitise action string — no newlines allowed on a single [STEP] line
    action_clean = action.replace("\n", " ").replace("\r", "").strip()
    print(
        f"[STEP] step={step} action={action_clean!r} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ──────────────────────────────────────────────────────────
# GEE INITIALISATION (service account or token-based)
# ──────────────────────────────────────────────────────────
def init_gee() -> bool:
    """
    Initialise Google Earth Engine.
    In a Docker/server environment, use a service account JSON
    set via GEE_SERVICE_ACCOUNT_JSON env var.
    Falls back to default credentials for local dev.
    """
    sa_json = os.getenv("GEE_SERVICE_ACCOUNT_JSON")
    try:
        if sa_json:
            import json as _json
            credentials = ee.ServiceAccountCredentials(
                email=None, key_data=_json.loads(sa_json)
            )
            ee.Initialize(credentials, project=GEE_PROJECT)
        else:
            ee.Initialize(project=GEE_PROJECT)
        return True
    except Exception as exc:
        print(f"[DEBUG] GEE init failed: {exc}", flush=True)
        return False


# ──────────────────────────────────────────────────────────
# FLOOD DETECTION TOOLS (called by agent via tool dispatch)
# ──────────────────────────────────────────────────────────
def _get_aoi():
    return (
        ee.FeatureCollection("FAO/GAUL/2015/level1")
        .filter(ee.Filter.eq("ADM1_NAME", "Assam"))
        .geometry()
    )


def _get_sar_baseline(aoi):
    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate("2022-01-01", "2022-03-31")
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV")
        .mean()
        .clip(aoi)
    )


def tool_run_flood_detection(year: int) -> dict:
    """Run SAR flood detection for a given monsoon year (2022–2024)."""
    if year not in [2022, 2023, 2024]:
        return {"error": f"Year {year} not in dataset. Use 2022, 2023 or 2024."}

    aoi      = _get_aoi()
    baseline = _get_sar_baseline(aoi)
    pixel_area = ee.Image.pixelArea().divide(1e6)

    s1_flood = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(f"{year}-06-01", f"{year}-09-30")
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV")
        .min()   # .min() captures peak flood extent across full monsoon
        .clip(aoi)
    )
    flood_mask = s1_flood.lt(SAR_THRESHOLD).And(baseline.gt(SAR_THRESHOLD))

    area = (
        flood_mask.multiply(pixel_area)
        .reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi, scale=1000, maxPixels=1e10)
        .values()
        .get(0)
        .getInfo()
    )
    return {
        "year": year,
        "flood_area_km2": round(area, 1),
        "sar_threshold_db": SAR_THRESHOLD,
        "method": "Sentinel-1 VV SAR change detection (min-monsoon vs dry baseline)",
    }


def tool_get_accuracy_metrics() -> dict:
    """Compute accuracy of the multi-factor flood risk model vs SAR ground truth."""
    aoi        = _get_aoi()
    baseline   = _get_sar_baseline(aoi)
    pixel_area = ee.Image.pixelArea().divide(1e6)
    dem        = ee.Image("USGS/SRTMGL1_003").clip(aoi)
    slope      = ee.Terrain.slope(dem)
    flow_accum = ee.Image("WWF/HydroSHEDS/15ACC").clip(aoi)

    # July 2022 rainfall (for risk zone thresholds)
    july_rain_2022 = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate("2022-07-01", "2022-07-31")
        .filterBounds(aoi)
        .sum()
        .clip(aoi)
    )

    # Risk zones
    high_risk = (
        flow_accum.gt(5000).And(dem.lt(60)).And(slope.lt(3)).And(july_rain_2022.gt(300))
    )
    mod_risk = (
        flow_accum.gt(1000).And(dem.lt(100)).And(slope.lt(6))
        .And(july_rain_2022.gt(200)).And(high_risk.Not())
    )
    predicted_flood = high_risk.Or(mod_risk).rename("predicted")

    # SAR ground truth — full monsoon, NDWI-cleaned
    s1_flood_full = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate("2022-06-01", "2022-09-30")
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV").min().clip(aoi)
    )
    sar_raw = s1_flood_full.lt(SAR_THRESHOLD).And(baseline.gt(SAR_THRESHOLD))

    # Remove permanent water via NDWI
    landsat_dry = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(aoi)
        .filterDate("2022-01-01", "2022-03-31")
        .filter(ee.Filter.lt("CLOUD_COVER", 20))
        .median()
        .clip(aoi)
    )
    ndwi = landsat_dry.normalizedDifference(["SR_B3", "SR_B5"])
    permanent_water = ndwi.gt(0.3)
    sar_clean = sar_raw.And(permanent_water.Not()).rename("actual_clean")

    def area_km2(img):
        return (
            img.multiply(pixel_area)
            .reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi, scale=1000, maxPixels=1e10)
            .values().get(0).getInfo()
        )

    tp_a = area_km2(predicted_flood.eq(1).And(sar_clean.eq(1)))
    fp_a = area_km2(predicted_flood.eq(1).And(sar_clean.eq(0)))
    tn_a = area_km2(predicted_flood.eq(0).And(sar_clean.eq(0)))
    fn_a = area_km2(predicted_flood.eq(0).And(sar_clean.eq(1)))

    precision = tp_a / (tp_a + fp_a) if (tp_a + fp_a) > 0 else 0.0
    recall    = tp_a / (tp_a + fn_a) if (tp_a + fn_a) > 0 else 0.0
    accuracy  = (tp_a + tn_a) / (tp_a + fp_a + tn_a + fn_a)
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "accuracy_pct": round(accuracy * 100, 2),
        "precision_pct": round(precision * 100, 2),
        "recall_pct": round(recall * 100, 2),
        "f1_score": round(f1, 4),
        "tp_km2": round(tp_a, 1),
        "fp_km2": round(fp_a, 1),
        "tn_km2": round(tn_a, 1),
        "fn_km2": round(fn_a, 1),
        "ground_truth": "Sentinel-1 SAR VV (NDWI-corrected, permanent water removed)",
    }


def tool_get_chronic_inundation() -> dict:
    """Return areas flooded across 1, 2, or all 3 years (2022–2024)."""
    aoi        = _get_aoi()
    baseline   = _get_sar_baseline(aoi)
    pixel_area = ee.Image.pixelArea().divide(1e6)

    yearly_masks = []
    for yr in [2022, 2023, 2024]:
        s1 = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(aoi)
            .filterDate(f"{yr}-06-01", f"{yr}-09-30")
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .select("VV").min().clip(aoi)
        )
        yearly_masks.append(s1.lt(SAR_THRESHOLD).And(baseline.gt(SAR_THRESHOLD)))

    freq = yearly_masks[0].add(yearly_masks[1]).add(yearly_masks[2])

    result = {}
    for n in [1, 2, 3]:
        area = (
            freq.gte(n).multiply(pixel_area)
            .reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi, scale=1000, maxPixels=1e10)
            .values().get(0).getInfo()
        )
        result[f"flooded_gte_{n}yr_km2"] = round(area, 1)

    result["chronically_inundated_km2"] = result["flooded_gte_3yr_km2"]
    return result


# ──────────────────────────────────────────────────────────
# TOOL REGISTRY & DISPATCH
# ──────────────────────────────────────────────────────────
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "run_flood_detection",
            "description": (
                "Run SAR-based flood detection for Assam/Brahmaputra for a monsoon year. "
                "Returns total flood area in sq km."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {
                        "type": "integer",
                        "enum": [2022, 2023, 2024],
                        "description": "Monsoon year to analyse.",
                    }
                },
                "required": ["year"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_accuracy_metrics",
            "description": (
                "Compute precision, recall, F1, and confusion matrix (km²) of the "
                "multi-factor flood risk model against NDWI-corrected SAR ground truth."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_chronic_inundation",
            "description": (
                "Return areas flooded in 1+, 2+, or all 3 years (2022–2024). "
                "Identifies chronically inundated zones."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

TOOL_DISPATCH = {
    "run_flood_detection":  lambda args: tool_run_flood_detection(args["year"]),
    "get_accuracy_metrics": lambda args: tool_get_accuracy_metrics(),
    "get_chronic_inundation": lambda args: tool_get_chronic_inundation(),
}


def dispatch_tool(name: str, args: dict) -> str:
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = fn(args)
        return json.dumps(result)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ──────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are ChronostasisAgent — a GIS flood intelligence system for the Brahmaputra Valley.
    Your task is to analyse SAR satellite data and produce an accurate, data-backed flood report.

    Available tools:
    - run_flood_detection(year)  → flood area in km² for 2022, 2023, or 2024
    - get_accuracy_metrics()     → model precision, recall, F1, confusion matrix
    - get_chronic_inundation()   → areas flooded across 1, 2, or all 3 years

    Behaviour rules:
    1. ALWAYS call tools to get real numbers — never guess or hallucinate flood areas.
    2. Run flood detection for ALL three years before making comparisons.
    3. Always call get_accuracy_metrics() to validate your model's performance.
    4. Identify the most flood-prone year and chronically inundated areas.
    5. Keep responses factual, concise, and cite specific km² values.
    6. Final output must be a structured flood report with all stats included.
""").strip()


# ──────────────────────────────────────────────────────────
# AGENT LOOP (multi-turn with tool calling)
# ──────────────────────────────────────────────────────────
def build_user_prompt(step: int, obs: str, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(f"""
        Step: {step}
        Current observation: {obs}
        Last reward: {last_reward:.2f}
        History:
        {history_block}

        Analyse the Brahmaputra flood situation. Use your tools to gather data,
        then produce a comprehensive flood intelligence report.
    """).strip()


def run_agent_step(
    client: OpenAI,
    messages: List[dict],
    step: int,
    last_reward: float,
    history: List[str],
    obs: str,
) -> tuple[str, List[dict]]:
    """
    One agent step: may include multiple tool calls (internal) before
    producing a final text action for the environment.
    Returns (action_string, updated_messages).
    """
    messages.append({
        "role": "user",
        "content": build_user_prompt(step, obs, last_reward, history),
    })

    # Inner loop: resolve all tool calls before returning action
    for _ in range(6):  # max 6 tool call rounds per step
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
        except Exception as exc:
            print(f"[DEBUG] LLM call failed: {exc}", flush=True)
            return "flood analysis step failed", messages

        choice  = response.choices[0]
        message = choice.message
        messages.append(message.model_dump(exclude_unset=True))

        # No tool call → model produced final text
        if not message.tool_calls:
            return (message.content or "no output").strip(), messages

        # Process tool calls
        for tc in message.tool_calls:
            args        = json.loads(tc.function.arguments or "{}")
            tool_result = dispatch_tool(tc.function.name, args)
            print(f"[DEBUG] tool={tc.function.name} args={args} result={tool_result[:120]}", flush=True)
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      tool_result,
            })

    # Fallback if tool loop exhausted
    return "max tool calls reached", messages


# ──────────────────────────────────────────────────────────
# SCORE CALCULATION
# ──────────────────────────────────────────────────────────
def compute_score(action_text: str, tool_results: List[str]) -> float:
    """
    Score is based on:
      - Whether flood areas for all 3 years were retrieved (0.3)
      - Whether accuracy metrics were computed (0.3)
      - Whether chronic inundation was identified (0.2)
      - Whether the final action is a substantive report (0.2)
    Score is in [0, 1].
    """
    score = 0.0
    combined = " ".join(tool_results) + " " + action_text

    if all(str(yr) in combined for yr in [2022, 2023, 2024]):
        score += 0.3
    if any(k in combined for k in ["accuracy", "precision", "recall", "f1"]):
        score += 0.3
    if "chronic" in combined.lower() or "flooded_gte_3yr" in combined:
        score += 0.2
    if len(action_text) > 200:  # Substantive final report
        score += 0.2

    return min(score, 1.0)


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────
async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Init GEE
    gee_ok = init_gee()
    if not gee_ok:
        print("[DEBUG] GEE unavailable — tool calls will fail gracefully", flush=True)

    env = await MyEnvV4Env.from_docker_image(IMAGE_NAME)

    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    history:  List[str]  = []
    rewards:  List[float] = []
    tool_results_seen:    List[str] = []

    steps_taken = 0
    score       = 0.0
    success     = False
    last_action = ""

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result     = await env.reset()
        obs        = getattr(result.observation, "echoed_message", "Brahmaputra flood analysis task started.")
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action, messages = run_agent_step(
                client, messages, step, last_reward, history, obs
            )

            # Collect any tool results seen this step for scoring
            tool_results_seen.extend([
                m.get("content", "")
                for m in messages
                if isinstance(m, dict) and m.get("role") == "tool"
            ])

            result      = await env.step(MyEnvV4Action(message=action))
            obs         = getattr(result.observation, "echoed_message", obs)
            reward      = result.reward or 0.0
            done        = result.done
            error       = getattr(result, "last_action_error", None)

            rewards.append(reward)
            steps_taken  = step
            last_reward  = reward
            last_action  = action

            log_step(step=step, action=action, reward=reward, done=done, error=error)
            history.append(f"Step {step}: reward={reward:+.2f} | {action[:80]}")

            if done:
                break

        # Score based on what the agent actually did
        score   = compute_score(last_action, tool_results_seen)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Unhandled exception: {exc}", flush=True)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
