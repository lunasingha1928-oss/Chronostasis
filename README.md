---
title: Chronostasis
emoji: 🌊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - flood-detection
  - remote-sensing
pinned: false
---
# Chronostasis — Brahmaputra Valley Flood Intelligence

**OpenEnv Environment | Real-World GIS + Remote Sensing**

An AI agent environment for multi-year SAR flood analysis of the Brahmaputra river system in Assam, India. Agents learn to query satellite data, quantify flood extents, identify chronic inundation zones, and forecast future risk — all validated against a model with **92.39% accuracy**.

---

## What makes this real-world

This is not a simulated or toy environment. The data sources are live operational satellite datasets:

| Source            | What it provides                               |
| ----------------- | ---------------------------------------------- |
| Sentinel-1 SAR VV | Flood detection via backscatter change         |
| CHIRPS Daily      | Ground-calibrated rainfall (June–Sept)         |
| HydroSHEDS 15ACC  | Flow accumulation for drainage modelling       |
| SRTM DEM          | Elevation and slope for inundation risk        |
| Landsat 8 NDWI    | Permanent water mask (removes false positives) |
| FAO GAUL          | Administrative district boundaries             |

The task domain — flood early warning in South Asia — is one of the highest-impact applications of Earth Observation AI.

---

## Environment Description

### Action space

Free-form natural language string. The agent issues analysis, queries GIS tools, and provides data-backed conclusions.

```json
{
  "message": "Based on SAR analysis, 2022 had the largest flood extent at 4,812 km²..."
}
```

### Observation space

```json
{
  "task_id": "flood_year_comparison",
  "task_description": "...",
  "step": 2,
  "max_steps": 6,
  "available_data": ["Sentinel-1 SAR VV (2022–2024)", "CHIRPS rainfall", ...],
  "last_action_result": "Correct: 2022 identified as peak year (+0.40)",
  "last_action_error": null,
  "context": { "years_available": [2022, 2023, 2024], "sar_threshold_db": -16 },
  "echoed_message": "Based on SAR analysis..."
}
```

### Reward function

Rewards are **partial** (per step), **deterministic**, and **reproducible**. Each criterion is binary and keyed to specific numeric or factual content in the agent's response.

```
reward ∈ [0.0, 1.0] per step
No randomness — same action = same reward, always.
```

Penalises vague unsupported claims (−0.10) to discourage hallucination.

---

## Tasks

### Task 1 — Flood Year Comparison `easy` · max 6 steps

> Which monsoon year (2022–2024) had the largest SAR-detected flood extent?

| Criterion                                     | Weight |
| --------------------------------------------- | ------ |
| Correctly identifies 2022 as peak year        | 0.40   |
| Reports area figures for all 3 years (±15%)   | 0.35   |
| Cites ≥2 causal factors (rainfall, DEM, flow) | 0.25   |

Expected score for a capable agent: **0.85+**

---

### Task 2 — Chronic District Inundation Report `medium` · max 8 steps

> Which Assam districts have been chronically inundated (all 3 years)? What is the affected area and population?

| Criterion                                | Weight    |
| ---------------------------------------- | --------- |
| Each correct district named (×5 max)     | 0.10 each |
| Chronic flood area within 20% tolerance  | 0.25      |
| Affected population within 30% tolerance | 0.25      |

Expected score: **0.60–0.75**

---

### Task 3 — Next-Season Flood Risk Forecast `hard` · max 10 steps

> Using SAR history, rainfall trends, and the risk model (92.39% accuracy), forecast the highest-risk zones for the 2025 monsoon season.

| Criterion                          | Weight |
| ---------------------------------- | ------ |
| Cites accuracy metrics             | 0.15   |
| Cites risk zone statistics         | 0.20   |
| Names ≥2 specific geographic zones | 0.20   |
| Cites rainfall trend data          | 0.15   |
| Uses 2022 as reference benchmark   | 0.10   |
| Forward-looking 2025 forecast      | 0.05   |
| Vague/unsupported claims           | −0.10  |

Expected score: **0.65–0.80** for a well-grounded agent

---

## Setup & Running

### Local

```bash
# 1. Install
pip install -r requirements.txt

# 2. Authenticate GEE
earthengine authenticate

# 3. Set env vars
export GEE_PROJECT=brahmaputravalley
export OPENAI_API_KEY=sk-...
export HF_TOKEN=hf_...

# 4. Start the environment server
python server.py

# 5. Run the baseline agent
python inference.py
```

### Docker

```bash
docker build -t chronostasis .

docker run -p 7860:7860 \
  -e GEE_PROJECT=your-gee-project-id \
  -e GEE_SERVICE_ACCOUNT_JSON='{"type":"service_account",...}' \
  -e HF_TOKEN=hf_... \
  chronostasis
```

### Validate before submitting

```bash
./validate-submission.sh https://your-space.hf.space .
```

---

## Baseline Results

Baseline agent: `inference.py` with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Router

| Task                       | Score |
| -------------------------- | ----- |
| flood_year_comparison      | 0.850 |
| district_inundation_report | 0.620 |
| flood_risk_forecast        | 0.710 |

---

## HuggingFace Space deployment

1. Create a new HF Space tagged with `openenv`
2. Set Secrets: `GEE_PROJECT`, `GEE_SERVICE_ACCOUNT_JSON`, `HF_TOKEN`
3. Push this repo — the Dockerfile starts the server on port 7860
4. Verify: `curl -X POST https://your-space.hf.space/reset` returns HTTP 200

---

## GEE Service Account Setup

For headless Docker/HF Space deployment:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a service account with Earth Engine API access
3. Download the JSON key
4. Set it as `GEE_SERVICE_ACCOUNT_JSON` in your HF Space secrets (paste the entire JSON)
5. Register the service account at [earthengine.google.com/signup](https://signup.earthengine.google.com/)
