"""
tasks.py — Three Graded OpenEnv Tasks for Chronostasis
=======================================================

Task 1 (EASY):   flood_year_comparison
  "Which monsoon year (2022–2024) had the largest SAR-detected flood extent?"
  Agent must correctly identify the peak flood year and provide area figures.
  Score: 0.0 → 1.0  |  fully deterministic

Task 2 (MEDIUM): district_inundation_report
  "Identify which Assam districts have been chronically inundated (flooded all 3 years)
   and estimate the affected population."
  Partial credit for each correct district identified.
  Score: 0.0 → 1.0  |  partial progress rewards

Task 3 (HARD):   flood_risk_forecast
  "Given monsoon rainfall trends and SAR-detected flood history (2022–2024),
   forecast which zones are at HIGHEST risk in the next monsoon season.
   Your answer will be evaluated against the multi-factor risk model output."
  Score: 0.0 → 1.0  |  rubric-based, rewards methodology + specificity
"""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────
# BASE CLASS
# ──────────────────────────────────────────────

class BaseTask(ABC):
    task_id:       str
    name:          str
    description:   str
    difficulty:    str   # easy | medium | hard
    max_steps:     int
    available_data: List[str]

    def __init__(self, gee_available: bool = False):
        self.gee_available = gee_available
        self._step_count   = 0

    @abstractmethod
    def step(self, action: str, step_num: int) -> Dict[str, Any]:
        """Grade the agent's action. Returns {reward, done, result, error}."""

    def get_context(self) -> Dict[str, Any]:
        """Task-specific context data included in every observation."""
        return {}


# ──────────────────────────────────────────────
# GROUND TRUTH DATA
# (Values match the GEE analysis. In production these come live from GEE;
#  here we embed them so the server works without GEE connectivity.)
# ──────────────────────────────────────────────

SAR_FLOOD_AREAS = {
    2022: 4812.3,   # sq km — worst year
    2023: 3601.7,
    2024: 4101.2,
}
CHRONIC_DISTRICTS = [
    "Morigaon", "Dhubri", "Barpeta", "Goalpara", "Kamrup",
]
CHRONIC_AREA_KM2  = 1247.6   # flooded all 3 years
CHRONIC_POPULATION = 2_400_000  # approximate affected population

RISK_ZONE_STATS = {
    "high_risk_km2":     3218.4,
    "moderate_risk_km2": 5901.2,
    "low_risk_km2":      8240.1,
    "peak_year":         2022,
    "sar_accuracy_pct":  92.39,
}

HIGH_RISK_ZONES = [
    "lower Brahmaputra floodplain",
    "Dhubri district riverbank",
    "Barpeta wetland belt",
    "Morigaon char lands",
]


# ──────────────────────────────────────────────
# TASK 1 — EASY
# ──────────────────────────────────────────────

class FloodYearComparisonTask(BaseTask):
    task_id    = "flood_year_comparison"
    name       = "Flood Year Comparison"
    difficulty = "easy"
    max_steps  = 6
    description = (
        "Using Sentinel-1 SAR data for Assam (2022–2024 monsoon seasons), "
        "determine which year had the LARGEST flood extent and report the area "
        "in square kilometres for all three years. Explain what drove the difference."
    )
    available_data = [
        "Sentinel-1 SAR VV (2022–2024 June–Sept)",
        "CHIRPS rainfall (2022–2024)",
        "HydroSHEDS flow accumulation",
        "SRTM DEM",
    ]

    # Scoring thresholds
    _CORRECT_YEAR = 2022
    _AREA_TOLERANCE = 0.15   # 15% tolerance on area figures

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rewarded_year    = False
        self._rewarded_areas   = False
        self._rewarded_reason  = False

    def step(self, action: str, step_num: int) -> Dict[str, Any]:
        txt = action.lower()
        reward = 0.0
        done   = False
        result_notes = []

        # ── Criterion 1: Correctly identifies 2022 as peak year (0.40) ──
        if not self._rewarded_year:
            if "2022" in txt and any(w in txt for w in [
                "highest", "largest", "most", "greatest", "worst",
                "maximum", "peak", "biggest"
            ]):
                reward += 0.40
                self._rewarded_year = True
                result_notes.append("Correct: 2022 identified as peak flood year (+0.40)")

        # ── Criterion 2: Provides area figures for all 3 years (0.35) ──
        if not self._rewarded_areas:
            years_mentioned = sum(1 for yr in [2022, 2023, 2024] if str(yr) in action)
            # Check any km² figure near known values
            nums = re.findall(r"[\d,]+\.?\d*", action)
            nums_clean = [float(n.replace(",", "")) for n in nums]
            close_count = sum(
                1 for yr_area in SAR_FLOOD_AREAS.values()
                for n in nums_clean
                if abs(n - yr_area) / yr_area < self._AREA_TOLERANCE
            )
            if years_mentioned >= 3 and close_count >= 2:
                reward += 0.35
                self._rewarded_areas = True
                result_notes.append("Areas reported for all 3 years with acceptable accuracy (+0.35)")
            elif years_mentioned >= 2 and close_count >= 1:
                reward += 0.15
                result_notes.append("Partial: areas for some years (+0.15)")

        # ── Criterion 3: Explains a causal factor (0.25) ──
        if not self._rewarded_reason:
            causal_keywords = [
                "rainfall", "chirps", "precipitation", "monsoon", "flow", "accumulation",
                "dem", "elevation", "slope", "drainage", "basin", "discharge", "upstream",
            ]
            if sum(1 for kw in causal_keywords if kw in txt) >= 2:
                reward += 0.25
                self._rewarded_reason = True
                result_notes.append("Causal explanation provided (+0.25)")

        # Episode ends when full score reached or max steps hit
        total_so_far = self._rewarded_year + self._rewarded_areas + self._rewarded_reason
        done = total_so_far == 3 or step_num >= self.max_steps

        return {
            "reward": min(reward, 1.0),
            "done":   done,
            "result": " | ".join(result_notes) if result_notes else "No scoring criteria met this step.",
            "error":  None,
        }

    def get_context(self) -> Dict[str, Any]:
        return {
            "years_available": [2022, 2023, 2024],
            "sar_threshold_db": -16,
            "region": "Assam, India",
            "hint": "Use run_flood_detection() tool for each year then compare.",
        }


# ──────────────────────────────────────────────
# TASK 2 — MEDIUM
# ──────────────────────────────────────────────

class DistrictInundationTask(BaseTask):
    task_id    = "district_inundation_report"
    name       = "Chronic District Inundation Report"
    difficulty = "medium"
    max_steps  = 8
    description = (
        "Using flood frequency analysis (2022–2024), identify which Assam districts "
        "have been CHRONICALLY INUNDATED (flooded in all three monsoon seasons). "
        "Report the total chronically flooded area (km²) and estimate the affected population."
    )
    available_data = [
        "Sentinel-1 SAR flood extents 2022–2024",
        "Assam district boundaries (FAO GAUL)",
        "Flood frequency raster (0–3 years)",
        "WorldPop population grid",
        "NDWI permanent water mask",
    ]

    _DISTRICT_SCORE = 0.10   # per correct district (max 5 × 0.10 = 0.50)
    _AREA_SCORE     = 0.25
    _POP_SCORE      = 0.25

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._found_districts: set = set()
        self._rewarded_area   = False
        self._rewarded_pop    = False

    def step(self, action: str, step_num: int) -> Dict[str, Any]:
        txt    = action.lower()
        reward = 0.0
        result_notes = []

        # ── Districts (up to 0.50) ──
        for district in CHRONIC_DISTRICTS:
            d_key = district.lower()
            if d_key not in self._found_districts and d_key in txt:
                self._found_districts.add(d_key)
                reward += self._DISTRICT_SCORE
                result_notes.append(f"District found: {district} (+{self._DISTRICT_SCORE:.2f})")

        # ── Chronic area ±20% tolerance (0.25) ──
        if not self._rewarded_area:
            nums = [float(n.replace(",", "")) for n in re.findall(r"[\d,]+\.?\d*", action)]
            if any(abs(n - CHRONIC_AREA_KM2) / CHRONIC_AREA_KM2 < 0.20 for n in nums):
                reward += self._AREA_SCORE
                self._rewarded_area = True
                result_notes.append(f"Chronic area reported correctly (~{CHRONIC_AREA_KM2} km²) (+{self._AREA_SCORE})")

        # ── Population estimate ±30% tolerance (0.25) ──
        if not self._rewarded_pop:
            big_nums = [float(n.replace(",", "")) for n in re.findall(r"[\d,]+", action) if len(n.replace(",", "")) >= 6]
            if any(abs(n - CHRONIC_POPULATION) / CHRONIC_POPULATION < 0.30 for n in big_nums):
                reward += self._POP_SCORE
                self._rewarded_pop = True
                result_notes.append(f"Population estimate in range (~{CHRONIC_POPULATION:,}) (+{self._POP_SCORE})")

        n_districts = len(self._found_districts)
        done = (
            (n_districts == len(CHRONIC_DISTRICTS) and self._rewarded_area and self._rewarded_pop)
            or step_num >= self.max_steps
        )

        return {
            "reward": min(reward, 1.0),
            "done":   done,
            "result": (
                " | ".join(result_notes) if result_notes
                else f"No new criteria met. Districts found so far: {list(self._found_districts)}"
            ),
            "error":  None,
        }

    def get_context(self) -> Dict[str, Any]:
        return {
            "total_assam_districts": 35,
            "analysis_years": [2022, 2023, 2024],
            "chronic_definition": "Flooded in ALL 3 monsoon seasons",
            "hint": "Use get_chronic_inundation() then cross-reference with district boundaries.",
        }


# ──────────────────────────────────────────────
# TASK 3 — HARD
# ──────────────────────────────────────────────

class FloodRiskForecastTask(BaseTask):
    task_id    = "flood_risk_forecast"
    name       = "Next-Season Flood Risk Forecast"
    difficulty = "hard"
    max_steps  = 10
    description = (
        "Based on SAR flood history (2022–2024), rainfall trends (CHIRPS), "
        "DEM/slope/flow accumulation data, and the multi-factor risk model, "
        "forecast which zones in Assam are at HIGHEST risk in the 2025 monsoon season. "
        "Justify your forecast with specific data: risk zone areas, accuracy metrics, "
        "and at least two specific geographic zones. "
        "Your answer is evaluated against the validated risk model (92.39% accuracy)."
    )
    available_data = [
        "Sentinel-1 SAR 2022–2024 flood extents",
        "CHIRPS rainfall 2022–2024 monsoon totals",
        "Flood frequency map (0–3 years)",
        "Multi-factor risk zones (high/moderate/low)",
        "Accuracy assessment (precision, recall, F1)",
        "SRTM DEM + slope + HydroSHEDS flow accumulation",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._used_accuracy  = False
        self._used_risk_zones = False
        self._named_zones    = 0
        self._cited_rainfall = False
        self._cited_year     = False
        self._step_rewards: List[float] = []

    def step(self, action: str, step_num: int) -> Dict[str, Any]:
        txt    = action.lower()
        reward = 0.0
        result_notes = []

        # ── Uses accuracy metrics (0.15) ──
        if not self._used_accuracy:
            if any(kw in txt for kw in ["92", "precision", "recall", "f1", "accuracy"]):
                reward += 0.15
                self._used_accuracy = True
                result_notes.append("Cites model accuracy (+0.15)")

        # ── Cites risk zone statistics (0.20) ──
        if not self._used_risk_zones:
            zone_hit = sum(1 for kw in ["high risk", "high-risk", "moderate risk", "low risk",
                                         "3218", "5901", "8240"] if kw in txt)
            if zone_hit >= 2:
                reward += 0.20
                self._used_risk_zones = True
                result_notes.append("Cites risk zone area statistics (+0.20)")
            elif zone_hit == 1:
                reward += 0.08
                result_notes.append("Partial: one risk zone cited (+0.08)")

        # ── Names specific high-risk geographic zones (0.10 each, max 2 = 0.20) ──
        if self._named_zones < 2:
            for zone in HIGH_RISK_ZONES:
                zone_key = zone.lower()
                if zone_key in txt and self._named_zones < 2:
                    self._named_zones += 1
                    reward += 0.10
                    result_notes.append(f"Specific zone named: '{zone}' (+0.10)")

        # ── Cites rainfall trend (0.15) ──
        if not self._cited_rainfall:
            if any(kw in txt for kw in ["rainfall", "chirps", "precipitation", "monsoon trend",
                                         "mm", "1500", "1200"]):
                reward += 0.15
                self._cited_rainfall = True
                result_notes.append("Cites rainfall data (+0.15)")

        # ── Cites specific year as reference (0.10) ──
        if not self._cited_year:
            if "2022" in txt and any(w in txt for w in ["worst", "baseline", "reference",
                                                          "peak", "highest", "most severe"]):
                reward += 0.10
                self._cited_year = True
                result_notes.append("Uses 2022 as reference benchmark (+0.10)")

        # ── Bonus: forecast is forward-looking to 2025 (0.05) ──
        if "2025" in txt and any(w in txt for w in ["forecast", "predict", "expect", "anticipate",
                                                      "project", "risk", "likely"]):
            reward += 0.05
            result_notes.append("Forward-looking 2025 forecast (+0.05)")

        # ── Penalty: vague or unsupported claims (−0.10) ──
        vague_only = (
            len(result_notes) == 0
            and len(action) < 120
            and not any(c.isdigit() for c in action)
        )
        if vague_only:
            reward -= 0.10
            result_notes.append("Response too vague / no data cited (−0.10)")

        reward = max(reward, 0.0)
        self._step_rewards.append(reward)

        # Done when agent has cited all major components or exhausted steps
        criteria_met = (
            self._used_accuracy
            + self._used_risk_zones
            + (self._named_zones >= 2)
            + self._cited_rainfall
            + self._cited_year
        )
        done = criteria_met >= 4 or step_num >= self.max_steps

        return {
            "reward": min(reward, 1.0),
            "done":   done,
            "result": (
                " | ".join(result_notes) if result_notes
                else "No scoring criteria met this step. Include specific data and zone names."
            ),
            "error":  None,
        }

    def get_context(self) -> Dict[str, Any]:
        return {
            "model_accuracy_pct": RISK_ZONE_STATS["sar_accuracy_pct"],
            "risk_zones_km2": {
                "high":     RISK_ZONE_STATS["high_risk_km2"],
                "moderate": RISK_ZONE_STATS["moderate_risk_km2"],
                "low":      RISK_ZONE_STATS["low_risk_km2"],
            },
            "peak_year":       RISK_ZONE_STATS["peak_year"],
            "chronic_area_km2": CHRONIC_AREA_KM2,
            "hint": (
                "Call get_accuracy_metrics(), run_flood_detection() for all years, "
                "get_chronic_inundation(), then synthesise a justified forecast."
            ),
        }


# ──────────────────────────────────────────────
# TASK REGISTRY
# ──────────────────────────────────────────────

TASK_REGISTRY: Dict[str, type] = {
    "flood_year_comparison":     FloodYearComparisonTask,
    "district_inundation_report": DistrictInundationTask,
    "flood_risk_forecast":       FloodRiskForecastTask,
}
