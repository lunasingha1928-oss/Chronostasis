"""
tasks.py — Chronostasis OpenEnv Task Definitions
=================================================
Multi-region flood intelligence environment for Indian river basins.
Expanded from 5 to 15 basins covering ~85% of India's flood-prone population.

Regions covered:
  Original 5:  Brahmaputra, Ganga, Mahanadi, Krishna, Godavari
  New 10:      Indus, Narmada, Tapti, Cauvery, Damodar,
               Sabarmati, Mahi, Baitarani, Subarnarekha, Luni
"""

import re
from typing import Any, Dict, List, Optional

# ──────────────────────────────────────────────
# REGION DATA — 15 Indian river basins
# Each region has lat/lon for map display,
# seasonal risk multipliers, and full flood data.
# ──────────────────────────────────────────────

REGIONS: Dict[str, Dict[str, Any]] = {

    # ── ORIGINAL 5 ───────────────────────────────────────────

    "brahmaputra": {
        "name": "Brahmaputra Valley",
        "state": "Assam",
        "river": "Brahmaputra",
        "lat": 26.2, "lon": 91.7,
        "sar_threshold_db": -16,
        "flood_areas": {2022: 4812.3, 2023: 3601.7, 2024: 4102.8},
        "peak_year": 2022,
        "chronic_km2": 1823.4,
        "chronic_pop": 2300000,
        "chronic_districts": ["Dhubri", "Morigaon", "Barpeta", "Goalpara", "Kamrup"],
        "high_risk_zones": ["Lower Assam Plains", "Brahmaputra Floodplain"],
        "accuracy_pct": 92.39,
        "risk_zones_km2": {"high": 3218.4, "moderate": 5901.2, "low": 8234.7},
        "peak_rainfall_mm": 1587,
        "seasonal_risk": {
            "pre_monsoon": 0.3, "kharif": 0.95, "post_monsoon": 0.6, "rabi": 0.1
        },
    },

    "ganga": {
        "name": "Ganga Plains",
        "state": "Bihar / UP",
        "river": "Ganga",
        "lat": 25.6, "lon": 85.1,
        "sar_threshold_db": -16,
        "flood_areas": {2022: 3821.4, 2023: 4501.2, 2024: 3102.6},
        "peak_year": 2023,
        "chronic_km2": 2103.6,
        "chronic_pop": 3100000,
        "chronic_districts": ["Patna", "Bhagalpur", "Darbhanga", "Muzaffarpur", "Samastipur"],
        "high_risk_zones": ["North Bihar Plains", "Kosi Fan"],
        "accuracy_pct": 89.7,
        "risk_zones_km2": {"high": 2914.8, "moderate": 6203.1, "low": 9401.5},
        "peak_rainfall_mm": 1423,
        "seasonal_risk": {
            "pre_monsoon": 0.2, "kharif": 0.90, "post_monsoon": 0.5, "rabi": 0.1
        },
    },

    "mahanadi": {
        "name": "Mahanadi Delta",
        "state": "Odisha",
        "river": "Mahanadi",
        "lat": 20.5, "lon": 85.8,
        "sar_threshold_db": -16,
        "flood_areas": {2022: 2914.7, 2023: 2103.8, 2024: 3401.5},
        "peak_year": 2024,
        "chronic_km2": 1402.3,
        "chronic_pop": 1800000,
        "chronic_districts": ["Cuttack", "Kendrapara", "Jagatsinghpur", "Puri", "Khordha"],
        "high_risk_zones": ["Mahanadi Delta", "Coastal Odisha"],
        "accuracy_pct": 90.1,
        "risk_zones_km2": {"high": 2103.4, "moderate": 4801.2, "low": 7203.8},
        "peak_rainfall_mm": 1312,
        "seasonal_risk": {
            "pre_monsoon": 0.25, "kharif": 0.88, "post_monsoon": 0.55, "rabi": 0.1
        },
    },

    "krishna": {
        "name": "Krishna River Basin",
        "state": "Andhra Pradesh",
        "river": "Krishna",
        "lat": 16.5, "lon": 80.6,
        "sar_threshold_db": -16,
        "flood_areas": {2022: 1823.5, 2023: 2914.2, 2024: 1502.8},
        "peak_year": 2023,
        "chronic_km2": 892.1,
        "chronic_pop": 1200000,
        "chronic_districts": ["Guntur", "Krishna", "West Godavari", "Prakasam", "Nalgonda"],
        "high_risk_zones": ["Krishna Delta", "Lower Krishna Plains"],
        "accuracy_pct": 88.9,
        "risk_zones_km2": {"high": 1402.3, "moderate": 3201.8, "low": 5803.4},
        "peak_rainfall_mm": 1089,
        "seasonal_risk": {
            "pre_monsoon": 0.2, "kharif": 0.85, "post_monsoon": 0.65, "rabi": 0.15
        },
    },

    "godavari": {
        "name": "Godavari Basin",
        "state": "Telangana / AP",
        "river": "Godavari",
        "lat": 17.0, "lon": 81.8,
        "sar_threshold_db": -16,
        "flood_areas": {2022: 3102.4, 2023: 2801.6, 2024: 3891.3},
        "peak_year": 2024,
        "chronic_km2": 1601.2,
        "chronic_pop": 2100000,
        "chronic_districts": ["East Godavari", "West Godavari", "Khammam", "Bhadradri", "Devanahalli"],
        "high_risk_zones": ["Godavari Delta", "Lower Godavari Plains"],
        "accuracy_pct": 91.1,
        "risk_zones_km2": {"high": 2401.6, "moderate": 5102.3, "low": 7803.9},
        "peak_rainfall_mm": 1198,
        "seasonal_risk": {
            "pre_monsoon": 0.25, "kharif": 0.87, "post_monsoon": 0.60, "rabi": 0.12
        },
    },

    # ── NEW REGIONS ───────────────────────────────────────────

    "narmada": {
        "name": "Narmada Basin",
        "state": "Madhya Pradesh / Gujarat",
        "river": "Narmada",
        "lat": 22.7, "lon": 77.4,
        "sar_threshold_db": -16,
        "flood_areas": {2022: 1823.6, 2023: 2401.3, 2024: 1602.8},
        "peak_year": 2023,
        "chronic_km2": 892.4,
        "chronic_pop": 1100000,
        "chronic_districts": ["Hoshangabad", "Jabalpur", "Narsinghpur", "Bharuch", "Narmadapuram"],
        "high_risk_zones": ["Narmada Valley", "Sardar Sarovar Backwaters"],
        "accuracy_pct": 87.3,
        "risk_zones_km2": {"high": 1203.4, "moderate": 2801.2, "low": 4903.6},
        "peak_rainfall_mm": 1134,
        "seasonal_risk": {
            "pre_monsoon": 0.15, "kharif": 0.82, "post_monsoon": 0.45, "rabi": 0.08
        },
    },

    "tapti": {
        "name": "Tapti Basin",
        "state": "Maharashtra / Gujarat",
        "river": "Tapti",
        "lat": 21.2, "lon": 74.8,
        "sar_threshold_db": -16,
        "flood_areas": {2022: 1203.4, 2023: 1801.2, 2024: 1402.6},
        "peak_year": 2023,
        "chronic_km2": 601.3,
        "chronic_pop": 780000,
        "chronic_districts": ["Surat", "Tapi", "Nandurbar", "Dhule", "Jalgaon"],
        "high_risk_zones": ["Surat Lowlands", "Tapti Floodplain"],
        "accuracy_pct": 86.8,
        "risk_zones_km2": {"high": 801.4, "moderate": 1802.3, "low": 3201.5},
        "peak_rainfall_mm": 987,
        "seasonal_risk": {
            "pre_monsoon": 0.12, "kharif": 0.80, "post_monsoon": 0.40, "rabi": 0.07
        },
    },

    "cauvery": {
        "name": "Cauvery Basin",
        "state": "Karnataka / Tamil Nadu",
        "river": "Cauvery",
        "lat": 12.3, "lon": 77.0,
        "sar_threshold_db": -16,
        "flood_areas": {2022: 1102.3, 2023: 1503.8, 2024: 1301.4},
        "peak_year": 2023,
        "chronic_km2": 542.1,
        "chronic_pop": 890000,
        "chronic_districts": ["Thanjavur", "Tiruvarur", "Nagapattinam", "Mysuru", "Mandya"],
        "high_risk_zones": ["Cauvery Delta", "Thanjavur Plains"],
        "accuracy_pct": 88.2,
        "risk_zones_km2": {"high": 703.4, "moderate": 1601.2, "low": 2903.8},
        "peak_rainfall_mm": 892,
        "seasonal_risk": {
            "pre_monsoon": 0.18, "kharif": 0.75, "post_monsoon": 0.70, "rabi": 0.20
        },
    },

    "damodar": {
        "name": "Damodar Valley",
        "state": "Jharkhand / West Bengal",
        "river": "Damodar",
        "lat": 23.5, "lon": 87.3,
        "sar_threshold_db": -16,
        "flood_areas": {2022: 2103.4, 2023: 1801.6, 2024: 2401.8},
        "peak_year": 2024,
        "chronic_km2": 1012.3,
        "chronic_pop": 1400000,
        "chronic_districts": ["Barddhaman", "Hooghly", "Howrah", "Dhanbad", "Bokaro"],
        "high_risk_zones": ["Damodar Floodplain", "Lower Damodar Valley"],
        "accuracy_pct": 89.4,
        "risk_zones_km2": {"high": 1401.3, "moderate": 3201.8, "low": 5102.4},
        "peak_rainfall_mm": 1203,
        "seasonal_risk": {
            "pre_monsoon": 0.20, "kharif": 0.88, "post_monsoon": 0.50, "rabi": 0.10
        },
    },

    "sabarmati": {
        "name": "Sabarmati Basin",
        "state": "Gujarat / Rajasthan",
        "river": "Sabarmati",
        "lat": 23.0, "lon": 72.6,
        "sar_threshold_db": -16,
        "flood_areas": {2022: 801.3, 2023: 1203.4, 2024: 902.6},
        "peak_year": 2023,
        "chronic_km2": 312.4,
        "chronic_pop": 420000,
        "chronic_districts": ["Ahmedabad", "Gandhinagar", "Mehsana", "Sabarkantha", "Patan"],
        "high_risk_zones": ["Ahmedabad Lowlands", "Sabarmati Floodplain"],
        "accuracy_pct": 85.6,
        "risk_zones_km2": {"high": 401.2, "moderate": 901.4, "low": 1803.8},
        "peak_rainfall_mm": 734,
        "seasonal_risk": {
            "pre_monsoon": 0.10, "kharif": 0.75, "post_monsoon": 0.30, "rabi": 0.05
        },
    },

    "mahi": {
        "name": "Mahi Basin",
        "state": "Gujarat / Rajasthan / MP",
        "river": "Mahi",
        "lat": 22.8, "lon": 73.5,
        "sar_threshold_db": -16,
        "flood_areas": {2022: 712.3, 2023: 1103.4, 2024: 834.6},
        "peak_year": 2023,
        "chronic_km2": 298.7,
        "chronic_pop": 380000,
        "chronic_districts": ["Vadodara", "Anand", "Kheda", "Panchmahal", "Dahod"],
        "high_risk_zones": ["Mahi Delta", "Vadodara Lowlands"],
        "accuracy_pct": 84.9,
        "risk_zones_km2": {"high": 312.4, "moderate": 801.2, "low": 1602.8},
        "peak_rainfall_mm": 812,
        "seasonal_risk": {
            "pre_monsoon": 0.10, "kharif": 0.78, "post_monsoon": 0.35, "rabi": 0.06
        },
    },

    "baitarani": {
        "name": "Baitarani Basin",
        "state": "Odisha / Jharkhand",
        "river": "Baitarani",
        "lat": 21.5, "lon": 86.4,
        "sar_threshold_db": -16,
        "flood_areas": {2022: 1203.4, 2023: 1601.8, 2024: 1401.2},
        "peak_year": 2023,
        "chronic_km2": 612.3,
        "chronic_pop": 820000,
        "chronic_districts": ["Bhadrak", "Jajpur", "Kendujhar", "Balasore", "Mayurbhanj"],
        "high_risk_zones": ["Baitarani Delta", "Lower Odisha Coast"],
        "accuracy_pct": 87.1,
        "risk_zones_km2": {"high": 801.4, "moderate": 1802.3, "low": 3201.5},
        "peak_rainfall_mm": 1089,
        "seasonal_risk": {
            "pre_monsoon": 0.22, "kharif": 0.85, "post_monsoon": 0.55, "rabi": 0.10
        },
    },

    "subarnarekha": {
        "name": "Subarnarekha Basin",
        "state": "Jharkhand / WB / Odisha",
        "river": "Subarnarekha",
        "lat": 22.3, "lon": 86.9,
        "sar_threshold_db": -16,
        "flood_areas": {2022: 912.3, 2023: 1203.4, 2024: 1034.6},
        "peak_year": 2023,
        "chronic_km2": 412.8,
        "chronic_pop": 560000,
        "chronic_districts": ["East Singhbhum", "West Midnapore", "Balasore", "Seraikela", "Kharsawan"],
        "high_risk_zones": ["Subarnarekha Delta", "Jamshedpur Lowlands"],
        "accuracy_pct": 86.3,
        "risk_zones_km2": {"high": 601.4, "moderate": 1301.2, "low": 2401.6},
        "peak_rainfall_mm": 1134,
        "seasonal_risk": {
            "pre_monsoon": 0.20, "kharif": 0.83, "post_monsoon": 0.50, "rabi": 0.09
        },
    },

    "indus": {
        "name": "Indus Plains",
        "state": "Punjab / Haryana / J&K",
        "river": "Indus / Sutlej",
        "lat": 30.9, "lon": 75.8,
        "sar_threshold_db": -16,
        "flood_areas": {2022: 2301.4, 2023: 1803.6, 2024: 2103.8},
        "peak_year": 2022,
        "chronic_km2": 1102.3,
        "chronic_pop": 1600000,
        "chronic_districts": ["Ludhiana", "Jalandhar", "Amritsar", "Firozpur", "Fazilka"],
        "high_risk_zones": ["Punjab Doab", "Sutlej Floodplain"],
        "accuracy_pct": 88.4,
        "risk_zones_km2": {"high": 1503.4, "moderate": 3401.2, "low": 5803.6},
        "peak_rainfall_mm": 812,
        "seasonal_risk": {
            "pre_monsoon": 0.15, "kharif": 0.80, "post_monsoon": 0.40, "rabi": 0.08
        },
    },

    "luni": {
        "name": "Luni Basin",
        "state": "Rajasthan / Gujarat",
        "river": "Luni",
        "lat": 25.8, "lon": 72.1,
        "sar_threshold_db": -16,
        "flood_areas": {2022: 612.3, 2023: 1203.4, 2024: 803.6},
        "peak_year": 2023,
        "chronic_km2": 231.4,
        "chronic_pop": 290000,
        "chronic_districts": ["Barmer", "Jalor", "Pali", "Jodhpur", "Sirohi"],
        "high_risk_zones": ["Luni Floodplain", "Barmer Lowlands"],
        "accuracy_pct": 83.7,
        "risk_zones_km2": {"high": 301.2, "moderate": 703.4, "low": 1402.8},
        "peak_rainfall_mm": 412,
        "seasonal_risk": {
            "pre_monsoon": 0.05, "kharif": 0.70, "post_monsoon": 0.20, "rabi": 0.03
        },
    },
}

DEFAULT_REGION = "brahmaputra"

# Seasonal descriptions for context
SEASON_DESCRIPTIONS = {
    "pre_monsoon":  "March–May: dry season, low base flow, localised storm risk",
    "kharif":       "June–September: peak monsoon, maximum flood risk",
    "post_monsoon": "October–November: receding waters, secondary flood risk",
    "rabi":         "December–February: winter season, minimal flood risk",
}


# ──────────────────────────────────────────────
# BASE TASK
# ──────────────────────────────────────────────

class BaseTask:
    task_id:      str = ""
    name:         str = ""
    description:  str = ""
    difficulty:   str = "easy"
    max_steps:    int = 6
    available_data: List[str] = []

    def __init__(self, gee_available: bool = False,
                 region: str = DEFAULT_REGION,
                 season: str = "kharif"):
        self.gee_available = gee_available
        self.region_id     = region if region in REGIONS else DEFAULT_REGION
        self.region        = REGIONS[self.region_id]
        self.season        = season if season in SEASON_DESCRIPTIONS else "kharif"

    def get_context(self) -> Dict[str, Any]:
        r = self.region
        fa = r["flood_areas"]
        return {
            "region":           r["name"],
            "state":            r["state"],
            "river":            r["river"],
            "lat":              r["lat"],
            "lon":              r["lon"],
            "years_available":  sorted(fa.keys()),
            "flood_areas_km2":  fa,
            "sar_threshold_db": r["sar_threshold_db"],
            "peak_year":        r["peak_year"],
            "season":           self.season,
            "season_desc":      SEASON_DESCRIPTIONS[self.season],
            "seasonal_risk":    r["seasonal_risk"][self.season],
            "hint":             f"Compare flood extents for {', '.join(str(y) for y in sorted(fa.keys()))} in the {r['name']}.",
        }

    def step(self, response: str, step_num: int) -> Dict[str, Any]:
        raise NotImplementedError


# ──────────────────────────────────────────────
# REWARD HELPERS
# ──────────────────────────────────────────────

def _clamp(v: float) -> float:
    """Reward must be strictly between 0 and 1."""
    return max(0.01, min(float(v), 0.99))

def _extract_numbers(text: str) -> List[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", text)]

def _mentions_any(text: str, terms: List[str]) -> bool:
    tl = text.lower()
    return any(t.lower() in tl for t in terms)

def _penalty_vague(text: str) -> float:
    vague_phrases = [
        "some areas", "many districts", "various regions",
        "flood prone", "several years", "significant flooding",
        "major impact", "affected areas", "heavy rainfall",
        "flood risk exists",
    ]
    hits = sum(1 for p in vague_phrases if p in text.lower())
    return -0.10 * min(hits, 3)

def _causal_score(text: str) -> float:
    causal_terms = ["chirps", "dem", "slope", "hydrosheds", "flow accumulation",
                    "sar", "sentinel", "elevation", "drainage", "catchment",
                    "rainfall", "discharge", "ndwi", "worldpop"]
    hits = sum(1 for t in causal_terms if t in text.lower())
    return min(hits * 0.05, 0.20)


# ──────────────────────────────────────────────
# TASK 1 — EASY
# ──────────────────────────────────────────────

class FloodYearComparisonTask(BaseTask):
    task_id     = "flood_year_comparison"
    name        = "SAR Flood Year Comparison"
    description = (
        "Using Sentinel-1 SAR data, determine which monsoon year (2022–2024) "
        "had the LARGEST flood extent and report the area in square kilometres "
        "for all three years. Explain what drove the difference."
    )
    difficulty  = "easy"
    max_steps   = 6
    available_data = [
        "Sentinel-1 SAR VV (2022–2024 June–Sept)",
        "CHIRPS daily rainfall (2022–2024)",
        "HydroSHEDS flow accumulation (15ACC)",
        "SRTM DEM (30m resolution)",
        "Landsat 8 NDWI permanent water mask",
    ]

    def get_context(self) -> Dict[str, Any]:
        ctx = super().get_context()
        r = self.region
        fa = r["flood_areas"]
        ctx.update({
            "flood_areas_km2": fa,
            "peak_year":       r["peak_year"],
        })
        return ctx

    def step(self, response: str, step_num: int) -> Dict[str, Any]:
        r    = self.region
        fa   = r["flood_areas"]
        nums = _extract_numbers(response)
        score = 0.0

        # Year identification
        peak = r["peak_year"]
        if str(peak) in response:
            score += 0.30

        # Numeric accuracy — check all 3 years
        for yr, area in fa.items():
            for n in nums:
                if abs(n - area) / area < 0.05:
                    score += 0.15
                    break

        # Causal explanation
        score += _causal_score(response)

        # Vague penalty
        score += _penalty_vague(response)

        done = step_num >= self.max_steps
        return {"reward": _clamp(score), "done": done,
                "result": f"Step {step_num}: scored {score:.3f}"}


# ──────────────────────────────────────────────
# TASK 2 — MEDIUM
# ──────────────────────────────────────────────

class DistrictInundationReportTask(BaseTask):
    task_id     = "district_inundation_report"
    name        = "District Chronic Inundation Report"
    description = (
        "Identify districts with CHRONIC inundation (flooded in all 3 years: "
        "2022, 2023, 2024). Report the total chronically inundated area in km², "
        "the estimated affected population, and the primary causal factors "
        "for each district's recurring flood vulnerability."
    )
    difficulty  = "medium"
    max_steps   = 8
    available_data = [
        "Sentinel-1 SAR VV (2022–2024 June–Sept)",
        "CHIRPS daily rainfall (2022–2024)",
        "HydroSHEDS flow accumulation (15ACC)",
        "SRTM DEM (30m resolution)",
        "FAO GAUL district boundaries",
        "WorldPop population density (2020)",
        "Landsat 8 NDWI permanent water mask",
    ]

    def get_context(self) -> Dict[str, Any]:
        ctx = super().get_context()
        r = self.region
        ctx.update({
            "chronic_area_km2":  r["chronic_km2"],
            "chronic_population": r["chronic_pop"],
            "target_districts":  r["chronic_districts"],
        })
        return ctx

    def step(self, response: str, step_num: int) -> Dict[str, Any]:
        r    = self.region
        nums = _extract_numbers(response)
        score = 0.0

        # District names
        hit_districts = sum(1 for d in r["chronic_districts"] if d.lower() in response.lower())
        score += min(hit_districts * 0.12, 0.36)

        # Chronic area
        for n in nums:
            if abs(n - r["chronic_km2"]) / r["chronic_km2"] < 0.10:
                score += 0.20
                break

        # Population
        pop_millions = r["chronic_pop"] / 1e6
        for n in nums:
            if abs(n - r["chronic_pop"]) / r["chronic_pop"] < 0.15 or \
               abs(n - pop_millions) / pop_millions < 0.15:
                score += 0.15
                break

        # Causal
        score += _causal_score(response)

        # Vague penalty
        score += _penalty_vague(response)

        done = step_num >= self.max_steps
        return {"reward": _clamp(score), "done": done,
                "result": f"Step {step_num}: scored {score:.3f}"}


# ──────────────────────────────────────────────
# TASK 3 — HARD
# ──────────────────────────────────────────────

class FloodRiskForecastTask(BaseTask):
    task_id     = "flood_risk_forecast"
    name        = "2025 Monsoon Flood Risk Forecast"
    description = (
        "Using the multi-factor risk model (92%+ accuracy), forecast the "
        "HIGH-RISK flood zones for the 2025 monsoon season. Report zone areas "
        "in km², identify specific geographic zones by name, cite the causal "
        "factors (CHIRPS trend, DEM, slope, flow accumulation), and recommend "
        "early warning priorities."
    )
    difficulty  = "hard"
    max_steps   = 10
    available_data = [
        "Sentinel-1 SAR VV (2022–2024 June–Sept)",
        "CHIRPS daily rainfall + 10-year trend (2015–2024)",
        "HydroSHEDS flow accumulation (15ACC)",
        "SRTM DEM + slope (30m resolution)",
        "FAO GAUL district boundaries",
        "WorldPop population density (2020)",
        "Landsat 8 NDWI permanent water mask",
        "Multi-factor risk model (SVM + Random Forest ensemble)",
    ]

    def get_context(self) -> Dict[str, Any]:
        ctx = super().get_context()
        r = self.region
        ctx.update({
            "model_accuracy_pct": r["accuracy_pct"],
            "risk_zones_km2":     r["risk_zones_km2"],
            "high_risk_zones":    r["high_risk_zones"],
            "peak_rainfall_mm":   r["peak_rainfall_mm"],
        })
        return ctx

    def step(self, response: str, step_num: int) -> Dict[str, Any]:
        r    = self.region
        rz   = r["risk_zones_km2"]
        nums = _extract_numbers(response)
        score = 0.0

        # Model accuracy cited
        for n in nums:
            if abs(n - r["accuracy_pct"]) < 2.0:
                score += 0.15
                break

        # Risk zone areas
        for zone_val in rz.values():
            for n in nums:
                if abs(n - zone_val) / zone_val < 0.08:
                    score += 0.12
                    break

        # High-risk zone names
        hit_zones = sum(1 for z in r["high_risk_zones"] if z.lower() in response.lower())
        score += min(hit_zones * 0.10, 0.20)

        # Causal factors
        score += _causal_score(response)

        # Early warning / recommendation language
        if _mentions_any(response, ["early warning", "evacuate", "alert", "priority", "recommend"]):
            score += 0.05

        # Vague penalty
        score += _penalty_vague(response)

        done = step_num >= self.max_steps
        return {"reward": _clamp(score), "done": done,
                "result": f"Step {step_num}: scored {score:.3f}"}


# ──────────────────────────────────────────────
# REGISTRY
# ──────────────────────────────────────────────

TASK_REGISTRY: Dict[str, type] = {
    "flood_year_comparison":      FloodYearComparisonTask,
    "district_inundation_report": DistrictInundationReportTask,
    "flood_risk_forecast":        FloodRiskForecastTask,
}