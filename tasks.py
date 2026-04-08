"""
tasks.py — Region-aware OpenEnv Tasks for Chronostasis
"""
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


def _safe_float(s):
    try:
        return float(str(s).replace(",", "").strip())
    except Exception:
        return None

def _extract_nums(text):
    return [x for x in [_safe_float(n) for n in re.findall(r"\d[\d,.]*", text)] if x is not None]


REGIONS = {
    "brahmaputra": {
        "name": "Brahmaputra Valley", "state": "Assam", "river": "Brahmaputra",
        "flood_areas": {2022: 4812.3, 2023: 3601.7, 2024: 4101.2},
        "peak_year": 2022, "chronic_km2": 1247.6, "chronic_pop": 2_400_000,
        "chronic_districts": ["Morigaon", "Dhubri", "Barpeta", "Goalpara", "Kamrup"],
        "high_risk_zones": ["lower Brahmaputra floodplain", "Dhubri district riverbank", "Barpeta wetland belt", "Morigaon char lands"],
        "accuracy_pct": 92.39, "risk_zones_km2": {"high": 3218.4, "moderate": 5901.2, "low": 8240.1},
        "peak_rainfall_mm": 1500, "sar_threshold_db": -16,
    },
    "ganga": {
        "name": "Ganga Plains", "state": "Bihar", "river": "Ganga",
        "flood_areas": {2022: 6241.8, 2023: 4987.3, 2024: 5614.6},
        "peak_year": 2022, "chronic_km2": 2108.4, "chronic_pop": 3_800_000,
        "chronic_districts": ["Darbhanga", "Sitamarhi", "Madhubani", "Saharsa", "Supaul"],
        "high_risk_zones": ["North Bihar Kosi belt", "Darbhanga low-lying plains", "Gandak floodplain", "Bagmati river corridor"],
        "accuracy_pct": 89.74, "risk_zones_km2": {"high": 4812.1, "moderate": 7234.5, "low": 9801.2},
        "peak_rainfall_mm": 1200, "sar_threshold_db": -16,
    },
    "mahanadi": {
        "name": "Mahanadi Delta", "state": "Odisha", "river": "Mahanadi",
        "flood_areas": {2022: 3142.7, 2023: 2801.4, 2024: 3498.6},
        "peak_year": 2024, "chronic_km2": 891.3, "chronic_pop": 1_200_000,
        "chronic_districts": ["Kendrapara", "Jagatsinghpur", "Cuttack", "Puri"],
        "high_risk_zones": ["Mahanadi delta coastal belt", "Kendrapara mangrove zone", "Chilika lake periphery", "Cuttack riverine islands"],
        "accuracy_pct": 90.12, "risk_zones_km2": {"high": 2104.3, "moderate": 4312.7, "low": 6801.4},
        "peak_rainfall_mm": 1350, "sar_threshold_db": -16,
    },
    "krishna": {
        "name": "Krishna River Basin", "state": "Andhra Pradesh", "river": "Krishna",
        "flood_areas": {2022: 2418.9, 2023: 1934.2, 2024: 2701.5},
        "peak_year": 2024, "chronic_km2": 612.7, "chronic_pop": 820_000,
        "chronic_districts": ["Krishna", "Guntur", "West Godavari", "Prakasam"],
        "high_risk_zones": ["Krishna delta estuary", "Guntur low-lying agricultural belt", "Nagarjuna Sagar reservoir downstream", "Krishna-Godavari confluence zone"],
        "accuracy_pct": 88.91, "risk_zones_km2": {"high": 1502.4, "moderate": 3214.8, "low": 5401.3},
        "peak_rainfall_mm": 980, "sar_threshold_db": -16,
    },
    "godavari": {
        "name": "Godavari Basin", "state": "Telangana / Andhra Pradesh", "river": "Godavari",
        "flood_areas": {2022: 3814.2, 2023: 2612.8, 2024: 3109.4},
        "peak_year": 2022, "chronic_km2": 1051.8, "chronic_pop": 1_500_000,
        "chronic_districts": ["Bhadradri Kothagudem", "Mulugu", "East Godavari", "West Godavari"],
        "high_risk_zones": ["Godavari riverine forest belt", "Bhadrachalam flood plains", "Papikonda gorge downstream", "East Godavari delta"],
        "accuracy_pct": 91.08, "risk_zones_km2": {"high": 2401.7, "moderate": 4812.3, "low": 7204.6},
        "peak_rainfall_mm": 1100, "sar_threshold_db": -16,
    },
}

DEFAULT_REGION = "brahmaputra"


class BaseTask(ABC):
    task_id:        str
    name:           str
    difficulty:     str
    max_steps:      int
    available_data: List[str]

    def __init__(self, gee_available: bool = False, region: str = DEFAULT_REGION):
        self.gee_available = gee_available
        self.region_id     = region if region in REGIONS else DEFAULT_REGION
        self.region        = REGIONS[self.region_id]

    @property
    def description(self) -> str:
        return self._make_description()

    @abstractmethod
    def _make_description(self) -> str: ...

    @abstractmethod
    def step(self, action: str, step_num: int) -> Dict[str, Any]: ...

    def get_context(self) -> Dict[str, Any]:
        r = self.region
        return {
            "region": r["name"], "state": r["state"], "river": r["river"],
            "years_available": [2022, 2023, 2024],
            "sar_threshold_db": r["sar_threshold_db"],
            "flood_areas_km2": {str(k): v for k, v in r["flood_areas"].items()},
            "peak_year": r["peak_year"],
        }


class FloodYearComparisonTask(BaseTask):
    task_id    = "flood_year_comparison"
    name       = "Flood Year Comparison"
    difficulty = "easy"
    max_steps  = 6
    available_data = ["Sentinel-1 SAR VV (2022-2024 June-Sept)", "CHIRPS rainfall (2022-2024)", "HydroSHEDS flow accumulation", "SRTM DEM"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rewarded_year   = False
        self._rewarded_areas  = False
        self._rewarded_reason = False

    def _make_description(self) -> str:
        r = self.region
        return (f"Using Sentinel-1 SAR data for the {r['name']} ({r['state']}), "
                f"determine which monsoon year (2022-2024) had the LARGEST flood extent "
                f"and report the area in square kilometres for all three years. "
                f"Explain what drove the difference.")

    def step(self, action: str, step_num: int) -> Dict[str, Any]:
        txt    = action.lower()
        r      = self.region
        reward = 0.0
        notes  = []

        if not self._rewarded_year:
            peak = str(r["peak_year"])
            if peak in txt and any(w in txt for w in ["highest","largest","most","greatest","worst","maximum","peak","biggest","severe"]):
                reward += 0.40
                self._rewarded_year = True
                notes.append(f"Correct peak year {peak} (+0.40)")

        if not self._rewarded_areas:
            years_hit = sum(1 for yr in [2022, 2023, 2024] if str(yr) in action)
            nums      = _extract_nums(action)
            close     = sum(1 for yr_a in r["flood_areas"].values()
                           for n in nums if abs(n - yr_a) / yr_a < 0.15)
            if years_hit >= 3 and close >= 2:
                reward += 0.35
                self._rewarded_areas = True
                notes.append("All 3 year areas reported (+0.35)")
            elif years_hit >= 2 and close >= 1:
                reward += 0.15
                notes.append("Partial areas (+0.15)")

        if not self._rewarded_reason:
            causal = ["rainfall","chirps","precipitation","monsoon","flow","accumulation","dem","elevation","slope","drainage","basin"]
            if sum(1 for kw in causal if kw in txt) >= 2:
                reward += 0.25
                self._rewarded_reason = True
                notes.append("Causal explanation (+0.25)")

        done = (self._rewarded_year and self._rewarded_areas and self._rewarded_reason) or step_num >= self.max_steps
        return {"reward": float(max(0.01, min(reward, 0.99))), "done": done,
                "result": " | ".join(notes) if notes else "No criteria met.", "error": None}

    def get_context(self) -> Dict[str, Any]:
        ctx = super().get_context()
        ctx["hint"] = f"Compare flood extents for 2022, 2023, 2024 in the {self.region['name']}."
        return ctx


class DistrictInundationTask(BaseTask):
    task_id    = "district_inundation_report"
    name       = "Chronic District Inundation Report"
    difficulty = "medium"
    max_steps  = 8
    available_data = ["Sentinel-1 SAR flood extents 2022-2024", "District boundaries (FAO GAUL)", "Flood frequency raster (0-3 years)", "WorldPop population grid", "NDWI permanent water mask"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._found: set    = set()
        self._rewarded_area = False
        self._rewarded_pop  = False

    def _make_description(self) -> str:
        r = self.region
        return (f"Using flood frequency analysis (2022-2024) for the {r['name']} ({r['state']}), "
                f"identify which districts have been CHRONICALLY INUNDATED (flooded all 3 years). "
                f"Report the total chronic area (km2) and estimate the affected population.")

    def step(self, action: str, step_num: int) -> Dict[str, Any]:
        txt    = action.lower()
        r      = self.region
        reward = 0.0
        notes  = []

        for district in r["chronic_districts"]:
            dk = district.lower()
            if dk not in self._found and dk in txt:
                self._found.add(dk)
                reward += 0.10
                notes.append(f"District: {district} (+0.10)")

        if not self._rewarded_area:
            nums = _extract_nums(action)
            if any(abs(n - r["chronic_km2"]) / r["chronic_km2"] < 0.20 for n in nums):
                reward += 0.25
                self._rewarded_area = True
                notes.append(f"Chronic area ~{r['chronic_km2']} km2 (+0.25)")

        if not self._rewarded_pop:
            big = [n for n in _extract_nums(action) if n >= 100000]
            if any(abs(n - r["chronic_pop"]) / r["chronic_pop"] < 0.30 for n in big):
                reward += 0.25
                self._rewarded_pop = True
                notes.append("Population estimate (+0.25)")

        done = (len(self._found) == len(r["chronic_districts"]) and
                self._rewarded_area and self._rewarded_pop) or step_num >= self.max_steps
        return {"reward": float(max(0.01, min(reward, 0.99))), "done": done,
                "result": " | ".join(notes) if notes else f"Districts found: {list(self._found)}", "error": None}

    def get_context(self) -> Dict[str, Any]:
        ctx = super().get_context()
        ctx.update({"target_districts": self.region["chronic_districts"],
                    "chronic_area_km2": self.region["chronic_km2"],
                    "approx_population": self.region["chronic_pop"]})
        return ctx


class FloodRiskForecastTask(BaseTask):
    task_id    = "flood_risk_forecast"
    name       = "Next-Season Flood Risk Forecast"
    difficulty = "hard"
    max_steps  = 10
    available_data = ["Sentinel-1 SAR 2022-2024 flood extents", "CHIRPS rainfall trends",
                      "Flood frequency map (0-3 years)", "Multi-factor risk zones",
                      "Accuracy assessment metrics", "SRTM DEM + slope + HydroSHEDS"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._acc   = False
        self._zones = False
        self._named = 0
        self._rain  = False
        self._year  = False

    def _make_description(self) -> str:
        r = self.region
        return (f"Based on SAR flood history (2022-2024), rainfall trends, and the multi-factor "
                f"risk model ({r['accuracy_pct']}% accuracy) for the {r['name']} ({r['state']}), "
                f"forecast which zones face HIGHEST flood risk in the 2025 monsoon season.")

    def step(self, action: str, step_num: int) -> Dict[str, Any]:
        txt    = action.lower()
        r      = self.region
        reward = 0.0
        notes  = []

        if not self._acc:
            acc_str = str(round(r["accuracy_pct"], 1))
            if acc_str in action or any(k in txt for k in ["precision","recall","f1","accuracy"]):
                reward += 0.15; self._acc = True
                notes.append("Accuracy cited (+0.15)")

        if not self._zones:
            hits = sum(1 for kw in ["high risk","high-risk","moderate risk","low risk"] if kw in txt)
            nums = _extract_nums(action)
            hits += sum(1 for v in r["risk_zones_km2"].values()
                       for n in nums if abs(n - v) / v < 0.05)
            if hits >= 2:
                reward += 0.20; self._zones = True
                notes.append("Risk zones cited (+0.20)")
            elif hits == 1:
                reward += 0.08
                notes.append("Partial zones (+0.08)")

        if self._named < 2:
            for zone in r["high_risk_zones"]:
                if zone.lower() in txt and self._named < 2:
                    self._named += 1; reward += 0.10
                    notes.append(f"Zone: {zone} (+0.10)")

        if not self._rain:
            if any(k in txt for k in ["rainfall","chirps","precipitation","mm",str(r["peak_rainfall_mm"])]):
                reward += 0.15; self._rain = True
                notes.append("Rainfall data (+0.15)")

        if not self._year:
            peak = str(r["peak_year"])
            if peak in txt and any(w in txt for w in ["worst","baseline","reference","peak","benchmark"]):
                reward += 0.10; self._year = True
                notes.append(f"{peak} as benchmark (+0.10)")

        if "2025" in txt and any(w in txt for w in ["forecast","predict","expect","risk","likely"]):
            reward += 0.05
            notes.append("2025 forecast (+0.05)")

        if not notes and len(action) < 120:
            reward -= 0.10
            notes.append("Too vague (-0.10)")

        reward = max(reward, 0.0)
        criteria = self._acc + self._zones + (self._named >= 2) + self._rain + self._year
        done = criteria >= 4 or step_num >= self.max_steps
        return {"reward": float(max(0.01, min(reward, 0.99))), "done": done,
                "result": " | ".join(notes) if notes else "No criteria met.", "error": None}

    def get_context(self) -> Dict[str, Any]:
        ctx = super().get_context()
        r = self.region
        ctx.update({
            "model_accuracy_pct": r["accuracy_pct"],
            "risk_zones_km2": r["risk_zones_km2"],
            "high_risk_zones": r["high_risk_zones"],
            "peak_rainfall_mm": r["peak_rainfall_mm"],
        })
        return ctx


TASK_REGISTRY: Dict[str, type] = {
    "flood_year_comparison":      FloodYearComparisonTask,
    "district_inundation_report": DistrictInundationTask,
    "flood_risk_forecast":        FloodRiskForecastTask,
}