"""
gee_client.py — Chronostasis Real GEE Data Client
===================================================
Provides real Sentinel-1 SAR flood analysis for ANY point in India
using the GEE Python API with service account authentication.

All functions work with coordinates so the user can query any location,
not just the 15 hardcoded basins.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import ee
    EE_OK = True
except ImportError:
    EE_OK = False

# ── Auth ──────────────────────────────────────────────────────────────────
_GEE_INITIALIZED = False

def init_gee(project: str = None, sa_json: str = None) -> bool:
    global _GEE_INITIALIZED
    if _GEE_INITIALIZED:
        return True
    if not EE_OK:
        return False
    try:
        sa_json = sa_json or os.getenv("GEE_SERVICE_ACCOUNT_JSON")
        project = project or os.getenv("GEE_PROJECT", "chronostasis-gee")
        if sa_json:
            key = sa_json if isinstance(sa_json, dict) else json.loads(sa_json)
            creds = ee.ServiceAccountCredentials(
                email=key["client_email"], key_data=key)
            ee.Initialize(creds, project=project)
        else:
            ee.Initialize(project=project)
        _GEE_INITIALIZED = True
        print("[GEE] Initialized successfully", flush=True)
        return True
    except Exception as e:
        print(f"[GEE] Init failed: {e}", flush=True)
        return False


def gee_available() -> bool:
    return _GEE_INITIALIZED and EE_OK


# ── Core helpers ──────────────────────────────────────────────────────────
def make_aoi(lat: float, lon: float, radius_km: float = 100) -> Any:
    """Create circular AOI around a point."""
    return ee.Geometry.Point([lon, lat]).buffer(radius_km * 1000)


def make_rect_aoi(min_lon: float, min_lat: float,
                   max_lon: float, max_lat: float) -> Any:
    return ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])


def _get_s1_monsoon(aoi, year: int):
    """Sentinel-1 SAR VV descending, monsoon season (Jun-Sep)."""
    return (ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(f"{year}-06-01", f"{year}-09-30")
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
        .select("VV")
        .median())


def _get_s1_dry(aoi, year: int):
    """Sentinel-1 SAR VV, dry season baseline (Nov-Mar)."""
    return (ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(f"{year-1}-11-01", f"{year}-03-31")
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
        .select("VV")
        .median())


def _get_perm_water(aoi):
    """Permanent water mask from Landsat 8 NDWI."""
    def add_ndwi(img):
        nir   = img.select("SR_B5").multiply(0.0000275).add(-0.2)
        green = img.select("SR_B3").multiply(0.0000275).add(-0.2)
        return nir.subtract(green).divide(nir.add(green)).rename("NDWI")

    return (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(aoi)
        .filterDate("2020-01-01", "2022-05-31")
        .filter(ee.Filter.lt("CLOUD_COVER", 20))
        .map(add_ndwi)
        .median()
        .gt(0.3))


def _flood_mask(sar_monsoon, perm_water, threshold: float = -16):
    """Clean flood mask: SAR below threshold, not permanent water."""
    raw = sar_monsoon.lt(threshold).And(perm_water.Not())
    return (raw
        .focal_min(radius=1, kernelType="square", units="pixels")
        .focal_max(radius=1, kernelType="square", units="pixels"))


def _compute_area_km2(mask, aoi, scale: int = 100) -> float:
    """Returns area of mask in km²."""
    area = (mask
        .multiply(ee.Image.pixelArea().divide(1e6))
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=scale,
            maxPixels=1e10))
    result = area.getInfo()
    # VV is the band name for SAR-derived masks
    for key in ["VV", "constant", "NDWI"]:
        if key in result and result[key] is not None:
            return round(float(result[key]), 2)
    # Fallback — first numeric value
    for v in result.values():
        if v is not None:
            try:
                return round(float(v), 2)
            except:
                pass
    return 0.0


# ── Public API ────────────────────────────────────────────────────────────

def get_flood_stats(lat: float, lon: float,
                    radius_km: float = 100,
                    threshold: float = -16,
                    timeout: int = 60) -> Dict[str, Any]:
    """
    Returns real SAR flood stats for any lat/lon point in India.

    Returns:
        {
          "flood_areas_km2": {2022: float, 2023: float, 2024: float},
          "peak_year": int,
          "chronic_km2": float,
          "lat": float, "lon": float,
          "radius_km": float,
        }
    """
    if not gee_available():
        return {"error": "GEE not available", "mock": True}

    try:
        aoi       = make_aoi(lat, lon, radius_km)
        perm_w    = _get_perm_water(aoi)
        flood_areas = {}

        for year in [2022, 2023, 2024]:
            sar   = _get_s1_monsoon(aoi, year)
            flood = _flood_mask(sar, perm_w, threshold)
            area  = _compute_area_km2(flood, aoi)
            flood_areas[year] = area

        # Chronic = flooded all 3 years
        floods = {}
        for year in [2022, 2023, 2024]:
            sar       = _get_s1_monsoon(aoi, year)
            floods[year] = _flood_mask(sar, perm_w, threshold)

        chronic = floods[2022].And(floods[2023]).And(floods[2024])
        chronic_km2 = _compute_area_km2(chronic, aoi)
        peak_year   = max(flood_areas, key=flood_areas.get)

        return {
            "flood_areas_km2": flood_areas,
            "peak_year":       peak_year,
            "chronic_km2":     chronic_km2,
            "lat":             lat,
            "lon":             lon,
            "radius_km":       radius_km,
            "sar_threshold_db": threshold,
            "mock":            False,
        }

    except Exception as e:
        print(f"[GEE] get_flood_stats error: {e}", flush=True)
        return {"error": str(e), "mock": True}


def get_risk_zones(lat: float, lon: float,
                   radius_km: float = 100) -> Dict[str, Any]:
    """
    Returns multi-factor flood risk zone areas (high/moderate/low) for any point.
    Uses: flood frequency + CHIRPS rainfall + DEM elevation + HydroSHEDS flow
    """
    if not gee_available():
        return {"error": "GEE not available", "mock": True}

    try:
        aoi    = make_aoi(lat, lon, radius_km)
        perm_w = _get_perm_water(aoi)

        # Flood frequency (0-3)
        floods = {}
        for year in [2022, 2023, 2024]:
            sar = _get_s1_monsoon(aoi, year)
            floods[year] = _flood_mask(sar, perm_w)

        freq = floods[2022].add(floods[2023]).add(floods[2024])

        # Terrain
        dem      = ee.Image("USGS/SRTMGL1_003").clip(aoi)
        flow_acc = ee.Image("WWF/HydroSHEDS/15ACC").select("b1").clip(aoi)

        # CHIRPS rainfall
        chirps = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .filterBounds(aoi)
            .filterDate("2022-06-01", "2022-09-30")
            .sum()
            .clip(aoi))

        # Normalise
        freq_n  = freq.divide(3)
        rain_n  = chirps.subtract(400).divide(1400).clamp(0, 1)
        elev_n  = ee.Image(1).subtract(dem.divide(500).clamp(0, 1))
        flow_n  = flow_acc.log().divide(12).clamp(0, 1)

        # Weighted risk score
        risk = (freq_n.multiply(0.35)
            .add(rain_n.multiply(0.25))
            .add(elev_n.multiply(0.20))
            .add(flow_n.multiply(0.20)))

        high_mask = risk.gt(0.65)
        mod_mask  = risk.gt(0.35).And(risk.lte(0.65))
        low_mask  = risk.lte(0.35)

        pix = ee.Image.pixelArea().divide(1e6)
        high_km2 = _compute_area_km2(high_mask, aoi)
        mod_km2  = _compute_area_km2(mod_mask,  aoi)
        low_km2  = _compute_area_km2(low_mask,  aoi)

        return {
            "risk_zones_km2": {
                "high":     high_km2,
                "moderate": mod_km2,
                "low":      low_km2,
            },
            "lat":        lat,
            "lon":        lon,
            "radius_km":  radius_km,
            "mock":       False,
        }

    except Exception as e:
        print(f"[GEE] get_risk_zones error: {e}", flush=True)
        return {"error": str(e), "mock": True}


def get_flood_tile_url(lat: float, lon: float,
                       year: int = 2022,
                       radius_km: float = 200) -> Dict[str, Any]:
    """
    Returns a GEE map tile URL for flood risk visualization.
    Use with Leaflet: L.tileLayer(tile_url).addTo(map)
    """
    if not gee_available():
        return {"error": "GEE not available", "mock": True}

    try:
        aoi    = make_aoi(lat, lon, radius_km)
        perm_w = _get_perm_water(aoi)

        # Per-year flood masks
        flood_layers = {}
        for y in [2022, 2023, 2024]:
            sar = _get_s1_monsoon(aoi, y)
            flood_layers[y] = _flood_mask(sar, perm_w)

        chronic = flood_layers[2022].And(flood_layers[2023]).And(flood_layers[2024])
        freq    = flood_layers[2022].add(flood_layers[2023]).add(flood_layers[2024])

        # Terrain for risk overlay
        dem      = ee.Image("USGS/SRTMGL1_003").clip(aoi)
        flow_acc = ee.Image("WWF/HydroSHEDS/15ACC").select("b1").clip(aoi)
        chirps   = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .filterBounds(aoi).filterDate(f"{year}-06-01", f"{year}-09-30")
            .sum().clip(aoi))

        freq_n  = freq.divide(3)
        rain_n  = chirps.subtract(400).divide(1400).clamp(0, 1)
        elev_n  = ee.Image(1).subtract(dem.divide(500).clamp(0, 1))
        flow_n  = flow_acc.log().divide(12).clamp(0, 1)
        risk    = (freq_n.multiply(0.35).add(rain_n.multiply(0.25))
                   .add(elev_n.multiply(0.20)).add(flow_n.multiply(0.20)))

        risk_vis = risk.visualize(
            min=0, max=1,
            palette=["00aa00", "ffff00", "ff8800", "ff0000"])

        flood_vis = flood_layers[year].selfMask().visualize(
            palette=["0044ff"])

        chronic_vis = chronic.selfMask().visualize(
            palette=["ff0000"])

        # Get tile URLs
        risk_map    = risk_vis.getMapId()
        flood_map   = flood_vis.getMapId()
        chronic_map = chronic_vis.getMapId()

        def tile_url(map_id_obj):
            return (f"https://earthengine.googleapis.com/v1/"
                    f"{map_id_obj['mapid']}/tiles/{{z}}/{{x}}/{{y}}")

        return {
            "tiles": {
                "risk":    tile_url(risk_map),
                "flood":   tile_url(flood_map),
                "chronic": tile_url(chronic_map),
            },
            "year":       year,
            "lat":        lat,
            "lon":        lon,
            "mock":       False,
        }

    except Exception as e:
        print(f"[GEE] get_flood_tile_url error: {e}", flush=True)
        return {"error": str(e), "mock": True}


def get_chirps_rainfall(lat: float, lon: float,
                        year: int, radius_km: float = 100) -> float:
    """Returns total monsoon season CHIRPS rainfall in mm."""
    if not gee_available():
        return 0.0
    try:
        aoi = make_aoi(lat, lon, radius_km)
        chirps = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .filterBounds(aoi)
            .filterDate(f"{year}-06-01", f"{year}-09-30")
            .sum().clip(aoi))
        result = chirps.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi, scale=5000, maxPixels=1e9).getInfo()
        val = result.get("precipitation", result.get("b1", 0))
        return round(float(val or 0), 1)
    except Exception as e:
        print(f"[GEE] CHIRPS error: {e}", flush=True)
        return 0.0


def query_any_location(lat: float, lon: float,
                        radius_km: float = 80) -> Dict[str, Any]:
    """
    Master function — queries real GEE data for any lat/lon.
    Returns everything needed to populate a dynamic region.
    """
    if not gee_available():
        return {"error": "GEE not available", "mock": True,
                "lat": lat, "lon": lon}

    try:
        flood_stats = get_flood_stats(lat, lon, radius_km)
        risk_zones  = get_risk_zones(lat, lon, radius_km)
        rainfall    = get_chirps_rainfall(lat, lon, 2022, radius_km)
        tile_urls   = get_flood_tile_url(lat, lon, 2022, radius_km * 2)

        return {
            "lat":            lat,
            "lon":            lon,
            "radius_km":      radius_km,
            "flood_areas_km2": flood_stats.get("flood_areas_km2", {}),
            "peak_year":      flood_stats.get("peak_year", 2022),
            "chronic_km2":    flood_stats.get("chronic_km2", 0),
            "risk_zones_km2": risk_zones.get("risk_zones_km2", {}),
            "peak_rainfall_mm": rainfall,
            "tiles":          tile_urls.get("tiles", {}),
            "mock":           False,
        }

    except Exception as e:
        print(f"[GEE] query_any_location error: {e}", flush=True)
        return {"error": str(e), "mock": True,
                "lat": lat, "lon": lon}


# ── Mock fallback (when GEE not available) ────────────────────────────────
MOCK_STATS = {
    "flood_areas_km2": {2022: 4812.3, 2023: 3601.7, 2024: 4102.8},
    "peak_year":       2022,
    "chronic_km2":     1823.4,
    "risk_zones_km2":  {"high": 3218.4, "moderate": 5901.2, "low": 8234.7},
    "peak_rainfall_mm": 1587,
    "tiles":           {},
    "mock":            True,
}

def get_stats_or_mock(lat: float, lon: float,
                      radius_km: float = 80) -> Dict[str, Any]:
    """Returns real GEE stats or mock data if GEE unavailable."""
    if gee_available():
        return query_any_location(lat, lon, radius_km)
    return {**MOCK_STATS, "lat": lat, "lon": lon, "radius_km": radius_km}
