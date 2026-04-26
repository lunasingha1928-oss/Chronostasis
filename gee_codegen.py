"""
gee_codegen.py — Google Earth Engine Script Generator
Generates complete, runnable GEE JavaScript for flood analysis
across all 15 Indian river basins in Chronostasis.
"""

# ── Basin config: FAO GAUL state names + coordinates ─────────────────────────
BASIN_CONFIG = {
    "brahmaputra": {
        "name": "Brahmaputra Valley",
        "state": "Assam",
        "gaul_field": "ADM1_NAME",
        "gaul_value": "Assam",
        "gaul_level": 1,
        "center_lat": 26.2,
        "center_lon": 92.5,
        "zoom": 7,
        "sar_threshold": -16,
        "dem_max": 1000,
        "export_folder": "GEE_Chronostasis_Brahmaputra",
    },
    "ganga": {
        "name": "Ganga Basin",
        "state": "Bihar",
        "gaul_field": "ADM1_NAME",
        "gaul_value": "Bihar",
        "gaul_level": 1,
        "center_lat": 25.6,
        "center_lon": 85.1,
        "zoom": 7,
        "sar_threshold": -16,
        "dem_max": 500,
        "export_folder": "GEE_Chronostasis_Ganga",
    },
    "mahanadi": {
        "name": "Mahanadi Basin",
        "state": "Odisha",
        "gaul_field": "ADM1_NAME",
        "gaul_value": "Odisha",
        "gaul_level": 1,
        "center_lat": 20.5,
        "center_lon": 84.7,
        "zoom": 7,
        "sar_threshold": -16,
        "dem_max": 800,
        "export_folder": "GEE_Chronostasis_Mahanadi",
    },
    "krishna": {
        "name": "Krishna Basin",
        "state": "Andhra Pradesh",
        "gaul_field": "ADM1_NAME",
        "gaul_value": "Andhra Pradesh",
        "gaul_level": 1,
        "center_lat": 16.5,
        "center_lon": 79.5,
        "zoom": 7,
        "sar_threshold": -16,
        "dem_max": 900,
        "export_folder": "GEE_Chronostasis_Krishna",
    },
    "godavari": {
        "name": "Godavari Basin",
        "state": "Telangana",
        "gaul_field": "ADM1_NAME",
        "gaul_value": "Telangana",
        "gaul_level": 1,
        "center_lat": 18.0,
        "center_lon": 79.5,
        "zoom": 7,
        "sar_threshold": -16,
        "dem_max": 1000,
        "export_folder": "GEE_Chronostasis_Godavari",
    },
    "narmada": {
        "name": "Narmada Basin",
        "state": "Madhya Pradesh",
        "gaul_field": "ADM1_NAME",
        "gaul_value": "Madhya Pradesh",
        "gaul_level": 1,
        "center_lat": 22.7,
        "center_lon": 77.5,
        "zoom": 7,
        "sar_threshold": -16,
        "dem_max": 1200,
        "export_folder": "GEE_Chronostasis_Narmada",
    },
    "tapti": {
        "name": "Tapti Basin",
        "state": "Maharashtra",
        "gaul_field": "ADM1_NAME",
        "gaul_value": "Maharashtra",
        "gaul_level": 1,
        "center_lat": 21.1,
        "center_lon": 75.8,
        "zoom": 7,
        "sar_threshold": -16,
        "dem_max": 900,
        "export_folder": "GEE_Chronostasis_Tapti",
    },
    "cauvery": {
        "name": "Cauvery Basin",
        "state": "Karnataka",
        "gaul_field": "ADM1_NAME",
        "gaul_value": "Karnataka",
        "gaul_level": 1,
        "center_lat": 12.5,
        "center_lon": 76.5,
        "zoom": 7,
        "sar_threshold": -16,
        "dem_max": 1500,
        "export_folder": "GEE_Chronostasis_Cauvery",
    },
    "damodar": {
        "name": "Damodar Basin",
        "state": "Jharkhand",
        "gaul_field": "ADM1_NAME",
        "gaul_value": "Jharkhand",
        "gaul_level": 1,
        "center_lat": 23.6,
        "center_lon": 85.5,
        "zoom": 7,
        "sar_threshold": -16,
        "dem_max": 600,
        "export_folder": "GEE_Chronostasis_Damodar",
    },
    "sabarmati": {
        "name": "Sabarmati Basin",
        "state": "Gujarat",
        "gaul_field": "ADM1_NAME",
        "gaul_value": "Gujarat",
        "gaul_level": 1,
        "center_lat": 23.0,
        "center_lon": 72.5,
        "zoom": 7,
        "sar_threshold": -16,
        "dem_max": 600,
        "export_folder": "GEE_Chronostasis_Sabarmati",
    },
    "mahi": {
        "name": "Mahi Basin",
        "state": "Gujarat / Rajasthan",
        "gaul_field": "ADM1_NAME",
        "gaul_value": "Gujarat",
        "gaul_level": 1,
        "center_lat": 22.8,
        "center_lon": 73.6,
        "zoom": 7,
        "sar_threshold": -16,
        "dem_max": 700,
        "export_folder": "GEE_Chronostasis_Mahi",
    },
    "baitarani": {
        "name": "Baitarani Basin",
        "state": "Odisha",
        "gaul_field": "ADM1_NAME",
        "gaul_value": "Odisha",
        "gaul_level": 1,
        "center_lat": 21.0,
        "center_lon": 86.0,
        "zoom": 7,
        "sar_threshold": -16,
        "dem_max": 700,
        "export_folder": "GEE_Chronostasis_Baitarani",
    },
    "subarnarekha": {
        "name": "Subarnarekha Basin",
        "state": "Jharkhand / West Bengal",
        "gaul_field": "ADM1_NAME",
        "gaul_value": "Jharkhand",
        "gaul_level": 1,
        "center_lat": 22.5,
        "center_lon": 86.2,
        "zoom": 7,
        "sar_threshold": -16,
        "dem_max": 600,
        "export_folder": "GEE_Chronostasis_Subarnarekha",
    },
    "indus": {
        "name": "Indus Basin",
        "state": "Punjab",
        "gaul_field": "ADM1_NAME",
        "gaul_value": "Punjab",
        "gaul_level": 1,
        "center_lat": 30.9,
        "center_lon": 75.8,
        "zoom": 7,
        "sar_threshold": -16,
        "dem_max": 2000,
        "export_folder": "GEE_Chronostasis_Indus",
    },
    "luni": {
        "name": "Luni Basin",
        "state": "Rajasthan",
        "gaul_field": "ADM1_NAME",
        "gaul_value": "Rajasthan",
        "gaul_level": 1,
        "center_lat": 26.0,
        "center_lon": 72.5,
        "zoom": 7,
        "sar_threshold": -16,
        "dem_max": 500,
        "export_folder": "GEE_Chronostasis_Luni",
    },
}


def generate_gee_script(region_id: str, years: list = None) -> str:
    """
    Generate a complete, runnable GEE JavaScript for a given basin.
    Produces: SAR flood maps per year, flood frequency map,
    CHIRPS rainfall, DEM + hillshade, flow accumulation,
    NDWI permanent water mask, flood risk zones, accuracy assessment,
    and all export tasks.
    """
    if years is None:
        years = [2022, 2023, 2024]

    cfg = BASIN_CONFIG.get(region_id)
    if not cfg:
        raise ValueError(f"Unknown region: {region_id}. "
                         f"Available: {list(BASIN_CONFIG.keys())}")

    name         = cfg["name"]
    state        = cfg["state"]
    gaul_field   = cfg["gaul_field"]
    gaul_value   = cfg["gaul_value"]
    gaul_level   = cfg["gaul_level"]
    lat          = cfg["center_lat"]
    lon          = cfg["center_lon"]
    zoom         = cfg["zoom"]
    threshold    = cfg["sar_threshold"]
    dem_max      = cfg["dem_max"]
    folder       = cfg["export_folder"]

    # Build year-block snippets
    sar_blocks   = _sar_year_blocks(years, threshold, folder)
    freq_block   = _flood_frequency_block(years)
    chirps_block = _chirps_block(years, folder)

    script = f"""// ================================================================
// CHRONOSTASIS — GEE Flood Intelligence Script
// Basin  : {name}  ({state})
// Region : {region_id}
// Years  : {years}
// Generated automatically by Chronostasis gee_codegen.py
// Paste into https://code.earthengine.google.com/ and Run.
// ================================================================

// ── 1. AREA OF INTEREST ─────────────────────────────────────────
var stateFC = ee.FeatureCollection("FAO/GAUL/2015/level{gaul_level}")
  .filter(ee.Filter.eq('{gaul_field}', '{gaul_value}'));
var aoi = stateFC.geometry();
Map.centerObject(aoi, {zoom});
Map.addLayer(stateFC, {{color: 'white'}}, '{state} Boundary');

// ── 2. DEM + HILLSHADE ──────────────────────────────────────────
var dem = ee.Image("USGS/SRTMGL1_003").clip(aoi);
var slope = ee.Terrain.slope(dem);
var hillshade = ee.Terrain.hillshade(dem);
var demVisual = dem.visualize({{
  min: 0, max: {dem_max},
  palette: ['#006837','#78c679','#d9f0a3','#ffffbf','#fd8d3c','#d7191c','#800026']
}});
var hillshadeVisual = hillshade.visualize({{min: 0, max: 255}});
var blended = demVisual.multiply(0.7).add(hillshadeVisual.multiply(0.3)).uint8();
Map.addLayer(blended, {{}}, 'DEM + Hillshade', false);

// ── 3. FLOW ACCUMULATION (HydroSHEDS) ──────────────────────────
var flowAccum = ee.Image("WWF/HydroSHEDS/15ACC").clip(aoi);
var flowAccumLog = flowAccum.log10();
Map.addLayer(flowAccumLog, {{
  min: 0, max: 6,
  palette: ['white','#c6dbef','#6baed6','#2171b5','#08306b']
}}, 'Flow Accumulation', false);
var majorDrainage = flowAccum.gt(1000);
Map.addLayer(majorDrainage.selfMask(), {{palette: ['blue']}}, 'Major Drainage Channels', false);

// ── 4. NDWI PERMANENT WATER MASK ────────────────────────────────
var landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
  .filterBounds(aoi)
  .filterDate('2022-01-01', '2022-03-31')
  .filter(ee.Filter.lt('CLOUD_COVER', 20))
  .median().clip(aoi);
var ndwi = landsat.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI');
var permanentWater = ndwi.gt(0.3);
Map.addLayer(permanentWater.selfMask(), {{palette: ['navy']}}, 'Permanent Water (NDWI)', false);

// ── 5. SENTINEL-1 SAR PRE-FLOOD BASELINE ────────────────────────
// Using Jan-Mar 2022 as shared dry-season baseline
var s1Before = ee.ImageCollection("COPERNICUS/S1_GRD")
  .filterBounds(aoi)
  .filterDate('2022-01-01', '2022-03-31')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .select('VV').mean().clip(aoi);
Map.addLayer(s1Before, {{min: -25, max: 0}}, 'SAR Pre-flood Baseline (Jan-Mar 2022)', false);

// ── 6. SAR FLOOD EXTENT — PER YEAR ──────────────────────────────
// Each year shown in a distinct colour:
//   2022 → Blue  (#2196F3)
//   2023 → Orange (#FF9800)
//   2024 → Red   (#F44336)
// SAR threshold: {threshold} dB

{sar_blocks}

// ── 7. FLOOD FREQUENCY MAP (2022-2023-2024) ──────────────────────
// Yellow = flooded 1 year, Orange = 2 years, Red = all 3 (chronic)

{freq_block}

// ── 8. CHIRPS RAINFALL (Monsoon June-September) ──────────────────
{chirps_block}

// ── 9. FLOOD RISK ZONES ──────────────────────────────────────────
// Multi-factor: flow accumulation + DEM elevation + slope + rainfall
var rain2022chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
  .filterDate('2022-06-01', '2022-09-30')
  .filterBounds(aoi).sum().clip(aoi);

var highRisk = flowAccum.gt(5000)
  .and(dem.lt(60)).and(slope.lt(3)).and(rain2022chirps.gt(300));

var moderateRisk = flowAccum.gt(1000)
  .and(dem.lt(100)).and(slope.lt(6)).and(rain2022chirps.gt(200))
  .and(highRisk.not());

var lowRisk = flowAccum.gt(200)
  .and(dem.lt(150)).and(slope.lt(10)).and(rain2022chirps.gt(150))
  .and(moderateRisk.not()).and(highRisk.not());

var floodRisk = ee.Image(0)
  .where(lowRisk, 1).where(moderateRisk, 2).where(highRisk, 3)
  .selfMask().clip(aoi);

Map.addLayer(floodRisk, {{
  min: 1, max: 3, palette: ['#FFFF00', '#FFA500', '#FF0000']
}}, 'Flood Risk Zones (Low/Moderate/High)');

// Risk area stats
var pixelArea = ee.Image.pixelArea().divide(1e6);
print('=== FLOOD RISK AREA STATS (sq km) ===');
print('High Risk:', floodRisk.eq(3).multiply(pixelArea)
  .reduceRegion({{reducer: ee.Reducer.sum(), geometry: aoi, scale: 500, maxPixels: 1e10}})
  .values().get(0));
print('Moderate Risk:', floodRisk.eq(2).multiply(pixelArea)
  .reduceRegion({{reducer: ee.Reducer.sum(), geometry: aoi, scale: 500, maxPixels: 1e10}})
  .values().get(0));
print('Low Risk:', floodRisk.eq(1).multiply(pixelArea)
  .reduceRegion({{reducer: ee.Reducer.sum(), geometry: aoi, scale: 500, maxPixels: 1e10}})
  .values().get(0));

// ── 10. ACCURACY ASSESSMENT (SAR vs Risk Model) ──────────────────
var s1FloodSeason = ee.ImageCollection("COPERNICUS/S1_GRD")
  .filterBounds(aoi).filterDate('2022-06-01','2022-09-30')
  .filter(ee.Filter.eq('instrumentMode','IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'))
  .select('VV').min().clip(aoi);
var sarFlood = s1FloodSeason.lt({threshold}).and(s1Before.gt({threshold}));
// Remove permanent water from SAR ground truth
var sarFloodClean = sarFlood.and(permanentWater.not()).rename('actual_clean');
var predictedFlood = highRisk.or(moderateRisk).rename('predicted');

var tp = predictedFlood.eq(1).and(sarFloodClean.eq(1));
var fp = predictedFlood.eq(1).and(sarFloodClean.eq(0));
var tn = predictedFlood.eq(0).and(sarFloodClean.eq(0));
var fn = predictedFlood.eq(0).and(sarFloodClean.eq(1));

print('=== ACCURACY ASSESSMENT (NDWI-corrected) ===');
print('TP (sq km):', tp.multiply(pixelArea).reduceRegion({{reducer: ee.Reducer.sum(), geometry: aoi, scale: 1000, maxPixels: 1e10}}).values().get(0));
print('FP (sq km):', fp.multiply(pixelArea).reduceRegion({{reducer: ee.Reducer.sum(), geometry: aoi, scale: 1000, maxPixels: 1e10}}).values().get(0));
print('TN (sq km):', tn.multiply(pixelArea).reduceRegion({{reducer: ee.Reducer.sum(), geometry: aoi, scale: 1000, maxPixels: 1e10}}).values().get(0));
print('FN (sq km):', fn.multiply(pixelArea).reduceRegion({{reducer: ee.Reducer.sum(), geometry: aoi, scale: 1000, maxPixels: 1e10}}).values().get(0));

// ── 11. EXPORT TASKS ─────────────────────────────────────────────
// Uncomment each Export block to run. Tasks appear in the Tasks tab.

// Flood Frequency Map
/*
Export.image.toDrive({{
  image: floodFrequency,
  description: 'FloodFrequency_{region_id}',
  folder: '{folder}',
  fileNamePrefix: 'FloodFrequency_{region_id}_2022_2024',
  region: aoi, scale: 100, maxPixels: 1e10
}});
*/

// Flood Risk Zones
/*
Export.image.toDrive({{
  image: floodRisk,
  description: 'FloodRiskZones_{region_id}',
  folder: '{folder}',
  fileNamePrefix: 'FloodRiskZones_{region_id}',
  region: aoi, scale: 30, maxPixels: 1e10
}});
*/

// DEM
/*
Export.image.toDrive({{
  image: dem,
  description: 'DEM_{region_id}',
  folder: '{folder}',
  fileNamePrefix: 'DEM_{region_id}',
  region: aoi, scale: 30, maxPixels: 1e10
}});
*/

// Boundary shapefile
/*
Export.table.toDrive({{
  collection: stateFC,
  description: 'Boundary_{region_id}',
  folder: '{folder}',
  fileNamePrefix: 'Boundary_{region_id}',
  fileFormat: 'SHP'
}});
*/

print('=== Chronostasis GEE Script loaded for: {name} ===');
print('Layers visible in map. Check Console for stats.');
print('Uncomment Export blocks in section 11 to download data.');
"""
    return script


def generate_all_india_script() -> str:
    """
    Generate a single GEE script that loads all 15 basins
    and displays a nationwide flood frequency comparison.
    """
    basin_rows = []
    for rid, cfg in BASIN_CONFIG.items():
        basin_rows.append(
            f"""  {{id: '{rid}', name: '{cfg["name"]}', state: '{cfg["state"]}', """
            f"""lat: {cfg["center_lat"]}, lon: {cfg["center_lon"]}}}"""
        )
    basins_js = ",\n".join(basin_rows)

    return f"""// ================================================================
// CHRONOSTASIS — ALL-INDIA FLOOD INTELLIGENCE
// 15 River Basins — Nationwide SAR Flood Frequency Comparison
// Generated by Chronostasis gee_codegen.py
// ================================================================

// India boundary
var india = ee.FeatureCollection("FAO/GAUL/2015/level0")
  .filter(ee.Filter.eq('ADM0_NAME', 'India'));
Map.centerObject(india, 5);
Map.addLayer(india, {{color: 'white', fillColor: '00000000'}}, 'India Boundary');

// All basins metadata
var basins = [
{basins_js}
];

// For each basin: load SAR flood frequency and display
// Using a shared national pre-flood baseline (Jan-Mar 2022)
// Flood threshold: -16 dB

basins.forEach(function(b) {{
  var stateFC = ee.FeatureCollection("FAO/GAUL/2015/level1")
    .filter(ee.Filter.eq('ADM1_NAME', b.state));
  var aoi = stateFC.geometry();

  var s1Before = ee.ImageCollection("COPERNICUS/S1_GRD")
    .filterBounds(aoi).filterDate('2022-01-01','2022-03-31')
    .filter(ee.Filter.eq('instrumentMode','IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'))
    .select('VV').mean().clip(aoi);

  var flood = function(yr) {{
    var s1 = ee.ImageCollection("COPERNICUS/S1_GRD")
      .filterBounds(aoi).filterDate(yr+'-06-01', yr+'-09-30')
      .filter(ee.Filter.eq('instrumentMode','IW'))
      .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'))
      .select('VV').min().clip(aoi);
    return s1.lt(-16).and(s1Before.gt(-16));
  }};

  var f22 = flood('2022');
  var f23 = flood('2023');
  var f24 = flood('2024');
  var freq = f22.add(f23).add(f24);

  Map.addLayer(freq.selfMask(), {{
    min:1, max:3, palette:['#FFFF00','#FF6600','#FF0000']
  }}, b.name + ' Flood Frequency', false);
}});

print('All 15 basins loaded. Toggle layers in the Layers panel.');
print('Yellow = flooded 1 year | Orange = 2 years | Red = all 3 (chronic)');
"""


# ── Private helpers ──────────────────────────────────────────────────────────

_YEAR_COLORS = {2022: "#2196F3", 2023: "#FF9800", 2024: "#F44336"}
_YEAR_NAMES  = {2022: "Blue", 2023: "Orange", 2024: "Red"}


def _sar_year_blocks(years: list, threshold: int, folder: str) -> str:
    blocks = []
    for yr in years:
        color = _YEAR_COLORS.get(yr, "#FFFFFF")
        cname = _YEAR_NAMES.get(yr, str(yr))
        blocks.append(f"""// SAR {yr}
var s1Flood{yr} = ee.ImageCollection("COPERNICUS/S1_GRD")
  .filterBounds(aoi).filterDate('{yr}-06-01', '{yr}-09-30')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .select('VV').min().clip(aoi);
var sarFlood{yr} = s1Flood{yr}.lt({threshold}).and(s1Before.gt({threshold}));
Map.addLayer(sarFlood{yr}.selfMask(), {{palette: ['{color}']}}, 'SAR Flood {yr} ({cname})');
print('SAR Flood Area {yr} (sq km):', sarFlood{yr}.multiply(pixelArea)
  .reduceRegion({{reducer: ee.Reducer.sum(), geometry: aoi, scale: 1000, maxPixels: 1e10}})
  .values().get(0));
""")
    return "\n".join(blocks)


def _flood_frequency_block(years: list) -> str:
    add_expr = " + ".join([f"sarFlood{yr}" for yr in years])
    chronic_yr = len(years)
    return f"""var floodFrequency = {add_expr};
Map.addLayer(floodFrequency.selfMask(), {{
  min: 1, max: {chronic_yr}, palette: ['#FFFF00', '#FF6600', '#FF0000']
}}, 'Flood Frequency {years[0]}-{years[-1]}');
print('Flooded all {chronic_yr} years (chronic, sq km):', floodFrequency.eq({chronic_yr})
  .multiply(pixelArea)
  .reduceRegion({{reducer: ee.Reducer.sum(), geometry: aoi, scale: 1000, maxPixels: 1e10}})
  .values().get(0));
"""


def _chirps_block(years: list, folder: str) -> str:
    blocks = []
    rain_palette = "['#f7fbff','#c6dbef','#9ecae1','#6baed6','#2171b5','#08306b']"
    for i, yr in enumerate(years):
        visible = "true" if i == 0 else "false"
        blocks.append(f"""var rain{yr} = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
  .filterDate('{yr}-06-01', '{yr}-09-30').filterBounds(aoi).sum().clip(aoi);
Map.addLayer(rain{yr}, {{min:0, max:1500, palette:{rain_palette}}}, 'CHIRPS Rainfall {yr}', {visible});
print('Max CHIRPS Rainfall {yr} (mm):', rain{yr}.reduceRegion({{
  reducer: ee.Reducer.max(), geometry: aoi, scale: 5000, maxPixels: 1e9
}}).values().get(0));""")
    return "\n".join(blocks)


# ── Public API (called from server.py) ──────────────────────────────────────

def get_script(region_id: str, years: list = None) -> str:
    return generate_gee_script(region_id, years)


def get_all_india_script() -> str:
    return generate_all_india_script()


def list_regions() -> list:
    return [
        {
            "id": rid,
            "name": cfg["name"],
            "state": cfg["state"],
            "lat": cfg["center_lat"],
            "lon": cfg["center_lon"],
        }
        for rid, cfg in BASIN_CONFIG.items()
    ]


if __name__ == "__main__":
    # Quick test — print Brahmaputra script
    print(generate_gee_script("brahmaputra"))