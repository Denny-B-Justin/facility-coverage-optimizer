# Databricks notebook source
# MAGIC %pip install shapely rasterio pycountry gurobipy folium plotly scikit-learn pyproj gadm

# COMMAND ----------

# MAGIC %pip install -U geopandas shapely

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import urllib.request
import requests
import pandas as pd
import geopandas as gpd
import numpy as np
import pycountry
import rasterio
from rasterio.windows import Window
from gadm import GADMDownloader
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DoubleType, StringType
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import time
import zipfile
import tempfile

# COMMAND ----------

# CONFIGURATION

COUNTRY = "Zambia"
ADM_LEVEL1 = None
ADM_LEVEL2 = None
POPULATION_YEAR = 2025

UC_CATALOG = "prd_mega"
UC_SCHEMA = "pim"
VOLUME_DIR = f"/Volumes/{UC_CATALOG}/sboost4/vboost4"

# COMMAND ----------

def get_country_codes(country_name: str):
    """Look up ISO codes for a country name."""
    try:
        country = pycountry.countries.lookup(country_name)
        return {
            "name": country.name,
            "alpha_2": country.alpha_2,
            "alpha_3": country.alpha_3,
            "numeric": country.numeric,
        }
    except LookupError:
        return None


def ensure_dir(path: str):
    """Create directory if it doesn't exist using dbutils."""
    try:
        dbutils.fs.ls(path)
        print(f"Directory exists: {path}")
    except Exception:
        dbutils.fs.mkdirs(path)
        print(f"Directory created: {path}")


def file_exists(path: str) -> bool:
    """Check if file exists using dbutils."""
    try:
        dbutils.fs.ls(path)
        return True
    except Exception:
        return False


def table_exists(table_name: str) -> bool:
    """Check if UC table exists."""
    try:
        spark.sql(f"DESCRIBE TABLE {table_name}")
        return True
    except Exception:
        return False


def gdf_to_uc_table(gdf: gpd.GeoDataFrame, table_name: str, mode: str = "overwrite"):
    """Save GeoDataFrame to Unity Catalog table with geometry as WKT."""
    pdf = pd.DataFrame(gdf.drop(columns=["geometry"]))
    pdf["geometry_wkt"] = gdf.geometry.apply(lambda g: g.wkt if g else None)

    # Handle duplicate column names (case-insensitive in Spark)
    cols_lower = [c.lower() for c in pdf.columns]
    seen = set()
    new_cols = []
    for i, col in enumerate(pdf.columns):
        col_lower = cols_lower[i]
        if col_lower in seen:
            # Rename duplicate by appending suffix
            new_col = f"{col}_dup"
            while new_col.lower() in seen:
                new_col = f"{new_col}_"
            new_cols.append(new_col)
            seen.add(new_col.lower())
        else:
            new_cols.append(col)
            seen.add(col_lower)
    pdf.columns = new_cols

    # pdf = pdf.reset_index(drop=True)

    # records = pdf.to_dict('records')
    # sdf = spark.createDataFrame(records)
    sdf = spark.createDataFrame(pdf.to_dict('records')) 
    sdf.write.mode(mode).saveAsTable(table_name)
    print(f"Table saved: {table_name} ({len(gdf)} rows)")


def uc_table_to_gdf(table_name: str) -> gpd.GeoDataFrame:
    """Load Unity Catalog table as GeoDataFrame."""
    from shapely.wkt import loads as wkt_loads

    pdf = spark.table(table_name).toPandas()
    pdf["geometry"] = pdf["geometry_wkt"].apply(lambda w: wkt_loads(w) if w else None)
    pdf = pdf.drop(columns=["geometry_wkt"])
    return gpd.GeoDataFrame(pdf, geometry="geometry", crs="EPSG:4326")


def pdf_to_uc_table(pdf: pd.DataFrame, table_name: str, mode: str = "overwrite"):
    """Save pandas DataFrame to Unity Catalog table."""
    sdf = spark.createDataFrame(pdf)
    sdf.write.mode(mode).saveAsTable(table_name)
    print(f"Table saved: {table_name} ({len(pdf)} rows)")

# COMMAND ----------

# EXTRACT: GADM ADMINISTRATIVE BOUNDARIES

def extract_gadm_boundaries(
    country: str,
    adm_level1: str | None,
    adm_level2: str | None,
    table_name: str,
    force: bool = False,
) -> gpd.GeoDataFrame:
    """
    Downloads GADM boundaries and saves to UC table.
    Returns the selected boundary GeoDataFrame.
    """
    if not force and table_exists(table_name):
        print(f"GADM boundaries already exist, loading: {table_name}")
        return uc_table_to_gdf(table_name)

    downloader = GADMDownloader(version="4.0")

    if adm_level1 is not None:
        df_shp = downloader.get_shape_data_by_country_name(country_name=country, ad_level=1)
        selected_gadm = df_shp[df_shp["NAME_1"] == adm_level1]
    elif adm_level2 is not None:
        df_shp = downloader.get_shape_data_by_country_name(country_name=country, ad_level=2)
        selected_gadm = df_shp[df_shp["NAME_2"] == adm_level2]
    else:
        df_shp = downloader.get_shape_data_by_country_name(country_name=country, ad_level=0)
        selected_gadm = df_shp

    gdf_to_uc_table(selected_gadm, table_name)
    return selected_gadm

# COMMAND ----------

# EXTRACT: WORLDPOP POPULATION RASTER

def extract_worldpop_raster(
    country_iso3: str,
    population_year: int,
    output_path: str,
    force: bool = False,
) -> str:
    """
    Downloads WorldPop GeoTIFF raster to Volume.
    Returns the file path.
    """
    if not force and file_exists(output_path):
        print(f"Raster already exists, skipping download: {output_path}")
        return output_path

    url = (
        f"https://data.worldpop.org/GIS/Population/Global_2015_2030/R2024B/"
        f"{population_year}/{country_iso3}/v1/100m/constrained/"
        f"{country_iso3.lower()}_pop_{population_year}_CN_100m_R2024B_v1.tif"
    )
    print(f"Downloading: {url}")

    urllib.request.urlretrieve(url, output_path)
    print(f"WorldPop raster saved: {output_path}")
    return output_path

# COMMAND ----------

# EXTRACT: POPULATION RASTER TO UC TABLE (CHUNKED)

def extract_population_chunked(
    raster_path: str,
    table_name: str,
    chunk_size: int = 1024,
    h3_resolution: int = 8,
    force: bool = False,
) -> int:
    """
    Reads WorldPop raster in chunks and saves populated pixels to UC table.
    Uses windowed reading to avoid loading entire raster into memory.
    Adds H3 index for fast spatial filtering.
    Returns total number of populated pixels.
    """
    if not force and table_exists(table_name):
        count = spark.table(table_name).count()
        print(f"Population table already exists: {table_name} ({count:,} rows)")
        return count

    print(f"Processing raster in chunks: {raster_path}")
    print(f"  H3 resolution: {h3_resolution}")

    schema = StructType([
        StructField("xcoord", DoubleType(), False),
        StructField("ycoord", DoubleType(), False),
        StructField("population", DoubleType(), False),
    ])

    total_pixels = 0
    first_chunk = True

    with rasterio.open(raster_path) as src:
        height = src.height
        width = src.width
        transform_affine = src.transform

        n_row_chunks = (height + chunk_size - 1) // chunk_size
        n_col_chunks = (width + chunk_size - 1) // chunk_size
        total_chunks = n_row_chunks * n_col_chunks

        print(f"  Raster size: {width} x {height}")
        print(f"  Chunk size: {chunk_size} x {chunk_size}")
        print(f"  Total chunks: {total_chunks}")

        chunk_num = 0
        for row_off in range(0, height, chunk_size):
            for col_off in range(0, width, chunk_size):
                chunk_num += 1

                win_height = min(chunk_size, height - row_off)
                win_width = min(chunk_size, width - col_off)
                window = Window(col_off, row_off, win_width, win_height)

                data = src.read(1, window=window)

                rows, cols = np.where(data > 0)
                if len(rows) == 0:
                    continue

                values = data[rows, cols].astype(float)

                abs_rows = rows + row_off
                abs_cols = cols + col_off
                x_coords, y_coords = rasterio.transform.xy(
                    transform_affine, abs_rows, abs_cols, offset="center"
                )

                pdf = pd.DataFrame({
                    "xcoord": np.array(x_coords, dtype=float),
                    "ycoord": np.array(y_coords, dtype=float),
                    "population": values,
                })

                sdf = spark.createDataFrame(pdf, schema=schema)

                # Add H3 index (Photon-accelerated)
                sdf = sdf.withColumn(
                    "h3_index",
                    F.expr(f"h3_longlatash3(xcoord, ycoord, {h3_resolution})")
                )

                if first_chunk:
                    sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)
                    first_chunk = False
                else:
                    sdf.write.mode("append").saveAsTable(table_name)

                total_pixels += len(rows)

                if chunk_num % 50 == 0 or chunk_num == total_chunks:
                    print(f"  Processed chunk {chunk_num}/{total_chunks}, pixels so far: {total_pixels:,}")

    print(f"Population table saved: {table_name} ({total_pixels:,} rows)")
    return total_pixels

# COMMAND ----------

# EXTRACT: HEALTH FACILITIES FROM OSM

def extract_health_facilities_osm(iso_2: str, table_name: str, force: bool = False) -> pd.DataFrame:
    """
    Queries OSM Overpass API for hospitals and clinics.
    Saves to UC table and returns DataFrame.
    """
    if not force and table_exists(table_name):
        print(f"OSM facilities already exist, loading: {table_name}")
        return spark.table(table_name).toPandas()

    def query_osm_amenity(amenity: str) -> pd.DataFrame:
        query = f"""
        [out:json];
        area["ISO3166-1"="{iso_2}"];
        (
          node["amenity"="{amenity}"](area);
          way["amenity"="{amenity}"](area);
          rel["amenity"="{amenity}"](area);
        );
        out center;
        """
        response = requests.get(
            "http://overpass-api.de/api/interpreter",
            params={"data": query},
            timeout=120,
        )
        response.raise_for_status()
        elements = response.json()["elements"]
        df = pd.DataFrame(elements)
        if df.empty:
            return pd.DataFrame(columns=["osm_id", "lat", "lon", "name"])
        df["name"] = df["tags"].apply(lambda x: x.get("name") if isinstance(x, dict) else None)
        df = df.rename(columns={"id": "osm_id"})
        return df[["osm_id", "lat", "lon", "name"]].dropna(subset=["lat", "lon"])

    print(f"Querying OSM for hospitals in {iso_2}...")
    df_hospitals = query_osm_amenity("hospital")
    print(f"  Hospitals: {len(df_hospitals)}")

    print(f"Querying OSM for clinics in {iso_2}...")
    df_clinics = query_osm_amenity("clinic")
    print(f"  Clinics: {len(df_clinics)}")

    df_health = (
        pd.concat([df_hospitals, df_clinics])
        .drop_duplicates(subset="osm_id")
        .reset_index(drop=True)
    )

    pdf_to_uc_table(df_health, table_name)
    return df_health

# COMMAND ----------

# EXTRACT: COPY EXISTING HEALTH FACILITIES FILE (if using pre-curated data)

def extract_existing_facilities(input_path: str, table_name: str, force: bool = False) -> gpd.GeoDataFrame:
    """
    Loads existing health facilities GeoJSON and saves to UC table.
    Use this if you have curated facility data instead of OSM.
    """
    if not force and table_exists(table_name):
        print(f"Facilities already exist, loading: {table_name}")
        return uc_table_to_gdf(table_name)

    gdf = gpd.read_file(input_path)
    gdf_to_uc_table(gdf, table_name)
    return gdf

# COMMAND ----------

# RUN EXTRACTION PIPELINE

iso_codes = get_country_codes(COUNTRY)
ISO_2 = iso_codes["alpha_2"]
ISO_3 = iso_codes["alpha_3"]
print(f"Country: {COUNTRY} | ISO-2: {ISO_2} | ISO-3: {ISO_3}")

# COMMAND ----------

# Extract GADM boundaries
gadm_table = f"{UC_CATALOG}.{UC_SCHEMA}.gadm_boundaries_{ISO_3.lower()}"
selected_gadm_gdf = extract_gadm_boundaries(
    country=COUNTRY,
    adm_level1=ADM_LEVEL1,
    adm_level2=ADM_LEVEL2,
    table_name=gadm_table,
)

# COMMAND ----------

# Extract WorldPop raster (download to Volume)
ensure_dir(VOLUME_DIR)
raster_path = os.path.join(VOLUME_DIR, f"worldpop_{ISO_3.lower()}_{POPULATION_YEAR}.tif")
extract_worldpop_raster(
    country_iso3=ISO_3,
    population_year=POPULATION_YEAR,
    output_path=raster_path,
)

# COMMAND ----------

# Convert raster to UC table using chunked processing
# Long-running – takes 7 minutes to process
population_table = f"{UC_CATALOG}.{UC_SCHEMA}.population_{ISO_3.lower()}_{POPULATION_YEAR}"
extract_population_chunked(
    raster_path=raster_path,
    table_name=population_table,
    chunk_size=1024,
)

# COMMAND ----------

# Extract health facilities
# Option A: Query OSM (uncomment to use)
# facilities_table = f"{UC_CATALOG}.{UC_SCHEMA}.health_facilities_{ISO_3.lower()}_osm"
# extract_health_facilities_osm(ISO_2, facilities_table)

# Option B: Use existing curated file
INPUT_FACILITIES_PATH = f"{VOLUME_DIR}/selected_hosp_input_data.geojson"
facilities_table = f"{UC_CATALOG}.{UC_SCHEMA}.health_facilities_{ISO_3.lower()}"
extract_existing_facilities(INPUT_FACILITIES_PATH, facilities_table)

# COMMAND ----------

# EXTRACTION SUMMARY

print("\n" + "=" * 60)
print("EXTRACTION COMPLETE")
print("=" * 60)
print(f"GADM boundaries:    {gadm_table}")
print(f"Population raster:  {raster_path}")
print(f"Population table:   {population_table}")
print(f"Health facilities:  {facilities_table}")
print("=" * 60)

# COMMAND ----------

def extract_gadm_boundaries_lgu(
    country: str,
    table_name: str,
    force: bool = False,
) -> gpd.GeoDataFrame:
    """
    Downloads GADM level-2 (district/LGU) boundaries for the given country
    and saves to UC table with ONLY two columns:
        - LGU           : district name (e.g. "Lusaka", "Luangwa")
        - geometry_wkt  : polygon geometry in WKT (EPSG:4326)

    Mirrors extract_gadm_boundaries but always uses ad_level=2 and
    normalises the output to the mandatory (LGU, geometry_wkt) schema.

    Args:
        country    : country name recognised by GADMDownloader (e.g. "Zambia")
        table_name : fully-qualified UC table  (catalog.schema.table)
        force      : overwrite even if the table already exists

    Returns:
        GeoDataFrame with columns [LGU, geometry]
    """
    # if not force and table_exists(table_name):
    #     print(f"LGU boundaries already exist, loading: {table_name}")
    #     return uc_table_to_gdf(table_name)

    downloader = GADMDownloader(version="4.0")
    df_shp = downloader.get_shape_data_by_country_name(country_name=country, ad_level=2)
    print(df_shp)
    # Keep only the district name and geometry; rename to mandatory columns
    lgu_gdf = (
        df_shp[["NAME_2", "geometry"]]
        .copy()
        .rename(columns={"NAME_2": "LGU"})
        .reset_index(drop=True)
    )

    print(
        f"Downloaded {len(lgu_gdf)} LGU boundaries for {country} "
        "| Uploading to UC Table"
    )
    # gdf_to_uc_table writes all non-geometry columns + geometry_wkt
    # → the table will have exactly: LGU, geometry_wkt
    # gdf_to_uc_table(lgu_gdf, table_name)
    return lgu_gdf

lgu_table = f"{UC_CATALOG}.{UC_SCHEMA}.gadm_boundaries_lgu_zambia"
lgu_gdf = extract_gadm_boundaries_lgu(
    country=COUNTRY,
    table_name=lgu_table,
)
print(f"LGU count: {len(lgu_gdf)}")
print(lgu_gdf[["LGU"]].head(5))

# COMMAND ----------

print("\n" + "=" * 60)
print("EXTRACTION COMPLETE")
print("=" * 60)
print(f"GADM country boundary: {gadm_table}")
print(f"GADM LGU boundaries:   {lgu_table}  ({len(lgu_gdf)} LGUs)")
print(f"Population raster:     {raster_path}")
print(f"Population table:      {population_table}")
print(f"Health facilities:     {facilities_table}")
print("=" * 60)

# COMMAND ----------

import zipfile
import tempfile

def extract_gadm_boundaries_lgu(
    country_iso3: str,
    table_name: str,
    gadm_version: str = "4.1",
    ad_level: int = 2,
    force: bool = False,
) -> gpd.GeoDataFrame:
    """
    Downloads GADM boundaries at the specified admin level directly from the
    UCDAVIS geodata server (supports GADM 4.1 which has all 116 Zambia districts)
    and saves to UC table with ONLY two columns:
        - LGU           : district name (NAME_2 for ad_level=2)
        - geometry_wkt  : polygon geometry in WKT (EPSG:4326)

    Args:
        country_iso3 : ISO-3 country code  (e.g. "ZMB")
        table_name   : fully-qualified UC table  (catalog.schema.table)
        gadm_version : GADM dataset version, default "4.1"
        ad_level     : admin level to extract  (2 = district/LGU)
        force        : overwrite even if the table already exists

    Returns:
        GeoDataFrame with columns [LGU, geometry]
    """
    if not force and table_exists(table_name):
        print(f"LGU boundaries already exist, loading: {table_name}")
        return uc_table_to_gdf(table_name)

    # ── Build the download URL ────────────────────────────────────────────
    # Example: https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_ZMB_shp.zip
    version_nodot = gadm_version.replace(".", "")
    zip_url = (
        f"https://geodata.ucdavis.edu/gadm/gadm{gadm_version}/shp/"
        f"gadm{version_nodot}_{country_iso3}_shp.zip"
    )
    print(f"Downloading GADM {gadm_version} shapefile: {zip_url}")

    # ── Download zip to a temp file on the Volume ─────────────────────────
    zip_path = os.path.join(VOLUME_DIR, f"gadm{version_nodot}_{country_iso3}_shp.zip")

    if not file_exists(zip_path):
        urllib.request.urlretrieve(zip_url, zip_path)
        print(f"  Downloaded: {zip_path}")
    else:
        print(f"  Zip already cached: {zip_path}")

    # ── Unzip and read the level-specific shapefile ───────────────────────
    # The zip contains files named e.g. gadm41_ZMB_0.shp, gadm41_ZMB_1.shp, gadm41_ZMB_2.shp
    extract_dir = os.path.join(VOLUME_DIR, f"gadm{version_nodot}_{country_iso3}_shp")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
        print(f"  Extracted to: {extract_dir}")

    shp_filename = f"gadm{version_nodot}_{country_iso3}_{ad_level}.shp"
    shp_path = os.path.join(extract_dir, shp_filename)

    lgu_raw_gdf = gpd.read_file(shp_path)
    print(f"  Shapefile loaded: {len(lgu_raw_gdf)} features (GADM {gadm_version}, level {ad_level})")

    # ── Normalise to mandatory schema: LGU + geometry ─────────────────────
    name_col = f"NAME_{ad_level}"          # e.g. NAME_2 for districts
    if name_col not in lgu_raw_gdf.columns:
        raise ValueError(
            f"Expected column '{name_col}' not found. "
            f"Available: {list(lgu_raw_gdf.columns)}"
        )

    lgu_gdf = (
        lgu_raw_gdf[[name_col, "geometry"]]
        .copy()
        .rename(columns={name_col: "LGU"})
        .reset_index(drop=True)
    )

    # Ensure WGS-84
    if lgu_gdf.crs is None or lgu_gdf.crs.to_epsg() != 4326:
        lgu_gdf = lgu_gdf.to_crs(epsg=4326)

    print(
        f"Normalised to {len(lgu_gdf)} LGU boundaries "
        f"| Uploading to UC table: {table_name}"
    )
    # gdf_to_uc_table produces exactly: LGU, geometry_wkt
    gdf_to_uc_table(lgu_gdf, table_name)
    return lgu_gdf


# ============================================================
# ── Execution block — append after health facilities cell ──
# ============================================================

lgu_table = f"{UC_CATALOG}.{UC_SCHEMA}.gadm_boundaries_lgu_zambia"
lgu_gdf = extract_gadm_boundaries_lgu(
    country_iso3=ISO_3,          # "ZMB"  (already resolved above)
    table_name=lgu_table,
    gadm_version="4.1",          # <-- GADM 4.1 = 116 districts
    ad_level=2,
)
print(f"LGU count : {len(lgu_gdf)}")
print(lgu_gdf[["LGU"]].to_string(max_rows=10))

# COMMAND ----------

from datetime import datetime

# COMMAND ----------

import time
from datetime import datetime

import pandas as pd
import requests

# PySpark (available in Databricks / UC environments)
from pyspark.sql import SparkSession

# ── SPARK SESSION (skip if already initialised in your notebook env) ───────────
spark = SparkSession.builder.getOrCreate()


# COMMAND ----------

# EXTRACT: ROAD NETWORK FROM OSM (HISTORICAL, MULTI-COUNTRY)

# OSM highway types and their surface classification
# Paved: asphalt, concrete, paving_stones, sett, cobblestone, metal, wood
# Unpaved: gravel, dirt, mud, grass, sand, compacted, ground, track

HIGHWAY_SURFACE_MAP = {
    # Paved (by highway type default, may be overridden by explicit surface tag)
    "motorway":       "paved",
    "motorway_link":  "paved",
    "trunk":          "paved",
    "trunk_link":     "paved",
    "primary":        "paved",
    "primary_link":   "paved",
    "secondary":      "paved",
    "secondary_link": "paved",
    "tertiary":       "paved",
    "tertiary_link":  "paved",
    "residential":    "paved",
    "living_street":  "paved",
    "service":        "paved",
    # Unpaved (by highway type default)
    "unclassified":   "unknown",   # highly variable — resolved via surface tag
    "track":          "unpaved",
    "path":           "unpaved",
    "footway":        "unpaved",
    "cycleway":       "unpaved",
    "bridleway":      "unpaved",
}

PAVED_SURFACE_TAGS = {
    "asphalt", "concrete", "paving_stones", "sett",
    "cobblestone", "metal", "wood", "concrete:plates",
    "concrete:lanes",
}

UNPAVED_SURFACE_TAGS = {
    "gravel", "dirt", "mud", "grass", "sand", "compacted",
    "ground", "earth", "fine_gravel", "pebblestone", "rock",
    "unpaved", "clay", "soil",
}

SNAPSHOT_YEARS = list(range(2014, datetime.now().year + 1))


def classify_surface(highway_type: str, surface_tag: str | None) -> str:
    """
    Resolves paved / unpaved / unknown for a road element.
    Explicit OSM surface tag always wins; highway type is the fallback.
    """
    if surface_tag:
        tag = surface_tag.lower().strip()
        if tag in PAVED_SURFACE_TAGS:
            return "paved"
        if tag in UNPAVED_SURFACE_TAGS:
            return "unpaved"
    return HIGHWAY_SURFACE_MAP.get(highway_type, "unknown")


def _build_overpass_road_query(
    iso_2: str,
    highway_types: list[str],
    snapshot_date: str | None = None,
) -> str:
    """
    Builds an Overpass QL query for road ways in a given country.

    Args:
        iso_2:          ISO 3166-1 alpha-2 country code (e.g. "NG").
        highway_types:  List of OSM highway values to fetch.
        snapshot_date:  ISO-8601 date string "YYYY-MM-01" for historical
                        snapshots. None = current data.
    """
    date_directive = f'[date:"{snapshot_date}T00:00:00Z"]' if snapshot_date else ""
    union_clauses = "\n  ".join(
        f'way["highway"="{ht}"](area);' for ht in highway_types
    )
    return f"""
    [out:json]{date_directive}[timeout:300];
    area["ISO3166-1"="{iso_2}"];
    (
      {union_clauses}
    );
    out center tags;
    """


def _query_overpass(query: str, retries: int = 3, backoff: int = 30) -> list[dict]:
    """Robust Overpass fetch with retry / exponential back-off."""
    url = "https://overpass-api.de/api/interpreter"
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params={"data": query}, timeout=300)
            response.raise_for_status()
            return response.json().get("elements", [])
        except requests.exceptions.HTTPError as e:
            # 429 / 504 → server busy, always retry
            if attempt < retries:
                wait = backoff * attempt
                print(f"  [Attempt {attempt}] HTTP error: {e}. Retrying in {wait}s…")
                time.sleep(wait)
            else:
                raise
        except requests.exceptions.Timeout:
            if attempt < retries:
                print(f"  [Attempt {attempt}] Timeout. Retrying in {backoff}s…")
                time.sleep(backoff)
            else:
                raise
    return []


def _parse_road_elements(elements: list[dict], iso_2: str, snapshot_year: int) -> pd.DataFrame:
    """
    Parses raw Overpass elements into a flat DataFrame.

    Columns:
        osm_id, iso_2, snapshot_year, highway_type, name,
        surface_tag, surface_class, lat, lon, osm_type
    """
    rows = []
    for el in elements:
        tags = el.get("tags", {}) or {}
        highway_type = tags.get("highway", "")
        surface_tag = tags.get("surface") or tags.get("tracktype")

        # Lat/lon: ways expose a "center"; nodes are direct
        center = el.get("center", {})
        lat = center.get("lat") or el.get("lat")
        lon = center.get("lon") or el.get("lon")

        rows.append({
            "osm_id":        el.get("id"),
            "osm_type":      el.get("type"),          # node / way / relation
            "iso_2":         iso_2,
            "snapshot_year": snapshot_year,
            "highway_type":  highway_type,
            "name":          tags.get("name"),
            "ref":           tags.get("ref"),          # road number / route ref
            "surface_tag":   surface_tag,              # raw OSM surface value
            "surface_class": classify_surface(highway_type, surface_tag),
            "lanes":         tags.get("lanes"),
            "maxspeed":      tags.get("maxspeed"),
            "oneway":        tags.get("oneway"),
            "lat":           lat,
            "lon":           lon,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Coerce numeric columns
    for col in ["lat", "lon", "osm_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def extract_roads_osm(
    iso_2_codes: str | list[str],
    base_table_name: str,
    highway_types: list[str] | None = None,
    start_year: int = 2014,
    end_year: int | None = None,
    snapshot_month: str = "01",     # Jan snapshot per year → comparable YoY
    force: bool = False,
) -> pd.DataFrame:
    """
    Queries the OSM Overpass API for road networks across one or more countries
    and across a range of annual snapshots (default: 2014 → current year).

    For each (country, year) pair the function:
      1. Checks whether a partitioned UC table already exists (skip if not force).
      2. Fetches all road ways matching the requested highway types.
      3. Classifies each road as paved / unpaved / unknown.
      4. Persists the slice to Unity Catalog and accumulates results.

    Args:
        iso_2_codes:      Single ISO-2 code or list (e.g. "NG" or ["NG","KE","GH"]).
        base_table_name:  UC table prefix. Partitioned slices are saved as
                          ``{base_table_name}_{iso_2}_{year}``.
        highway_types:    OSM highway values to include. Defaults to the full
                          road network (motorway → track).
        start_year:       First snapshot year (inclusive). Default 2014.
        end_year:         Last snapshot year (inclusive). Default = current year.
        snapshot_month:   Month used for the Overpass date directive (``"01"``–
                          ``"12"``). January gives stable YoY snapshots.
        force:            Re-fetch even if the UC slice already exists.

    Returns:
        pd.DataFrame with columns:
            osm_id, osm_type, iso_2, snapshot_year, highway_type, name, ref,
            surface_tag, surface_class, lanes, maxspeed, oneway, lat, lon
    """
    # ------------------------------------------------------------------ setup
    if highway_types is None:
        highway_types = list(HIGHWAY_SURFACE_MAP.keys())

    if isinstance(iso_2_codes, str):
        iso_2_codes = [iso_2_codes]

    end_year = end_year or datetime.now().year
    years = list(range(start_year, end_year + 1))

    all_frames: list[pd.DataFrame] = []
    total_slices = len(iso_2_codes) * len(years)
    processed = 0

    # -------------------------------------------------- country × year loop
    for iso_2 in iso_2_codes:
        iso_2 = iso_2.upper()

        for year in years:
            processed += 1
            slice_table = f"{base_table_name}_{iso_2.lower()}_{year}"
            snapshot_date = f"{year}-{snapshot_month}-01"

            print(
                f"[{processed}/{total_slices}] "
                f"{iso_2} / {year}  →  {slice_table}"
            )

            # ── cache hit ────────────────────────────────────────────────────
            if not force and table_exists(slice_table):
                print(f"  Already cached — loading from UC.")
                df_slice = spark.table(slice_table).toPandas()
                all_frames.append(df_slice)
                continue

            # ── fetch from Overpass ──────────────────────────────────────────
            # Current-year snapshot: omit date directive (latest data)
            use_snapshot = snapshot_date if year < datetime.now().year else None

            query = _build_overpass_road_query(iso_2, highway_types, use_snapshot)

            try:
                elements = _query_overpass(query)
                print(f"  Raw elements fetched: {len(elements):,}")
            except Exception as exc:
                print(f"  ⚠ Failed to fetch {iso_2}/{year}: {exc}. Skipping.")
                continue

            if not elements:
                print(f"  No roads found — skipping persist.")
                continue

            # ── parse & classify ─────────────────────────────────────────────
            df_slice = _parse_road_elements(elements, iso_2, year)
            print(
                f"  Roads parsed: {len(df_slice):,}  "
                f"| paved: {(df_slice.surface_class == 'paved').sum():,}  "
                f"| unpaved: {(df_slice.surface_class == 'unpaved').sum():,}  "
                f"| unknown: {(df_slice.surface_class == 'unknown').sum():,}"
            )

            # ── persist to UC ────────────────────────────────────────────────
            # pdf_to_uc_table(df_slice, slice_table)
            all_frames.append(df_slice)

            # Polite delay between Overpass calls to avoid 429s
            time.sleep(5)

    # ---------------------------------------------------------------- combine
    if not all_frames:
        print("No data collected across all country/year combinations.")
        return pd.DataFrame()

    df_all = pd.concat(all_frames, ignore_index=True)
    print(
        f"\nDone. Total road segments: {len(df_all):,} "
        f"across {len(iso_2_codes)} countries × {len(years)} years."
    )
    return df_all

# COMMAND ----------

import time
from datetime import datetime

df_roads = extract_roads_osm(
    iso_2_codes=["NG"],
    base_table_name="prd_mega.sgpbpi163.osm_roads",
    start_year=2024,
    end_year = 2025,
    force=False,
)

# Filter to paved roads only across all countries
df_paved = df_roads[df_roads["surface_class"] == "paved"]

print(df_roads.head())

# Year-over-year road count by country
df_roads.groupby(["iso_2", "snapshot_year", "surface_class"]).size().unstack()

# COMMAND ----------

df_roads

# COMMAND ----------

df_roads.head()

# COMMAND ----------

# ─────────────────────────────────────────────────────────────────────────────
# EXTRACT: HEALTH FACILITIES FROM OSM  (v2 – multi-tag, polygon-aware)
# ─────────────────────────────────────────────────────────────────────────────

from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import requests, time, pandas as pd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

def _query_overpass(query: str, timeout: int = 180, retries: int = 2) -> list:
    """POST an Overpass QL query, rotating mirrors on failure. Returns element list."""
    for mirror in _OVERPASS_MIRRORS:
        for attempt in range(retries):
            try:
                resp = requests.get(mirror, params={"data": query}, timeout=timeout)
                resp.raise_for_status()
                return resp.json().get("elements", [])
            except Exception as exc:
                wait = 5 * (attempt + 1)
                print(f"  [warn] {mirror} attempt {attempt+1} failed: {exc}. Retrying in {wait}s…")
                time.sleep(wait)
        print(f"  [warn] All retries exhausted for mirror: {mirror}. Trying next mirror.")
    raise RuntimeError("All Overpass mirrors failed – check connectivity or rate limits.")


def _build_facility_query(iso_2: str, amenity_vals: list[str], healthcare_vals: list[str], building_vals: list[str]) -> str:
    """
    Build an Overpass QL query that fetches:
      • nodes / ways / relations for every amenity= and healthcare= value
      • ways for every building= value  (buildings rarely have relations)

    Uses `out geom` so closed ways return full polygon coordinates,
    enabling proper area construction rather than just bounding-box centres.
    """
    lines = []
    for v in amenity_vals:
        lines += [
            f'  node["amenity"="{v}"](area.a);',
            f'  way["amenity"="{v}"](area.a);',
            f'  relation["amenity"="{v}"](area.a);',
        ]
    for v in healthcare_vals:
        lines += [
            f'  node["healthcare"="{v}"](area.a);',
            f'  way["healthcare"="{v}"](area.a);',
            f'  relation["healthcare"="{v}"](area.a);',
        ]
    for v in building_vals:
        lines += [
            f'  way["building"="{v}"](area.a);',
            f'  relation["building"="{v}"](area.a);',
        ]

    return (
        f'[out:json][timeout:180];\n'
        f'area["ISO3166-1"="{iso_2}"]->.a;\n'
        f'(\n'
        + "\n".join(lines) +
        f'\n);\n'
        f'out geom;\n'           # full geometry for ways/relations; lat/lon for nodes
    )


def _way_geometry_to_polygon(geometry: list) -> Polygon | None:
    """
    Convert a way's `geometry` array (list of {lat, lon} dicts) to a Shapely
    Polygon if closed (first coord == last coord), else return None.
    """
    coords = [(g["lon"], g["lat"]) for g in geometry if "lon" in g and "lat" in g]
    if len(coords) < 4:
        return None
    # Overpass may or may not repeat the closing node – normalise.
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    try:
        poly = Polygon(coords)
        return poly if poly.is_valid else poly.buffer(0)   # fix self-intersections
    except Exception:
        return None


def _relation_to_polygon(members: list) -> Polygon | None:
    """
    Assemble a Shapely geometry from a relation's outer-ring members.
    Each outer member should carry its own `geometry` array (returned by `out geom`).
    """
    outer_rings = []
    for m in members:
        if m.get("role") != "outer" or not m.get("geometry"):
            continue
        coords = [(g["lon"], g["lat"]) for g in m["geometry"] if "lon" in g and "lat" in g]
        if len(coords) >= 3:
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            try:
                ring = Polygon(coords)
                outer_rings.append(ring if ring.is_valid else ring.buffer(0))
            except Exception:
                continue
    if not outer_rings:
        return None
    merged = unary_union(outer_rings)
    return merged if not merged.is_empty else None


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def _elements_to_deduplicated_points(elements: list) -> pd.DataFrame:
    """
    Convert raw Overpass elements to a flat DataFrame of deduplicated points.

    Strategy
    --------
    1. Parse all elements into two buckets:
         • `point_records`   – nodes, or open ways (rare for amenities)
         • `polygon_records` – closed ways and relations
    2. Build the union of all polygons.
    3. Drop any point that falls *inside* that union  (it's a sub-element of
       a larger mapped area and would produce a duplicate facility).
    4. Convert each polygon to its centroid.
    5. Concatenate surviving points + polygon centroids.
    """
    point_records   = []   # dicts with osm_id, osm_type, lat, lon, name, operator, geometry(Point)
    polygon_records = []   # dicts with osm_id, osm_type, name, operator, geometry(Polygon/Multi)

    for el in elements:
        etype  = el.get("type")
        osm_id = el.get("id")
        tags   = el.get("tags") or {}
        name   = tags.get("name") or tags.get("name:en")
        operator = tags.get("operator")
        operator_type = tags.get("operator:type")          # e.g. "private", "public", "ngo"

        shared = dict(osm_id=osm_id, name=name, operator=operator, operator_type=operator_type)

        # ── Node ──────────────────────────────────────────────────────────────
        if etype == "node":
            lat, lon = el.get("lat"), el.get("lon")
            if lat is not None and lon is not None:
                point_records.append({**shared, "osm_type": "node",
                                       "lat": lat, "lon": lon,
                                       "geometry": Point(lon, lat)})

        # ── Way ───────────────────────────────────────────────────────────────
        elif etype == "way":
            geom_list = el.get("geometry", [])
            poly = _way_geometry_to_polygon(geom_list)
            if poly is not None:
                polygon_records.append({**shared, "osm_type": "way", "geometry": poly})
            elif geom_list:
                # Open way (unusual for hospital/clinic) – use middle node as point
                mid = geom_list[len(geom_list) // 2]
                lat, lon = mid.get("lat"), mid.get("lon")
                if lat and lon:
                    point_records.append({**shared, "osm_type": "way_open",
                                           "lat": lat, "lon": lon,
                                           "geometry": Point(lon, lat)})

        # ── Relation ──────────────────────────────────────────────────────────
        elif etype == "relation":
            poly = _relation_to_polygon(el.get("members", []))
            if poly is not None:
                polygon_records.append({**shared, "osm_type": "relation", "geometry": poly})

    # ── Containment filter ────────────────────────────────────────────────────
    if polygon_records:
        poly_union = unary_union([p["geometry"] for p in polygon_records])
        # Keep a point only if it does NOT fall strictly inside any polygon
        filtered_points = [
            r for r in point_records
            if not poly_union.contains(r["geometry"])
        ]
    else:
        poly_union       = None
        filtered_points  = point_records

    # ── Polygon → centroid ────────────────────────────────────────────────────
    centroid_rows = []
    for p in polygon_records:
        c = p["geometry"].centroid
        centroid_rows.append({
            "osm_id":        p["osm_id"],
            "osm_type":      p["osm_type"],
            "lat":           c.y,
            "lon":           c.x,
            "name":          p["name"],
            "operator":      p["operator"],
            "operator_type": p["operator_type"],
        })

    point_rows = [
        {k: r[k] for k in ("osm_id", "osm_type", "lat", "lon", "name", "operator", "operator_type")}
        for r in filtered_points
    ]

    all_rows = centroid_rows + point_rows
    if not all_rows:
        return pd.DataFrame(columns=["osm_id", "osm_type", "lat", "lon", "name", "operator", "operator_type"])

    return (
        pd.DataFrame(all_rows)
        .drop_duplicates(subset="osm_id")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

#  Tag schema per facility type.
#  amenity / healthcare / building values to query for each category.
_FACILITY_TAG_SCHEMA = {
    "hospital": {
        "amenity":    ["hospital"],
        "healthcare": ["hospital"],
        "building":   ["hospital"],
    },
    "clinic": {
        "amenity":    ["clinic"],
        "healthcare": ["clinic", "health_post", "health_centre"],   # common in Zambia
        "building":   ["clinic"],
    },
}


def extract_health_facilities_osm(
    iso_2: str,
    table_name: str,
    force: bool = False,
    facility_types: list[str] | None = None,          # default = all in schema
) -> pd.DataFrame:
    """
    Query OSM Overpass API for health facilities, handling three tag schemes:
      • amenity=hospital / clinic
      • healthcare=hospital / clinic / health_post / health_centre
      • building=hospital / clinic

    For each facility type the function:
      1. Fetches nodes, ways, and relations with full geometry (`out geom`).
      2. Builds Shapely polygons for closed ways and relations.
      3. Drops any node/open-way whose point falls inside a larger polygon
         (avoids counting sub-elements of a compound hospital as separate facilities).
      4. Converts each polygon to its centroid.
      5. Concatenates surviving stand-alone points with polygon centroids.

    Output columns
    --------------
    osm_id, osm_type, lat, lon, name, operator, operator_type, facility_type

    operator_type carries the raw OSM `operator:type` tag (e.g. "private", "public",
    "ngo", "government") and can be used for a public/private split downstream.

    Parameters
    ----------
    iso_2          : ISO 3166-1 alpha-2 country code (e.g. "ZM")
    table_name     : Unity Catalog table to read from / write to
    force          : if True, re-query even if table already exists
    facility_types : subset of _FACILITY_TAG_SCHEMA keys to query; default = all
    """
    if not force and table_exists(table_name):
        print(f"OSM facilities already exist, loading: {table_name}")
        return spark.table(table_name).toPandas()

    if facility_types is None:
        facility_types = list(_FACILITY_TAG_SCHEMA.keys())

    all_dfs = []

    for ftype in facility_types:
        schema = _FACILITY_TAG_SCHEMA[ftype]
        print(f"\n── Querying OSM [{ftype}] for {iso_2} ──────────────────────────")

        query    = _build_facility_query(iso_2, schema["amenity"], schema["healthcare"], schema["building"])
        elements = _query_overpass(query)

        n_nodes = sum(1 for e in elements if e["type"] == "node")
        n_ways  = sum(1 for e in elements if e["type"] == "way")
        n_rels  = sum(1 for e in elements if e["type"] == "relation")
        print(f"  Raw elements → nodes: {n_nodes}, ways: {n_ways}, relations: {n_rels}")

        df = _elements_to_deduplicated_points(elements)
        df["facility_type"] = ftype

        n_poly  = sum(1 for e in elements if e["type"] in ("way", "relation"))
        n_pts   = len(df)
        print(f"  After polygon-containment filter + centroid conversion → {n_pts} points  "
              f"(dropped {n_nodes + n_ways + n_rels - n_pts} sub-elements / duplicates)")

        all_dfs.append(df)

    # Cross-facility deduplication: an OSM element tagged as both
    # amenity=hospital AND healthcare=hospital appears in both queries.
    df_health = (
        pd.concat(all_dfs, ignore_index=True)
        .drop_duplicates(subset="osm_id")
        .reset_index(drop=True)
    )

    print(f"\n✓ Total unique facilities: {len(df_health)}")
    print(df_health.groupby(["facility_type", "osm_type"]).size().to_string())

    # pdf_to_uc_table(df_health, table_name)
    return df_health

# COMMAND ----------

facilities_table = f"{UC_CATALOG}.{UC_SCHEMA}.health_facilities_zmb_osm_new"
df_facilities = extract_health_facilities_osm(iso_2="ZM", table_name=facilities_table)

# COMMAND ----------

df_facilities[["osm_id", "osm_type", "lat", "lon", "name", "operator", "operator_type", "facility_type"]].to_json(
    f"{VOLUME_DIR}/new_facilities_zmb_osm.json",
    orient="records",
    indent=2,
    force_ascii=False,
    default_handler=str,
)
