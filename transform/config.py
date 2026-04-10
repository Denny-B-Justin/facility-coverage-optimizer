# Databricks notebook source
# Transform pipeline configuration
# This file is imported by all transform tasks

# COMMAND ----------

# MAGIC %run "../shared/core"

# COMMAND ----------

# MAGIC %run "../shared/settings"

# COMMAND ----------

# Local imports (skipped in Databricks where %run loads modules)
import os
if not os.environ.get("DATABRICKS_RUNTIME_VERSION"):
    from shared.core import (
        get_k_rings,
        get_transform_table_names as _get_transform_table_names,
        build_transform_combinations as _build_transform_combinations,
        H3_EDGE_LENGTH_M,
    )
    from shared.settings import (
        UC_CATALOG,
        UC_SCHEMA,
        COUNTRY,
        ISO_3 as COUNTRY_ISO3,
        POPULATION_YEAR,
    )
else:
    # Databricks: alias the functions loaded via %run
    _get_transform_table_names = get_transform_table_names
    _build_transform_combinations = build_transform_combinations
    COUNTRY_ISO3 = ISO_3

# COMMAND ----------

# CONFIGURATION

# List of admin level 1 regions to process (set to None to process entire country)
# If empty list [], will process all provinces (requires extraction to have run first)
ADM_LEVEL1_LIST = []  # e.g., ["Northern", "North-Western"] or [] for all

# List of distances to analyze (in meters)
DISTANCES_METERS = [2000, 4000, 5000, 10000]  # e.g., [5000, 10000] for 5km and 10km

TRAVEL_API = ""  # "" for buffer, "osm", or "mapbox"
MAPBOX_ACCESS_TOKEN = ""
MAPBOX_MODE = "driving"

POTENTIAL_TYPE = "grid"  # "grid" or "kmeans"
GRID_SPACING = 0.03
N_CLUSTERS = 100

TARGET_NEW_FACILITIES = 50
H3_RESOLUTION = 8  # Must match extraction resolution

# Set to True to recompute cached results
FORCE_RECOMPUTE = False

# Target access rate for LGU equity analysis
TARGET_ACCESS_RATE_PCT = 90.0

# COMMAND ----------

# HELPER FUNCTIONS (partial applications of shared.core functions)


def get_transform_table_names(
    country: str,
    iso3: str,
    adm_level1: str | None,
    population_year: int,
    distance_meters: int,
):
    """Generate table names for transform step based on configuration."""
    return _get_transform_table_names(
        UC_CATALOG, UC_SCHEMA, country, iso3, adm_level1, population_year, distance_meters
    )


def build_transform_combinations():
    """Build list of (province, distance) combinations to process."""
    return _build_transform_combinations(ADM_LEVEL1_LIST, DISTANCES_METERS)
