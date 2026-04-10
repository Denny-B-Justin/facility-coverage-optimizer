# Databricks notebook source
# MAGIC %pip install geopandas shapely folium plotly

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# TASK: Run greedy MCLP optimization
#
# Dependencies:
#   - transform/02_coverage.py (Coverage tables)
#
# Outputs:
#   - Pareto frontier visualization
#   - Optimized facility selection maps
#
# This task runs the greedy Maximum Covering Location Problem algorithm
# to determine optimal locations for new facilities.

# COMMAND ----------

import plotly.graph_objects as go

from pyspark.sql import functions as F

# COMMAND ----------

# Import shared utilities and configuration
from shared.env import get_spark, uc_table_to_gdf
from shared.core import solve_mclp_greedy
from transform.config import (
    COUNTRY,
    COUNTRY_ISO3,
    POPULATION_YEAR,
    H3_RESOLUTION,
    TARGET_NEW_FACILITIES,
    get_k_rings,
    get_transform_table_names,
    build_transform_combinations,
)

spark = get_spark()

# Visualization sample size
_POP_VIZ_SAMPLE = 5_000

# COMMAND ----------

# EXECUTE TASK: Process each (province, distance) combination

transform_combinations = build_transform_combinations()
print(f"Will process {len(transform_combinations)} combination(s):")
for adm, dist in transform_combinations:
    region = adm if adm else "Country"
    print(f"  - {region} @ {int(dist/1000)}km")

# COMMAND ----------

# Store results for all combinations
all_pareto_results = {}

for adm_level1, distance_meters in transform_combinations:
    print("\n" + "=" * 60)
    region_name = adm_level1 if adm_level1 else "Country"
    distance_name = f"{int(distance_meters / 1000)}km"
    print(f"OPTIMIZING: {region_name} @ {distance_name}")
    print("=" * 60)

    tables = get_transform_table_names(
        COUNTRY, COUNTRY_ISO3, adm_level1, POPULATION_YEAR, distance_meters
    )
    k_rings = get_k_rings(distance_meters, H3_RESOLUTION)

    # Load prepared data
    print("\nLoading data...")
    population_aoi_sdf = spark.table(tables["population_aoi"]).cache()
    selected_hosp_sdf = spark.table(tables["facilities_h3"]).cache()
    potential_locations_sdf = spark.table(tables["potential_locations"]).cache()
    hosp_coverage_sdf = spark.table(tables["facilities_coverage"])
    potential_coverage_sdf = spark.table(tables["potential_coverage"])

    total_population = population_aoi_sdf.agg(F.sum("population")).collect()[0][0]

    # Load boundaries for visualization
    selected_boundary_gdf = uc_table_to_gdf(tables["boundaries"])
    centroid = selected_boundary_gdf.iloc[0]["geometry"].centroid
    CENTER_LAT = centroid.y
    CENTER_LON = centroid.x

    # Prepare optimization inputs (aggregated by H3 cell)
    print("\nPreparing optimization inputs...")

    # Aggregate population by H3 cell
    pop_by_h3_sdf = population_aoi_sdf.groupBy("h3_index").agg(
        F.sum("population").alias("population")
    )
    pop_h3_rows = pop_by_h3_sdf.collect()
    w = {row["h3_index"]: float(row["population"]) for row in pop_h3_rows}
    I = sorted(w.keys())
    print(f"  Demand cells (H3): {len(I):,}")

    # Facility IDs
    hosp_id_rows = selected_hosp_sdf.select("ID").collect()
    potential_id_rows = potential_locations_sdf.select("ID").collect()
    J_existing = sorted(row["ID"] for row in hosp_id_rows)
    J_potential = sorted(row["ID"] for row in potential_id_rows)
    J = sorted(set(J_existing) | set(J_potential))
    print(f"  Facilities: {len(J):,} (existing: {len(J_existing)}, potential: {len(J_potential)})")

    # Coverage mapping
    all_coverage_sdf = hosp_coverage_sdf.select("facility_ID", "pop_ID").union(
        potential_coverage_sdf.select("facility_ID", "pop_ID")
    )

    coverage_h3_sdf = all_coverage_sdf.join(
        population_aoi_sdf.select("ID", "h3_index"),
        all_coverage_sdf["pop_ID"] == population_aoi_sdf["ID"],
        "inner"
    ).select("facility_ID", "h3_index").distinct()

    JI_rows = (
        coverage_h3_sdf.groupBy("h3_index")
        .agg(F.collect_set("facility_ID").alias("fac_ids"))
        .collect()
    )

    JI = {row["h3_index"]: list(row["fac_ids"]) for row in JI_rows}
    print(f"  Coverage pairs: {sum(len(v) for v in JI.values()):,}")

    # Run greedy MCLP
    print("\nRunning greedy MCLP optimization...")
    pareto_results = solve_mclp_greedy(w, JI, J_existing, J_potential, TARGET_NEW_FACILITIES + 5)

    # Store results
    combo_key = (adm_level1, distance_meters)
    all_pareto_results[combo_key] = {
        "pareto_results": pareto_results,
        "J_existing": J_existing,
        "J_potential": J_potential,
        "total_population": total_population,
        "tables": tables,
    }

    # Calculate current access
    existing_covered_ids_sdf = hosp_coverage_sdf.select("pop_ID").distinct()
    pop_with_access_sdf = population_aoi_sdf.join(
        existing_covered_ids_sdf,
        population_aoi_sdf["ID"] == existing_covered_ids_sdf["pop_ID"],
        "inner",
    ).drop("pop_ID")
    covered_pop_val = pop_with_access_sdf.agg(F.sum("population")).collect()[0][0]
    current_access = round(covered_pop_val * 100 / total_population, 2)

    # Maximum possible coverage
    all_covered_ids_sdf = (
        hosp_coverage_sdf.select("pop_ID")
        .union(potential_coverage_sdf.select("pop_ID"))
        .distinct()
    )
    max_covered_pop = (
        population_aoi_sdf.join(
            all_covered_ids_sdf,
            population_aoi_sdf["ID"] == all_covered_ids_sdf["pop_ID"],
            "inner",
        )
        .agg(F.sum("population"))
        .collect()[0][0]
    )
    max_access_possible = round(max_covered_pop * 100 / total_population, 2)

    # Visualize Pareto frontier
    print("\nVisualizing Pareto frontier...")
    x_values = [len(J_existing) + item["p"] for item in pareto_results]
    y_values = [round(100 * item["objective"] / total_population, 2) for item in pareto_results]

    x_values.insert(0, len(J_existing))
    y_values.insert(0, current_access)

    fig = go.Figure(
        data=go.Scatter(x=x_values, y=y_values, mode="lines+markers", name="Pareto Frontier")
    )
    fig.update_layout(
        title=f"Pareto Frontier: {region_name} @ {distance_name}",
        xaxis_title="Number of facilities (existing + new)",
        yaxis_title="Percentage of population with access",
        plot_bgcolor="white",
        yaxis=dict(range=[0, 100]),
        xaxis=dict(range=[0, len(J_potential) + len(J_existing)]),
        width=1200,
    )
    fig.add_vline(x=len(J_existing), line_width=3, line_dash="dash", line_color="green")
    fig.add_annotation(
        x=len(J_existing),
        y=current_access,
        text="Number of<br>existing facilities",
        showarrow=True,
        arrowhead=1,
        ax=90,
        ay=-30,
    )
    fig.add_hline(y=max_access_possible, line_width=3, line_dash="dash", line_color="green")
    fig.add_annotation(
        x=len(J_existing) + 20,
        y=max_access_possible,
        text="Maximum access possible",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-30,
    )
    fig.show()

    print(f"\n  Completed: {region_name} @ {distance_name}")

# COMMAND ----------

print("\n" + "=" * 60)
print("OPTIMIZATION COMPLETE")
print("=" * 60)
print(f"Processed {len(transform_combinations)} combinations")
print("Next step: Run 04_lgu_metrics.py to compute LGU accessibility")
