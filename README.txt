# Dataset
Solomon_Instances
Contains the 56 benchmark instances developed by Marius Solomon.
These are considered the classical benchmark for VRPTW.

Gehring_Homberger_Instances
Contains the large-scale VRPTW instances developed by Gehring and Homberger.
These instances are designed to test the scalability of algorithms.
The data is organized into subfolders based on the number of customers:
    homberger_200_customer_instances
    homberger_400_customer_instances
    homberger_600_customer_instances
    homberger_800_customer_instances
    homberger_1000_customer_instances
The problems are grouped into six types based on customer distribution and time window characteristics:
    R1 & R2: Randomly distributed customers.
    C1 & C2: Clustered customers.
    RC1 & RC2: A mix of random and clustered customers.


# VRPTW Baseline + ML Guidance (Day 1/Day 2)
This project builds a VRPTW baseline solver using OR-Tools and then adds a
machine-learning (ML) module that learns edge compatibility (i -> j) from
multiple baseline solutions. The ML model produces edge scores that can be
used to guide route construction.

## Project Flow
1) Parse Solomon/Gehring-Homberger instance files.
2) Solve VRPTW with OR-Tools (baseline).
3) Generate many baseline solutions (30 runs).
4) Create edge labels from those solutions.
5) Train an XGBoost model to predict edge compatibility.
6) Visualize OR-Tools routes and ML edge predictions in map.ipynb.

## Files
main.py
Runs the OR-Tools solver on one instance, writes CSV results, generates 30
baseline runs, trains XGBoost, and exports ML edge scores for visualization.

func_collection.py
Holds all reusable utilities:
- dataset parsing
- OR-Tools VRPTW model helpers
- ML feature building, labeling, and XGBoost training

map.ipynb
Visualization notebook:
- plots customer locations
- plots OR-Tools routes (initial and final)
- overlays ML-predicted edges
- prints summary stats for OR-only runs and ML data