# Low-Voltage Grid Visibility with Graph Learning

This repository contains code for low-voltage distribution grid visibility using strategic metering, graph analytics, and graph neural networks. The workflow builds MV/LV node and topology representations, constructs data-driven adjacency matrices, selects representative meters through community analysis, trains GraphSAGE/GCN models, and evaluates voltage-estimation performance.

Raw grid, voltage, topology, and utility datasets are not included because of data privacy and usage restrictions.

## Main Scripts

- `extract_lv_topology.py` - extracts LV topology, MV/LV transformer links, and phase-load information from source files.
- `build_mvlv_node_lists.py` - builds phase-wise MV/LV node lists.
- `build_mvlv_adjacency.py` - builds phase-wise MV/LV physical adjacency matrices.
- `build_phase_voltage_matrices.py` - builds phase-wise voltage matrices aligned with the MV/LV node lists.
- `build_mi_adjacency.py` - builds data-driven mutual-information adjacency weights.
- `community_meter_selection.py` - performs Louvain/Leiden community detection and meter/captain selection.
- `train_graphsage_voltage_estimation.py` - trains the main weighted GraphSAGE voltage-estimation model.
- `evaluate_graphsage_testset.py` - evaluates GraphSAGE predictions on the test set.
- `calibrate_predictions_xgboost.py` - applies optional XGBoost-based prediction calibration.
- `train_graphsage_baseline.py` - trains a GraphSAGE baseline model.
- `train_gcn_baseline.py` - trains a GCN baseline model.
- `evaluate_gcn_testset.py` - evaluates GCN baseline predictions.
- `plot_model_comparison.py` and `plot_presentation_figures.py` - generate comparison and presentation figures from saved outputs.

## Data Policy

The following files are intentionally excluded:

- raw utility or grid datasets
- voltage CSV files
- topology CSV files
- MATLAB files
- NumPy arrays
- trained model checkpoints
- generated figures and logs

To run the scripts, place approved local data files in the expected paths described inside each script.
