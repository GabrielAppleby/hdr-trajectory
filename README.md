# Trajectory Analysis

This project is currently broken up into processing scripts, a small debugging dash app, and a config file.

The requirements to run these scripts are found in `requirements.txt`

## Processing scripts

- `parse.py`
  -  Parses a trajectory simulation given a file name.
  - Outputs N X M X K X 3 array
    - N is the number of trajectories
    - M is the number of time steps
    - K is the number of atoms
    - 3 is the xyz positions of the atoms
- `convert.py`
  - Converts the parsed simulation to pairwise distances given a file name.
  - Outputs N X M X J array
    - N is the number of trajectories
    - M is the number of time steps
    - K is the number of pairwise atom distances
- `cluster.py`
  - Clusters and Projects the pairwise distances of **last time step** (this will change in future) given a file name.
  - Outputs a number of HDBSCAN and UMAPs using various hyperparameters.
- `analysis.py`
  - Does all the above given a file name

## Config
`config.py` please see file for details.

## Debug app
`dash_plot.py`
Plots a debugging view to look at different hyperparameter combinations with HDBSCAN and UMAP.
