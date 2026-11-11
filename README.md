# spartan-lcl-simulations

Code to generate [LightCurveLynx](https://github.com/lincc-frameworks/lightcurvelynx) simulations for the Spartan SN Ia cosmology project. 
All notebooks here are based on the (excellent) publicly available LightCurveLynx notebooks. Note that this is not production code and that all notebooks are provided as-is for the time being.

## Repository layout

- `jem-notebooks/` — Notebooks for testing out production code before it is promoted to scripts.
  - `jem-rubin_dr1_snia.ipynb` — Rubin DR1 SN Ia simulation and SALT fitting -- the main one of interest.
  - `jem-rubin_dp2.ipynb` — Scratch simulation using the DP2 visit/detector table.
  - `salt_fitting.py` — SALT light-curve fitting helpers used by the notebooks.
- `sims/` — Fiducial and trial simulation outputs (Parquet light curves and SALT fit results).

## Setup

### 1. Get LightCurveLynx

The actual LightCurveLynx package isn't tracked in this repo, so it must be importable from your Python environment. 
Clone it and install it in editable mode:

```bash
git clone https://github.com/lincc-frameworks/lightcurvelynx.git
cd lightcurvelynx
pip install -e .
```

Alternatively, put its `src/` directory on your `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/lightcurvelynx/src:$PYTHONPATH
```

### 2. Install dependencies

Requires Python >= 3.11.

- `lightcurvelynx` (see above; brings in most of the packages below)
- `numpy` (>= 2.0)
- `scipy`
- `pandas`
- `pyarrow` (Parquet I/O)
- `astropy`
- `sncosmo`
- `matplotlib`
- `jupyter` (to run the notebooks)

```bash
pip install numpy scipy pandas pyarrow astropy sncosmo matplotlib jupyter
```
or 

```bash
conda install -c conda-forge numpy scipy pandas pyarrow astropy sncosmo matplotlib jupyter
```

### 3. Survey data

The notebooks expect the following survey files in an `obstable/` directory, which isn't tracked in this repo:

- `obstable/baseline_v5.3.7_10yrs.db` — Rubin OpSim baseline (used by `jem-rubin_dr1_snia.ipynb`)
- `obstable/visit_detector.parquet` — DP2 visit/detector table (used by `jem-rubin_dp2.ipynb`)

Before running a notebook, set `LCL_BASE_DIR` at its top to the directory that contains `obstable/` and `sims/` (or just rename to whatever you want).

Rubin OpSim baselines are available here: [https://s3df.slac.stanford.edu/data/rubin/sim-data/](https://s3df.slac.stanford.edu/data/rubin/sim-data/)
