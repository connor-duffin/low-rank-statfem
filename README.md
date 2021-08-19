# Low-rank statFEM for reaction-diffusion models

This repo accompanies our recent paper on low-rank methods for scaling up
statistical finite element methods to high-dimensional problems.

## Getting started

To get started, first `cd` into this directory and setup the Conda environment
through using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
```

This should install all the requisite packages, into the environment `reacdiff`.
Activate this environment through `conda activate reacdiff`. Then install the
local `statbz` package locally:

```bash
pip install -e statbz
```

If you wanted, you can run the unit tests through doing

```bash
cd statbz
python3 -m pytest tests
```

This should now set you up to be able to run the examples from the paper. For
example, to run the 1D cells example, with $k = 32$ modes, you can do:

```bash
cd 1-cells
python3 scripts/run_cell_lr.py
```

