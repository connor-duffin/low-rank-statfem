# Low-rank statFEM for reaction-diffusion models

This repo accompanies our recent paper on low-rank methods for scaling up
statistical finite element methods to high-dimensional problems.

## Getting started

To get started, first `cd` into this directory and setup the Conda environment
through using the provided `environment.yml` file:

```
conda env create -f environment.yml
```

This should install all the requisite packages, into the environment `reacdiff`.
Activate this environment through `conda activate reacdiff`. Then install the
local `statbz` package:

```
pip install -e statbz
```
