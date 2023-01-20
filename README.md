# Low-rank statFEM for reaction-diffusion models

This repo accompanies our JCP paper [Low-rank statistical finite elements for scalable model-data synthesis](https://doi.org/10.1016/j.jcp.2022.111261), which can be cited as:
```bibtex
@article{duffin2022lowrank,
  title = {Low-Rank Statistical Finite Elements for Scalable Model-Data Synthesis},
  author = {Duffin, Connor and Cripps, Edward and Stemler, Thomas and Girolami, Mark},
  year = {2022},
  month = aug,
  journal = {Journal of Computational Physics},
  volume = {463},
  issn = {0021-9991},
  doi = {10.1016/j.jcp.2022.111261}
}
```

The paper focuses on low-rank methods for scaling up
statistical finite element methods to high-dimensional problems.
The architecture of this repo consists of the `statbz` package, found in the
`statbz` directory, and three separate directories for each of the case studies
(`1-cells`, `2-spiral-wave`, `3-oscillatory`). Each of these accord to different
models that we study in the paper.

To run the examples, code is contained in a `Makefile` in each subdirectory to
run the examples. Consult the relevant `Makefile` in each subdirectory to see
what needs to be done for each example.

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

If you wanted, you can run the unit tests through doing (from this directory)

```bash
cd statbz
python3 -m pytest tests
```

## Getting the data

For the `cells` example, you need to download the data from:

Simpson, M. J., Baker, R. E., Vittadello, S. T., & Maclaren, O. J. (2020).
Practical parameter identifiability for spatio-temporal models of cell invasion.
Journal of The Royal Society Interface, 17(164), 20200055.
(https://doi.org/10.1098/rsif.2020.0055)

This should then be saved as `1-cells/data/rsif-data.xlsx`, to accord with the
local Makefile.


## Running the examples

With all this set up, you should be able to run the examples from the `Makefile`
in the local directories. For example, for the cells example, running (from this directory)

```bash
cd 1-cells
make outputs/cell-post.h5
```

Will run a statFEM model for the 1D cells example, computing the posterior
distribution of the FEM coefficients using the extended Kalman filter. To run
the low-rank statFEM, for the cells example, with 32 modes, simply do:

```bash
make outputs/cell-post-lr-32-modes.h5
```

The other examples are similar. All directories have a `Makefile` from which all
the examples should be able to be run.

## Generating the figures

Having run the requisite results for each example, the figures can now be
generated. Using the `Makefile`, once again for the cells example, these can be
generated via

```bash
make plots_cell
```

Again, the other examples are similar. Just check the local `Makefile` in each.
