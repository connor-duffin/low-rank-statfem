import h5py
import logging

from fenics import *
from petsc4py.PETSc import Mat

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


def thin_simulation(filename, thin=50):
    """ Thin a hdf5 output file from Oregonator simulations. """
    output = h5py.File(filename, "r")
    output_new = h5py.File("../outputs/bz-antispiral.h5", "w")

    for key, value in output.attrs.items():
        output_new.attrs[key] = value
    output_new.attrs["thin"] = thin

    x = output["x"][:]
    t = output["t"][:]

    u = output["u"][::thin, :]
    v = output["v"][::thin, :]
    t = output["t"][::thin]

    output_new.create_dataset("u", data=u)
    output_new.create_dataset("v", data=v)
    output_new.create_dataset("x", data=x)
    output_new.create_dataset("t", data=t)

    output.close()
    output_new.close()


def dolfin_to_csr(A):
    """
    Convert assembled matrix to scipy CSR.
    """
    if type(A) != Mat:
        mat = as_backend_type(A).mat()
    else:
        mat = A
    csr = csr_matrix(mat.getValuesCSR()[::-1], shape=mat.size)
    return csr


def build_observation_operator(x_obs, V, sub=0, out="scipy"):
    """
    Build interpolation matrix from `x_obs` on function space V. This
    assumes that the observations are from the first sub-function of V.

    From the fenics forums.
    """
    nx, dim = x_obs.shape
    mesh = V.mesh()
    coords = mesh.coordinates()
    mesh_cells = mesh.cells()
    bbt = mesh.bounding_box_tree()

    # dofs from first subspace
    if V.num_sub_spaces() > 1:
        dolfin_element = V.sub(sub).dolfin_element()
        dofmap = V.sub(sub).dofmap()
    else:
        dolfin_element = V.dolfin_element()
        dofmap = V.dofmap()

    sdim = dolfin_element.space_dimension()

    v = np.zeros(sdim)
    rows = np.zeros(nx * sdim, dtype='int')
    cols = np.zeros(nx * sdim, dtype='int')
    vals = np.zeros(nx * sdim)

    # loop over all interpolation points
    for k in range(nx):
        x = x_obs[k, :]
        if dim == 1:
            p = Point(x[0])
        elif dim == 2:
            p = Point(x[0], x[1])
        elif dim == 3:
            p = Point(x[0], x[1], x[2])
        else:
            logger.error("no support for higher dims")
            raise ValueError

        # find cell for the point
        cell_id = bbt.compute_first_entity_collision(p)

        # vertex coordinates for the cell
        xvert = coords[mesh_cells[cell_id, :], :]

        # evaluate the basis functions for the cell at x
        v = dolfin_element.evaluate_basis_all(x, xvert, cell_id)

        # set the sparse metadata
        j = np.arange(sdim * k, sdim * (k + 1))
        rows[j] = k
        cols[j] = dofmap.cell_dofs(cell_id)
        vals[j] = v

    ij = np.concatenate((np.array([rows]), np.array([cols])), axis=0)
    H = csr_matrix((vals, ij), shape=(nx, V.dim()))
    if out == "scipy":
        return H
    elif out == "petsc":
        pH = Mat().createAIJ(size=H.shape,
                             csr=(H.indptr, H.indices, H.data))
        return pH
    else:
        raise ValueError(f"out option {out} not recognised")


def write_csr_matrix_hdf5(S, name, h5_file):
    """ Store CSR matrix S in variable `name` in h5_file"""
    data, indices, indptr = S.data, S.indices, S.indptr

    h5_file[f"{name}/data"] = data
    h5_file[f"{name}/indices"] = indices
    h5_file[f"{name}/indptr"] = indptr


def read_csr_matrix_hdf5(h5_file, name, shape):
    """ Read CSR matrix from h5py file. """
    data = h5_file[f"{name}/data"]
    indices = h5_file[f"{name}/indices"]
    indptr = h5_file[f"{name}/indptr"]

    return csr_matrix((data, indices, indptr), shape)
