import h5py
import pytest
import os

import fenics as fe
from petsc4py import PETSc

import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix, rand
from statbz.utils import (build_observation_operator, dolfin_to_csr,
                          read_cell_data, read_csr_matrix_hdf5,
                          write_csr_matrix_hdf5)


def test_build_observation_operator():
    # unit interval, P1 elements
    mesh = fe.UnitIntervalMesh(128)
    V = fe.FunctionSpace(mesh, "P", 1)

    x_obs = np.linspace(0, 1, 30).reshape((30, 1))
    H = build_observation_operator(x_obs, V)

    assert H.shape == (30, 129)
    assert type(H) == csr_matrix
    np.testing.assert_allclose(np.sum(H, axis=1), 1.)

    H_petsc = build_observation_operator(x_obs, V, out="petsc")
    assert type(H_petsc) == PETSc.Mat
    with pytest.raises(ValueError):
        build_observation_operator(x_obs, V, out="fenics")

    # check application of operator
    step = 4
    u = fe.Function(V)
    u_np = np.copy(u.vector()[:])
    u_np[::step] = 2.
    H = build_observation_operator(V.tabulate_dof_coordinates()[::step], V)
    np.testing.assert_allclose(H @ u_np, 2.)

    # check that we raise error if outside the mesh
    x_obs = np.linspace(1, 2, 20).reshape((20, 1))
    with pytest.raises(IndexError):
        H = build_observation_operator(x_obs, V)


def test_build_observation_operator_vector_functionspace():
    # unit interval, P1 elements
    mesh = fe.UnitIntervalMesh(128)
    V = fe.VectorFunctionSpace(mesh, "P", 1, dim=2)

    thin = 5
    subspace_dofs = V.sub(0).dofmap().dofs()
    x = V.tabulate_dof_coordinates()[subspace_dofs[::thin]]
    H = build_observation_operator(x, V)
    H_subspace = build_observation_operator(x, V, sub=1)

    # basic example
    test = fe.interpolate(fe.Expression(("x[0]", "0"), degree=8), V)
    test = np.copy(test.vector()[:])
    desired = test[subspace_dofs[::thin]]
    np.testing.assert_allclose(H @ test, desired)

    # checking my own knowledge
    x = x.flatten()
    test = fe.interpolate(fe.Expression(("sin(x[0])", "0"), degree=8), V)
    test = np.copy(test.vector()[:])
    desired = np.sin(x)
    np.testing.assert_allclose(H @ test, desired)

    # basic example (diff subspace)
    subspace_dofs = V.sub(1).dofmap().dofs()
    test = fe.interpolate(fe.Expression(("0", "x[0]"), degree=8), V)
    test = np.copy(test.vector()[:])
    desired = test[subspace_dofs[::thin]]
    np.testing.assert_allclose(H_subspace @ test, desired)

    # more complex example (diff subspace)
    x = x.flatten()
    test = fe.interpolate(fe.Expression(("-10", "sin(x[0])"), degree=8), V)
    test = np.copy(test.vector()[:])
    desired = np.sin(x)
    np.testing.assert_allclose(H_subspace @ test, desired)


def test_dolfin_to_csr():
    # unit interval, P1 elements
    mesh = fe.UnitIntervalMesh(32)
    V = fe.FunctionSpace(mesh, "CG", 1)

    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)

    # check dolfin matrix
    form = u * v * fe.dx
    M = fe.assemble(form)
    M_csr = dolfin_to_csr(M)
    assert M_csr.shape == (33, 33)
    assert type(M_csr) == csr_matrix

    # check PETScMatrix
    M = fe.PETScMatrix()
    fe.assemble(form, tensor=M)
    M_csr = dolfin_to_csr(M)
    assert M_csr.shape == (33, 33)
    assert type(M_csr) == csr_matrix


@pytest.mark.skip()
def test_read_cell_data():
    """ Test that the data read-in is OK. """
    filename = "data/rsif-formatted.xlsx"
    dat = read_cell_data(filename)

    assert dat.shape == (96, 4)
    np.testing.assert_array_equal(dat.columns.values,
                                  np.array(["t", "u", "v", "total"]))

    x = dat.index.values[:24]

    plt.plot(x, dat["u"][:24], ".", label="u")
    plt.plot(x, dat["v"][:24], ".", label="v")
    plt.legend()
    plt.show()

    plt.plot(x, dat["u"][24:48], ".", label="u")
    plt.plot(x, dat["v"][24:48], ".", label="v")
    plt.legend()
    plt.show()

    plt.plot(x, dat["u"][48:72], ".", label="u")
    plt.plot(x, dat["v"][48:72], ".", label="v")
    plt.legend()
    plt.show()


def test_read_write_csr_matrix_hdf5():
    """ Test reading and writing CSR matrices. """
    tempfile = "temp.h5"
    M = rand(40, 10, format="csr")

    with h5py.File(tempfile, "w") as f:
        write_csr_matrix_hdf5(M, "M", f)

        np.testing.assert_allclose(f["M/data"], M.data)
        np.testing.assert_allclose(f["M/indices"], M.indices)
        np.testing.assert_allclose(f["M/indptr"], M.indptr)

    with h5py.File(tempfile, "r") as f:
        M_disk = read_csr_matrix_hdf5(f, "M", (40, 10))
        np.testing.assert_allclose(M_disk.todense(), M.todense())

    os.remove(tempfile)
