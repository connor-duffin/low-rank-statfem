import numpy as np

from dolfin import NonlinearVariationalSolver
from statbz.cell import Cell, StatCell, StatCellLowRank

SETTINGS = {"L": 1300, "nx": 200, "dt": 0.1}
PARAMS = {"D": 700, "ku": 0.025, "kv": 0.0725, "sigma_y": 1e-3}


def test_cell():
    cell = Cell(SETTINGS, PARAMS)

    assert cell.n_u_dofs == 201
    assert cell.n_v_dofs == 201
    assert cell.n_dofs == 402

    cell.setup_solve()
    np.testing.assert_allclose(np.unique(cell.u), np.array([0., 0.055]),
                               atol=1e-14)
    assert type(cell.solver) == NonlinearVariationalSolver

    w_vec = np.random.uniform(size=(402, ))
    cell.set_w_from_vector(w_vec)
    np.testing.assert_allclose(cell.w.vector()[:], w_vec)


def test_stat_cell():
    cell = StatCell(1., 100., SETTINGS, PARAMS)

    assert cell.sigma_y == 1e-3

    assert cell.G.shape == (402, 402)
    cell.G[np.diag_indices_from(cell.G)] += 1e-10
    G_chol = np.linalg.cholesky(cell.G)
    np.testing.assert_allclose(np.tril(G_chol), G_chol)

    cell_lr = StatCellLowRank(1., 100., SETTINGS, PARAMS, 8, 8)

    assert cell_lr.L_G.shape == (402, 8)
    assert cell_lr.L_cov.shape == (402, 8)
    assert cell_lr.L_temp.shape == (402, 16)
