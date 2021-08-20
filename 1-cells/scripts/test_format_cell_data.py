import numpy as np
import matplotlib.pyplot as plt

from format_cell_data import read_cell_data


def test_read_cell_data():
    """ Test that the data read-in is OK. """
    filename = "data/rsif-data.xlsx"
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
