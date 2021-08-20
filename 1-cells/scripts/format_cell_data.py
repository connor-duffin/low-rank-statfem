import numpy as np
import pandas as pd


def read_cell_data(filename):
    dat = pd.read_excel(filename,
                        engine="openpyxl",
                        usecols="E:U",
                        skiprows=2)
    dat = dat.loc[:, ~dat.columns.str.contains("^Unnamed")]
    names = dat.columns.values
    names[0:4] = ["x", "G1.0", "S/G2/M.0", "Total.0"]

    dat.columns = pd.Index(names)

    dat = pd.wide_to_long(dat,
                          i="x",
                          j="t",
                          stubnames=["G1", "S/G2/M", "Total"],
                          sep=".")
    dat.columns = pd.Index(["u", "v", "total"])
    dat = dat.astype({"u": np.float64, "v": np.float64, "total": np.float64})
    dat = dat.reset_index("t")

    # set the data scales appropriately
    dat[["u", "v", "total"]] /= (1745 * 54 * 0.004)
    dat["t"] *= 16.

    return dat
