import pandas as pd
from pathlib import Path
import numpy as np
from typing import Callable
from bisect import bisect
from numpy.core.multiarray import interp as compiled_interp


def read_file(file: Path) -> pd.Series:
    discharge = pd.read_csv(file, sep=";", index_col=[0])["Q"]
    discharge.index = pd.to_datetime(discharge.index)
    return discharge


def get_annual_maxima(series: pd.Series) -> pd.Series:
    idx = series.groupby(series.index.year).transform(max) == series
    return series.loc[idx]


def get_monthly_maxima(series: pd.Series) -> pd.DataFrame:
    tmp_series = series + np.linspace(0, 1e-3, len(series))
    gb = tmp_series.groupby([tmp_series.index.year, tmp_series.index.month])
    idx = gb.transform(max) == tmp_series
    mm = gb.max().reset_index(drop=True)
    df = pd.DataFrame(mm)
    df.index = series[idx].index
    df["month"] = df.index.month.astype(int)
    return df


def calc_return_period_am(maxima, a=0.3, b=0.4, n=None):
    """
    Calculate return for annual maxima
    2a + b = 1.0
    b = 1 - 2a

    Parameters
    ----------
    maxima : list or numpy.array
        Annual maxima
    a : float
        Plotting position
    b : float
        Plotting position
    n : int
        Number of samples. If None, length of series is used
    """

    if not (2 * a + b) == 1.0:
        raise ValueError("2a + b != 1.0")

    # Sort values
    if n is None:
        n = len(maxima)

    # Determine order
    k = np.arange(len(maxima)) + 1
    P = (k - a)[::-1] / (n + b)
    T = 1.0 / convert_freq_prob(P, reverse=True)
    # T = 1./P

    return T[np.argsort(np.argsort(maxima))]


def convert_freq_prob(inp, reverse=False):
    """
    Function to convert frequencies to probabilities, or vice versa.

    Parameters
    ----------
    inp : numpy.ndarray
        Input values, can be exceedance frequencies or exceedance probabilities
    reversed : boolean
        If True, convert probabilities to frequencies. (default: False)
    """

    if not reverse:
        # P = 1 - e(-f)
        out = 1 - np.exp(-inp)
    if reverse:
        # f = -log(1 - P)
        out = -np.log(1 - inp)

    return out


class GenExtreme:
    def __init__(self, c, loc, scale):
        self.c = np.atleast_1d(c)
        self.loc = np.atleast_1d(loc)
        self.scale = np.atleast_1d(scale)

    @classmethod
    def fit(cls, obs, weights=None):
        c0 = 0.0
        loc0 = obs.min()
        scale0 = 2 * obs.std()

        def opt(params):
            c, l, s = params
            loglikelihood = -cls(c, l, s).logp(obs, weights=weights)
            return loglikelihood

        # Allows bounds: Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, and trust-constr
        xopt = minimize(
            opt, x0=(c0, loc0, scale0), bounds=((None, None), (None, None), (1.0, None)), method="Nelder-Mead"
        )
        return {"c": xopt.x[0], "loc": xopt.x[1], "scale": xopt.x[2]}

    def ppf(self, quantiles: np.ndarray, out: np.ndarray = None):
        idx = self.c == 0.0
        if out is None:
            x = np.zeros(self.c.shape + np.atleast_1d(quantiles).shape)
        else:
            x = out
        if idx.any():
            x[idx] = self.loc[idx, None] - self.scale[idx, None] * np.log(-np.log(quantiles[None, :]))
        if (~idx).any():
            x[~idx] = self.loc[~idx, None] + self.scale[~idx, None] / -self.c[~idx, None] * (
                (-np.log(quantiles[None, :])) ** (self.c[~idx, None]) - 1
            )
        return x

    def pdf(self, obs):
        z = (obs - self.loc) / self.scale
        valid = (-self.c) * z > -1
        p = np.zeros(len(z), dtype=np.float64)
        #         if abs(self.c) < 1e-300:
        if self.c == 0.0:
            t = np.exp(-z[valid])
        else:
            t = (1 - self.c * z[valid]) ** (1 / self.c)
        #         np.clip(t, np.log(1e-300), np.log(1e300), out=t)
        p[valid] = 1.0 / self.scale * t ** (1 - self.c) * np.exp(-t)

        return p

    def logp(self, x, weights=None):
        p = self.pdf(x)
        if (p == 0).any():
            return -1e12

        logp = np.nan_to_num(np.log(p))
        if weights is not None:
            logp *= weights

        return logp.sum()
