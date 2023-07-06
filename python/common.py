from typing import Union, Tuple
from pathlib import Path
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde

# Import Anduryl
githubpath = Path("D:/Documents/Github")
sys.path.append(str(githubpath / "anduryl"))
import anduryl
from anduryl.core.main import Project
from anduryl.io.settings import CalculationSettings


def load_directories():
    """Load often used directories"""
    # Directories
    filedir = Path(__file__).parent.resolve()
    datadir = filedir / ".." / "Data"
    sampledir = datadir / "Samples"
    sampledir_emcee = datadir / "Samples (emcee)"
    elicitationdata = datadir / "Elicitation"
    _articledir = filedir / ".." / "Article"
    _dependence_articledir = filedir / ".." / "Dependence article"
    figuredir = _articledir / "Figures"
    figuredir_dependence = _dependence_articledir / "Figures"
    tabledir = _articledir / "Tables"
    tabledir_dependence = _dependence_articledir / "Tables"
    meuse_measurements = datadir / "Measurements" / "Meuse"
    meuse_gis = filedir / ".." / "GIS" / "Derived" / "Meuse"

    directories = {
        "datadir": datadir,
        "sampledir": sampledir,
        "elicitationdata": elicitationdata,
        # "figuredir": figuredir,
        # "tabledir": tabledir,
        "meuse_measurements": meuse_measurements,
        # "meuse_gis": meuse_gis,
        # "figuredir_dependence": figuredir_dependence,
        # "tabledir_dependence": tabledir_dependence,
    }

    for value in directories.values():
        if not value.exists():
            raise OSError(f"Directory {value} does not exist.")

    return directories


_directories = load_directories()


def load_anduryl_project(file=None, dm_settings_file=None, calculate_dms: bool = True) -> Project:

    # Load anduryl project
    if file is None:
        file = _directories["elicitationdata"] / "meuse_discharges_expertsessie.json"
    project = Project()
    project.io.load_json(file)

    if dm_settings_file is None:
        dm_settings_file = _directories["elicitationdata"] / "dm_settings.json"

    if calculate_dms:
        # Note that not specifying alpha results in optimization in case of global or item weights.
        with open(dm_settings_file, "r") as f:
            settings = json.load(f)
            for _, dm_settings in settings.items():
                project.add_results_from_settings(CalculationSettings(**dm_settings))

    return project


def hellinger_distance(Sigma1, Sigma2):
    """Compute the hellinger distance between two matrices"""

    assert Sigma1.shape == Sigma2.shape

    Nvar = Sigma1.shape[0]
    # Mean vector of the first distribution
    m1 = np.zeros((Nvar, 1))

    # Mean vector of the second distribution
    m2 = np.zeros((Nvar, 1))

    # Distance calculation
    # Hellinger distance
    # elements of the distance equation
    a = (np.linalg.det(Sigma1) ** (1 / 4) * np.linalg.det(Sigma2) ** (1 / 4)) / np.linalg.det(
        (1 / 2) * Sigma1 + (1 / 2) * Sigma2
    ) ** (1 / 2)
    b = np.exp(
        -(1 / 8) * (m1 - m2).reshape(1, -1) @ np.linalg.inv((1 / 2) * Sigma1 + (1 / 2) * Sigma2) @ (m1 - m2)
    )
    # equation proper
    D = (1 - (a * b)) ** (1 / 2)

    return D


def d_calibration(sigma1, sigma2):
    return 1 - hellinger_distance(sigma1, sigma2)[0][0]


def conditional_normal(
    mean: np.ndarray, cov: np.ndarray, idx_cond: Union[list, np.ndarray], values_cond: Union[list, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate mean and covariance matrix of the conditional normal distribution

    Parameters
    ----------
    mean : np.ndarray
        mean vector of the multivariate normal
    cov : np.ndarray
        covariance matrix of the multivariate normal
    idx_cond : list
        index of the conditioning nodes
    values_cond : Union[list, np.ndarray]
        values of the conditioning nodes

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        tuple(Mc, Sc) with:
        Mc: mean vector of the conditional multivariate normal on valC
        Sc: covariance matrix of the conditional multivariate normal valC
    """

    # Index of all the remaining (non conditional) variables
    if (np.asarray(idx_cond) < 0).any():
        raise ValueError("Specify all indices from the start of list/array")
    idx_noncond = [i for i in range(mean.shape[0]) if i not in idx_cond]

    # Calculation of the conditional normal distribution:
    M1 = mean[idx_noncond]
    S11 = cov[np.ix_(idx_noncond, idx_noncond)]
    X2 = values_cond
    M2 = mean[idx_cond]
    S22 = cov[np.ix_(idx_cond, idx_cond)]
    S12 = cov[np.ix_(idx_noncond, idx_cond)]
    S21 = cov[np.ix_(idx_cond, idx_noncond)]
    S22_inv = np.linalg.inv(S22)

    # Calculate the covariance matrix
    Sc = S11 - S12 @ S22_inv @ S21
    # Calculate the mean vector
    Mc = M1 + S12 @ S22_inv @ (X2 - M2)
    return Mc, Sc


def inextrp1d(x: float, xp: np.ndarray, fp: np.ndarray):
    # Determine lower bounds
    intidx = np.minimum(np.maximum(0, np.searchsorted(xp, x) - 1), len(xp) - 2)
    # Determine interpolation fractions
    fracs = (x - xp[intidx]) / (xp[intidx + 1] - xp[intidx])
    # Interpolate (1-frac) * f_low + frac * f_up
    f = (1 - fracs) * fp[intidx] + fp[intidx + 1] * fracs

    return f

def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scipy"""
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)



def set_rcparams(wide_margins=True, reset_defaults=False):

    # Reset defaults first
    if reset_defaults:
        plt.rcParams.update(plt.rcParamsDefault)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = "Verdana"
    plt.rcParams["font.size"] = 8

    plt.rcParams["grid.linestyle"] = "-"
    plt.rcParams["grid.alpha"] = 0.40

    plt.rcParams["axes.linewidth"] = 0.5
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["axes.titlesize"] = 8
    plt.rcParams["axes.titleweight"] = "bold"

    plt.rcParams["legend.handletextpad"] = 0.4
    plt.rcParams["legend.fontsize"] = 8
    plt.rcParams["legend.labelspacing"] = 0.2
    plt.rcParams["legend.fancybox"] = False

    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8

    plt.rcParams["figure.dpi"] = 150
    if wide_margins:
        plt.rcParams["figure.figsize"] = [16 / 2.54, 8 / 2.54]
    else:
        plt.rcParams["figure.figsize"] = [13 / 2.54, 8 / 2.54]
