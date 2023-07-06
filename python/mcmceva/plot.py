from multiprocessing.sharedctypes import Value
import matplotlib.pyplot as plt
import numpy as np
from mcmceva import stats
from scipy.stats import gumbel_r, genextreme
from mcmceva.stats import GenExtreme


def add_observations(ax, am, nyears, plot_params=None):
    sc_params = dict(color="k", label="Jaarmaxima", s=3)
    if plot_params is not None:
        sc_params.update(plot_params)

    # Plot observations, label top 3 years
    Tam = stats.calc_return_period_am(am)
    ovfreq_am = stats.convert_freq_prob(1.0 / Tam)
    ax.scatter(ovfreq_am, am.to_numpy(), marker=".", **sc_params)
    texts = []
    for i in range(nyears):
        texts.append(ax.text(ovfreq_am[i] / 1.1, am.to_numpy()[i], am.index.year[i], fontsize=7))

    return texts


def calc_percentiles(trace_data, dist, pps, exc_probs=None):

    if isinstance(pps, (tuple, list, np.ndarray)):
        pass
    elif pps == 5:
        pps = [0.025, 0.25, 0.5, 0.75, 0.975]
    elif pps == 3:
        pps = [0.025, 0.5, 0.975]
    else:
        raise TypeError()

    quantiles = np.empty((len(exc_probs), len(pps)))
    # Determine percentiles
    if dist == genextreme or dist == "genextreme":
        # valid = ~np.isnan(trace_data["loc"].data)
        vals = np.nan_to_num(
            GenExtreme(
                loc=trace_data["loc"],
                scale=trace_data["scale"],
                c=trace_data["c"],
            ).ppf(1 - exc_probs),
            nan=np.nan,
            posinf=np.nan,
            neginf=np.nan,
        ).T
        valid = ~np.isnan(vals)
        for i in range(len(exc_probs)):
            quantiles[i] = np.quantile(vals[i][valid[i]], pps)

    elif dist == gumbel_r:
        for i, ip in enumerate(exc_probs):
            vals = gumbel_r(
                loc=trace_data["loc"],
                scale=trace_data["scale"],
            ).ppf(1 - ip)
            quantiles[i] = np.quantile(vals, pps)

    else:
        KeyError(dist)

    return quantiles


def plot_percentiles_filled(ax, exc_probs: np.ndarray, values: np.ndarray, color: str, label=True):
    """Add fill with percentiles tot

    Parameters
    ----------
    ax : _type_
        _description_
    exc_probs : np.ndarray
        Exceendance probabilities
    values : np.ndarray
        Percentile point values [N exc_probs, N percentiles]
        Second dimension must be 3 or 5
    color : str
        Matplotlib plotting color

    Raises
    ------
    NotImplementedError
        _description_
    """

    nps = values.shape[1]
    imid = nps // 2

    ax.plot(exc_probs, values[:, imid], color=color, alpha=0.5, label="Median" if label else None )

    if nps == 5:

        ax.fill(
            np.r_[exc_probs, exc_probs[::-1]],
            np.r_[values[:, imid - 1], values[::-1, imid]],
            alpha=0.1 if nps == 3 else 0.3,
            color=color,
            ec=None,
            label="50% credibility interval" if label else None,
        )

        ax.fill(
            np.r_[exc_probs, exc_probs[::-1]],
            np.r_[values[:, imid], values[::-1, imid + 1]],
            alpha=0.1 if nps == 3 else 0.3,
            color=color,
            ec=None,
        )

    if nps == 3 or nps == 5:
        ax.fill(
            np.r_[exc_probs, exc_probs[::-1]],
            np.r_[values[:, imid - 2], values[::-1, imid - 1]],
            alpha=0.1,
            color=color,
            ec=None,
            label="95% credibility interval" if label else None,
        )

        ax.fill(
            np.r_[exc_probs, exc_probs[::-1]],
            np.r_[values[:, imid + 1], values[::-1, imid + 2]],
            alpha=0.1,
            color=color,
            ec=None,
        )

    if nps != 3 and nps != 5:
        raise ValueError(nps)

    # ax.fill(np.r_[p, p[::-1]], np.r_[percentiles[:, -2], percentiles[::-1, -1]], alpha=0.1, color=color, ec=None)


def plot_percentiles_lines(ax, p, percentiles, color, ls=None):

    if percentiles.shape[1] % 2 != 1:
        raise ValueError("Expected an uneven amount of percentiles")

    ntypes = percentiles.shape[1] // 2 + 1
    lss = ["-", "--", ":", "-."][:ntypes]
    lss = lss[1:][::-1] + lss[:]

    lws = [2, 1, 1, 1][:ntypes]
    lws = lws[1:][::-1] + lws[:]

    for i, row in enumerate(percentiles.T):
        ax.plot(p, row, color=color, label="Mediaan", ls=lss[i], lw=lws[i])


def get_expected_rp(obs, trace_data, dist):

    if dist == gumbel_r:
        vals = 1 - gumbel_r(
            loc=trace_data["loc"].data.ravel(),
            scale=trace_data["scale"].data.ravel(),
        ).cdf(obs)

    elif dist == genextreme:
        vals = 1 - genextreme(
            loc=trace_data["loc"].data.ravel(),
            scale=trace_data["scale"].data.ravel(),
            c=trace_data["c"].data.ravel(),
        ).cdf(obs)

    else:
        KeyError(dist)

    nonnan = vals[~np.isnan(vals) & (vals > 0.0)]

    return nonnan


def get_expected_discharge(exc_prob, trace_data, dist):

    if dist == gumbel_r:
        vals = gumbel_r(
            loc=trace_data["loc"].data.ravel(),
            scale=trace_data["scale"].data.ravel(),
        ).ppf(1 - exc_prob)

    elif dist == genextreme:
        vals = genextreme(
            loc=trace_data["loc"].data.ravel(),
            scale=trace_data["scale"].data.ravel(),
            c=trace_data["c"].data.ravel(),
        ).ppf(1 - exc_prob)

    else:
        KeyError(dist)

    nonnan = vals[~np.isnan(vals) & (vals > 0.0)]

    return nonnan


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


def add_panel_letters(
    axs: np.ndarray,
    transpose: bool = False,
    pos: tuple = (0.05, 0.95),
    case: str = "lower",
    fmt: str = "{}.",
    fontweight="bold",
    fontsize=8,
    ha="left",
    va="top",
):
    """Add letters to subfigure panels

    Args:
        axs (np.ndarray): numpy array with axes
        transpose (bool, optional): Whether to transpose the axes, which gives letters per column first instead of per row first. Defaults to False.
        pos (tuple, optional): Text position. Aligned upper left. Defaults to (0.0, 1.0).
        case (str, optional): Whether to use lowercase or uppercase characters. Defaults to 'lower'.

    """
    if transpose:
        axs = axs.T

    if case == "lower":
        start = 97
    elif case == "upper":
        start = 65
    else:
        raise ValueError(f"Case {case} not understood. Expected 'lower' or 'upper'.")

    for i, ax in enumerate(axs.ravel()):
        ax.annotate(
            xy=pos,
            text=fmt.format(chr(start + i)),
            xycoords="axes fraction",
            va=va,
            ha=ha,
            fontweight=fontweight,
            fontsize=fontsize,
        )
