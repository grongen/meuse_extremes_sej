import pickle
import numpy as np
import pandas as pd
import emcee
from mcmceva import stats, likelihood
from common import load_anduryl_project, load_directories
import bisect

# Locations to consider
locations = [
    "Franse Maas, Chooz",
    "Semois, Membre",
    "Lesse, Gendron",
    "Sambre, Salzinnes",
    "Vesdre, Chaudfontaine",
    "Ambleve, Martinrive",
    "Ourthe, Tabreux",
    "Roer, Stah",
    "Geul, Meerssen",
    "Niers, Goch",
]


class MixedMetalog:
    """
    Class to calculate pdf and logpdf from weighted set of metalogs
    The provided weights are normalized
    """

    def __init__(self, metalogs, weights):
        self.metalogs = metalogs
        self.weights = weights / weights.sum()

    def logpdf(self, x):
        return np.log(self.pdf(x))

    def pdf(self, x):
        return np.sum([ml.pdf(x) * w for ml, w in zip(self.metalogs, self.weights)], axis=0)

    def ppf(self, p):
        percentile_point = np.sum([ml.ppf(p) * w for ml, w in zip(self.metalogs, self.weights)], axis=0)
        return percentile_point


def interpol(x1, x2, f1, f2, x):
    return f1 + (x - x1) / (x2 - x1) * (f2 - f1)


def inextrp1d(x: float, xp: np.ndarray, fp: np.ndarray):
    # Determine lower bounds
    intidx = np.minimum(np.maximum(0, np.searchsorted(xp, x) - 1), len(xp) - 2)
    # Determine interpolation fractions
    fracs = (x - xp[intidx]) / (xp[intidx + 1] - xp[intidx])
    # Interpolate (1-frac) * f_low + frac * f_up
    f = (1 - fracs) * fp[intidx] + fp[intidx + 1] * fracs

    return f


class ProxyMetalog:
    """
    Class that interpolates the pdf from the metalog, and creates a
    scipy interp1d function from which the pdf's are approximated faster.
    """

    def __init__(self, xp, pdf_interp, estimates):
        self.xp = xp
        self.pdf_interp = pdf_interp
        self.estimates = estimates
    
    def f_pdf(self, x):
        if np.ndim(x) >= 1:
            return np.maximum(0.0, inextrp1d(x, np.asarray(self.xp), np.asarray(self.pdf_interp)))
        
        iright = bisect.bisect(self.xp, x, lo=1, hi=len(self.xp)-1)
        ileft = iright - 1
        f = interpol(self.xp[ileft], self.xp[iright], self.pdf_interp[ileft], self.pdf_interp[iright], x)
        return max(f, 0)
        
    @classmethod
    def from_empirical(cls, empirical, estimates=None):
        pdf_interp = np.gradient(empirical.fp, empirical.xp)
        return cls(empirical.xp, pdf_interp, estimates)

    @classmethod
    def from_metalog(cls, metalog, eps, n, estimates=None):

        # Create interp1d function
        xp, pdf_interp = cls._discretize(metalog, eps, n)
        return cls(xp, pdf_interp, estimates)

    @staticmethod
    def _discretize(metalog, eps=1e-12, n=100):
        """
        Discretization of the discharges is the union of:
        1. The range (eps, 1-eps) in n steps. This spans mainly the center part
        2. The range (Q(eps), Q(1-eps)) in n steps. This spans mainly the tails
        """
        qs1 = metalog.ppf(np.linspace(eps, 1 - eps, n))
        qs2 = np.linspace(metalog.ppf(eps), metalog.ppf(1 - eps), n)
        xp = np.sort(np.concatenate([qs1[1:-1], qs2]))
        pdf_interp = metalog.pdf(xp)
        return xp, pdf_interp

    def pdf(self, x):
        """Return pdf from interpolation function"""
        return self.f_pdf(x)

    def logpdf(self, x):
        """Return log pdf"""
        p = self.f_pdf(x)
        if np.ndim(x) > 0:
            # Convert to log
            p[p <= 0.0] = np.NINF
            p[p > 0] = np.log(p[p > 0])
            return p
        else:
            return np.log(p) if p > 0 else np.NINF

def get_metalog(project, location, exc_prob, expert):

    # Find relevant question id
    river, gauge = location.split(",")
    river, gauge = river.strip(), gauge.strip()
    T = f"T{int(round(1/exc_prob))}"
    dist = project.main_results.assessments.estimates[expert][f"{gauge}{T}"]

    xp = dist.xp if hasattr(dist, "xp") else dist.metalog.pps
    fp = dist.fp if hasattr(dist, "fp") else dist.metalog.prange

    if hasattr(dist, 'metalog'):
        # Make sure the lower end goes to discharge 0
        if xp.min() > 0:
            xp_low = np.linspace(0, xp.min(), 11)[:-1]
            fp_low = list(map(dist.metalog.pdf, xp_low))
            xp = np.concatenate([xp_low, xp])
            fp = np.concatenate([fp_low, fp])

        xp_high = np.linspace(xp.max(), xp.max()*2, 11)[1:]
        fp_high = list(map(dist.metalog.pdf, xp_high))
        xp = np.concatenate([xp, xp_high])
        fp = np.concatenate([fp, fp_high])

    pdf_interp = np.gradient(fp, xp)

    # return ProxyMetalog(xp=xp, pdf_interp=pdf_interp, estimates=dist.estimates)
    # If the estimate is an Expert estimate with a metalog property
    if hasattr(dist, 'metalog'):
        return ProxyMetalog.from_metalog(dist.metalog, eps=1e-12, n=100, estimates=dist.estimates)
    # If the estimate is an Empirical estimate without a metalog property:
    else:
        return ProxyMetalog.from_empirical(dist, estimates=dist.estimates)

if __name__ == "__main__":

    modes = ["ej", "obs", "both"]
    # modes = ["both"]

    # Load Anduryl project and calculate DM's
    project = load_anduryl_project()
    directories = load_directories()

    # Load discharge measurements
    afvoerpieken = pd.read_csv(directories["meuse_measurements"] / "peak_discharges_hourly.csv", index_col=[0])
    afvoerpieken.index = pd.to_datetime(afvoerpieken.index)

    excprobs = {"ej": [0.1, 0.001], "both": [0.001]}

    for expert in project.experts.get_exp():

        for location in locations:

            # if 'Chooz' not in location:
            #     continue

            mls = {exc_prob: get_metalog(project, location, exc_prob, expert) for exc_prob in [0.1, 0.001]}

            for mode in modes:

                # Output file
                if mode in ["ej", "both"]:
                    outfile = directories["sampledir"] / f"{location}_{mode}_{expert}.pkl"
                else:
                    outfile = directories["sampledir"] / f"{location}_{mode}.pkl"
                if outfile.exists():
                    continue

                # Import time series
                series = afvoerpieken[location]
                # Get annual maxima
                am = stats.get_annual_maxima(series)
                am_arr = am.to_numpy()

                dist = likelihood.GEV

                # Pick correct model
                sampler = likelihood.GEVObsEJSampler(
                    dist=dist, 
                    obs = am_arr if mode != 'ej' else None,
                    ej = {ep: mls[ep] for ep in excprobs[mode]} if mode != 'obs' else None,
                    fej = 1,
                    log_prob_fn=dist.log_posterior_infer,
                    param_variation_scale=[2.5, 2.5, 0.25]
                )

                # Prepare MCMC
                np.seterr(all="ignore", divide="raise")

                nwalkers = 100
                ndim = 3
                pos = sampler.initialize(nwalkers=nwalkers)

                sampler = emcee.EnsembleSampler(
                    nwalkers,
                    ndim,
                    log_prob_fn=sampler.log_prob_fn,
                    args=sampler.args,
                    vectorize=True
                )

                N = 1
                sampler.run_mcmc(pos, 5000 * N, progress=True)

                # If an autocorr error occurs, double the chain length, and discard/thinning
                try:
                    actimes = sampler.get_autocorr_time()
                except emcee.autocorr.AutocorrError:
                    N = 4
                    sampler.run_mcmc(pos, 5000 * N, progress=True)
                    actimes = sampler.get_autocorr_time()

                np.seterr()
                flat_samples = dict(zip(["loc", "scale", "c"], sampler.get_chain(discard=200 * N, thin=24 * N, flat=True).T))
                with open(outfile, "wb") as f:
                    pickle.dump(flat_samples, f)
