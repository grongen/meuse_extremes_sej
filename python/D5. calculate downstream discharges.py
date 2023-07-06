from pathlib import Path

import pickle
import numpy as np
from numpy.core.multiarray import interp as compiled_interp
import pandas as pd
from scipy.stats import norm, lognorm
from tqdm.auto import tqdm

from matlatzinca.core.bn import BayesianNetwork

from common import inextrp1d, load_directories, load_anduryl_project
from mcmceva import stats

# # Load project, directories, data, settings

# Load directories and project
directories = load_directories()
project = load_anduryl_project()


# Sample numbers
Nsamples = 10000
Nevents = 10000

# The ARI for which the exceedance frequency curves are sorted
Trepr = 1000

save = True


locations = {
    "Borgharen": [
        "Franse Maas, Chooz",
        #     "Semois, Membre",
        "Lesse, Gendron",
        "Sambre, Salzinnes",
        "Vesdre, Chaudfontaine",
        "Ambleve, Martinrive",
        "Ourthe, Tabreux",
    ]
}
locations["Roermond"] = locations["Borgharen"] + [
    "Roer, Stah",
    "Geul, Meerssen",
]
locations["Gennep"] = locations["Roermond"] + [
    "Niers, Goch",
]

# Gennep and Roermond are based on Venlo (where discharge measurements are available)
obs_lognorm = {
    "Borgharen": lognorm(0.13582158121959995, 0.6830475095289436, 0.30828214337074517),
    # "Borgharen": lognorm(0.00858556532594583, -3.192197517992893, 4.18116047439044),
    "Roermond": lognorm(0.3114752773766201, 0.7025770188153858, 0.23778644703362506),
    "Gennep": lognorm(0.3114752773766201, 0.7025770188153858, 0.23778644703362506),
}

# Load AM Borgharen

# Load discharge measurements
afvoermetingen = pd.read_csv(directories["meuse_measurements"] / "discharge_measurements.csv", index_col=[0])
afvoermetingen.index = pd.to_datetime(afvoermetingen.index)

# Get annual maxima
am_borgharen = stats.get_annual_maxima(
    afvoermetingen["Maas, Borgharen"] + np.linspace(0, 1e-3, len(afvoermetingen))
)

# Load peaks to calculate observed correlations
piekafvoeren = pd.read_csv(directories["meuse_measurements"] / "peak_discharges_hourly.csv", index_col=[0])


# # Define functions
# Define generalized extreme value dist for faster sampling
def genextreme_ppf(
    loc: np.ndarray,
    scale: np.ndarray,
    c: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:

    if quantiles.shape[0] == 1:
        percentile_point = np.full(loc.shape + quantiles[0].shape, np.nan)
    else:
        percentile_point = np.full(quantiles.shape, np.nan)

    shp = [1] * percentile_point.ndim
    shp[0] = -1

    gumbel_idx = c == 0.0

    if gumbel_idx.any():
        percentile_point[gumbel_idx] = loc[gumbel_idx].reshape(shp) - scale[gumbel_idx].reshape(shp) * np.log(
            -np.log(quantiles)
        )

    if (~gumbel_idx).any():
        percentile_point[~gumbel_idx] = loc[~gumbel_idx].reshape(shp) + scale[~gumbel_idx].reshape(shp) / -c[
            ~gumbel_idx
        ].reshape(shp) * ((-np.log(quantiles)) ** (c[~gumbel_idx].reshape(shp)) - 1)

    return percentile_point


def get_trace(mode, location, expert=None) -> Path:
    if mode in ["ej", "both"]:
        outfile = directories["sampledir"] / f"{location}_{mode}_{expert}.pkl"
    else:
        outfile = directories["sampledir"] / f"{location}_{mode}.pkl"
    if not outfile.exists():
        raise OSError(outfile)

    with open(outfile, 'rb') as f:
        trace = pickle.load(f)

    return trace



class FactorCalculator:
    def __init__(self, mode: str, location: str, expert: str = None):
        self.mode = mode
        self.location = location
        self.expert = expert
        self.params = ["s", "loc", "scale"]

        if mode == "ej":
            self.factor_dist = {
                rp: project.assessments.estimates[expert][f"f{location}{rp}"] for rp in ["T10", "T1000"]
            }
            self.single_estimate = self.factor_dist["T10"].estimates == self.factor_dist["T1000"].estimates

        elif mode == "obs":
            self.factor_dist = {rp: obs_lognorm[location] for rp in ["T10", "T1000"]}
            # Determine whether the expert made seperate T10 and T1000 estimates, or not
            self.single_estimate = True

        elif mode == "both":
            self.factor_dist = {
                "T10": obs_lognorm[location],
                "T1000": project.assessments.estimates[expert][f"f{location}T1000"],
            }
            self.single_estimate = False

    def multiply_with_factors(self, discharges: np.ndarray, exc_probs: np.ndarray, seed: int):

        if self.single_estimate:
            factors = self.sample_factors(Nevents=len(discharges), rp="T10", seed=seed)
            discharges *= factors

        else:
            # Assume full dependence between the magnitude of the factors, by using the same seed
            factorsT10 = self.sample_factors(Nevents=len(discharges), rp="T10", seed=seed)
            factorsT1000 = self.sample_factors(Nevents=len(discharges), rp="T1000", seed=seed)

            # Per event, we have a T10 and T1000 factor
            # Depending on the magnitude of the event (the return period), we multiply with a combination of the two factors
            part_T10 = inextrp1d(np.log(exc_probs), np.log([0.001, 0.1, 1.0]), np.array([0, 1, 1]))
            part_T1000 = 1 - part_T10

            factor_interp = factorsT10 * part_T10 + factorsT1000 * part_T1000
            factor_interp = np.maximum(0.0, factor_interp)

            discharges *= factor_interp

    def sample_factors(self, Nevents: int, rp: str, seed: int = None):

        if seed is not None:
            np.random.seed(seed)

        uvalues = np.random.rand(Nevents)
        dist = self.factor_dist[rp]
        # If Anduryl Metalog
        if hasattr(dist, "metalog"):
            factors = compiled_interp(uvalues, dist.metalog.prange, dist.metalog.pps)
        # Elif Anduryl empirical assessment
        elif hasattr(dist, "xp"):
            factors = compiled_interp(uvalues, dist.fp, dist.xp)
        # Else, lognormal dist
        elif hasattr(dist, "ppf"):
            factors = dist.ppf(uvalues)

        return factors


class CorrelationMatrix:
    def __init__(self, node_names: list, R: np.ndarray):
        self._node_names = node_names
        self.R = R

    @classmethod
    def from_data(cls, data, node_names):
        R = data[node_names].corr().to_numpy()
        return cls(node_names, R)

    def draw_mvn_sample(self, size, nodes: list = None):
        if nodes is None:
            return np.random.multivariate_normal(mean=np.zeros(len(self._node_names)), cov=self.R, size=size)
        else:
            order = [self._node_names.index(name) for name in nodes]
            return np.random.multivariate_normal(
                mean=np.zeros(len(nodes)), cov=self.R[np.ix_(order, order)], size=size
            )


# # Calculate correlation matrices for all experts
Rs = {}

for downstream in ["Borgharen", "Roermond", "Gennep"]:
    Rs[downstream] = {}
    # From observations
    node_names = [loc.split(",")[0] for loc in locations[downstream]]
    piekafvoeren.rename(columns=dict(zip(locations[downstream], node_names)), inplace=True)
    npbn = CorrelationMatrix.from_data(piekafvoeren, node_names)
    npbn._node_names[npbn._node_names.index("Franse Maas")] = "French Meuse"

    Rs[downstream]["obs"] = npbn.R

    # From experts
    for expert in project.experts.get_exp(exptype="actual"):

        # Import NPBN
        fp = directories["elicitationdata"] / f"BN_{expert}.json"
        npbn = BayesianNetwork.parse_file(fp)
        #     npbn.change_node_name("French Meuse", "Franse Maas")
        npbn.calculate_correlation_matrix()

        # Some experts might have added hierarchical nodes
        order = [npbn._node_names.index(name) for name in node_names]

        Rs[downstream][expert] = npbn.R[np.ix_(order, order)]

    # From DMs
    for expert in project.experts.get_exp(exptype="dm"):

        dmresults = project.results[expert]

        # Get weights
        weights = dmresults.experts.weights[:]
        weights = weights[~np.isnan(weights)] / np.nansum(weights)

        Rs[downstream][expert] = np.sum(
            [Rs[downstream][exp] * w for w, exp in zip(weights, dmresults.experts.get_exp("actual"))], axis=0
        )

# # Calculate exceedance frequencies
rps = [0.1, 0.01, 0.001]
rpstr = ["T10", "T100", "T1000"]

# As the number of samples is equal, the (empirical) exceedance probabilities are as well
# We generate them on beforehand, and pick the relevant ones for plotting
all_exc_probs = stats.convert_freq_prob(1.0 / stats.calc_return_period_am(np.arange(Nevents)))
int_exc_prob = np.sort(
    np.concatenate([rps, np.logspace(np.log10(all_exc_probs[0]), np.log10(all_exc_probs[-1]), 200, base=10)])
)[::-1]
# Find nearest
idx = np.unique([np.argmin(np.absolute(all_exc_probs - ep)) for ep in int_exc_prob])
exc_probs = all_exc_probs[idx]

qTidx = {s: np.argmin(np.absolute(exc_probs - rp)) for s, rp in zip(rpstr, rps)}

modes = ["ej", "obs", "both"]

# For each downstream location
for downstream, subset in locations.items():

    q1000res = {}

    # For the different modes: EJ, observations, and both
    for mode in modes:

        # For the different experts and decision makers
        for expert in project.experts.ids[:]:
            
            # Determine the path of the outfile
            outfile = (
                directories["sampledir"]
                / downstream
                / f"{downstream}_{mode}{'' if mode == 'obs' else '_'+expert}.csv"
            )

            if outfile.exists():
                continue

            # Create a correlation model from the weighted correlation matrices
            node_names = subset[:]  # [loc.split(",")[0] for loc in subset]
            cm = CorrelationMatrix(
                R=Rs[downstream]["obs" if mode == "obs" else expert], node_names=node_names
            )
            # cm.R[cm.R!=1.0] = 0.0

            # Load trace for each location. Calculate the Trepr discharge, and sort the traces
            # based on that discharge
            traces = []
            length = int(1e10)
            for location in subset:

                # Import GEV fits
                trace = get_trace(mode, location, expert)
                trace_size = trace['loc'].size
                
                # Sort the trace such that it represents discharges in ascending order
                # Note that we have to pick a representative ARI for this
                Qrepr = genextreme_ppf(**trace, quantiles=np.array([1 - 1.0 / Trepr]))
                isfinite = np.isfinite(Qrepr)
                order = np.argsort(Qrepr[isfinite]).squeeze()
                length = min(length, trace_size)
                traces.append(np.stack(list(trace.values()))[:, isfinite][:, order])

            trace_size = length

            # Stack the traces for all locations
            traces = np.stack([trace[:, :length] for trace in traces])

            # Draw a random sample for uncertainties
            np.random.seed(0)
            order = [cm._node_names.index(name) for name in node_names]
            sample_unc = cm.draw_mvn_sample(Nsamples)[:, order]

            # Convert to integers to select GEV fits such that the uncertainties are correlated as well between tributaries
            sample_unc = np.round(norm._cdf(sample_unc) * (trace_size - 1)).astype(int)
            # sample_unc[:] = 10000

            # Draw sample for factor (8000 lognormal parameter combinations)
            factor = FactorCalculator(mode, downstream, expert)

            # Initialize array
            qs = np.zeros((Nsamples, len(exc_probs)))

            # Loop through all samples (each of which results in a single exceedance frequency curve)
            for i in tqdm(range(Nsamples)):

                # Draw a random sample for discharges
                sample_q = norm._cdf(cm.draw_mvn_sample(Nevents, nodes=subset)[:, order].T)

                # Use uncertainty sample to pick selections of GEV-fits
                # One GEV param combination in selected per tributary
                gev_params = traces[np.arange(len(subset)), :, sample_unc[i]]

                # Use discharge sample to generate a sequence of events
                # Transform uniform discharge sample to discharge values
                qfull = genextreme_ppf(*gev_params.T, quantiles=sample_q)

                # Sum to get downstream discharge and multiply with factor
                q = np.sort(qfull.sum(axis=0))
                factor.multiply_with_factors(discharges=q, seed=i, exc_probs=all_exc_probs)

                # Get discharges at specific exceedance frequencies
                qs[i] = np.sort(q)[idx]

            percentiles = [2.5, 5.0, 25, 50, 75, 95.0, 97.5]
            df = pd.DataFrame(
                data=np.percentile(qs, percentiles, axis=0).T, index=exc_probs, columns=percentiles
            )
            if save:
                if not outfile.parent.exists():
                    outfile.parent.mkdir()
                df.to_csv(outfile)

                # Save specific return periods
                for rps, rp in zip(rpstr, rps):
                    npyfile = (
                        directories["sampledir"]
                        / downstream
                        / rps
                        / f"{mode}{'' if mode == 'obs' else '_'+expert}.npy"
                    )
                    if not npyfile.parent.exists():
                        npyfile.parent.mkdir()

                    np.save(npyfile, qs[:, qTidx[rps]])

            # Observations are not expert dependent, so break after first 'expert'
            if mode == "obs":
                break
