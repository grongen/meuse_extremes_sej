import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common import load_anduryl_project, load_directories

np.seterr("ignore")

# Load project and directories
project = load_anduryl_project()
directories = load_directories()

# Plot historgrams and datafit
fig, axs = plt.subplots(ncols=2, figsize=(8, 5))

factoren = {}

for ax, location in zip(axs, ["Borgharen", "Venlo"]):

    # Load factors at Borgharen from data
    piekafvoeren = pd.read_csv(
        directories["meuse_measurements"] / f"peak_discharges_hourly_{location}.csv",
        index_col=[0],
    )
    locs1 = [f"Maas, {location}"]
    locs2 = [
        "Vesdre, Chaudfontaine",
        "Franse Maas, Chooz",
        "Lesse, Gendron",
        "Ambleve, Martinrive",
        "Ourthe, Tabreux",
        "Sambre, Salzinnes",
    ] 

    if location == "Venlo":
        locs2 += ["Roer, Stah", "Geul, Meerssen"]

    selectie = piekafvoeren[locs1 + locs2].dropna()
    factoren[location] = selectie[locs1].sum(axis=1) / selectie[locs2].sum(axis=1)
    factoren[location] = factoren[location][factoren[location] < 1.1]

    ax.hist(factoren[location], range=(0.8, 1.4), bins=20)
    xrange = np.linspace(factoren[location].min() - 0.1, factoren[location].max() + 0.1, 100)
    ax.plot(xrange, stats.norm.pdf(xrange, *stats.norm.fit(factoren[location])))
    ax.plot(xrange, stats.lognorm.pdf(xrange, *stats.lognorm.fit(factoren[location])))
    print(stats.lognorm.fit(1./factoren[location]))

plt.show()
