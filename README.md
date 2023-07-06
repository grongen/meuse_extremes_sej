# Matlatzinca
This repository contains the Python routines and data used for the research that is currently under review: 

## Usage
This code is published for the reproducability and transparency of the aforementioned research. The following python script or notebooks in the "python" directory are used to calculate tributary and downstream discharges:
- D1. Fitting GEV with EJ.py: Fits the tributary generalized extreme value distribution to the observations and expert judgments. 
- D2. ANDURYL.ipynb: Processes the expert judgment results with Cooke's method (create table and figures)
- D3. Figures tributary estimates.ipynb: Generate figures for tributary GEVs.
- D4. fitting factor between upstream sum and downstream discharges.py: Determines the factor between the sum of tributary peak discharges and the downstream peak discharges. Fits a lognormal distribution to the observed (historical) discharges.
- D5. calculate downstream discharges.py: Calculated downstream discharges based on tributary GEVs, factors, and expert correlation estimates.
- D6. Plot exceedance frequencies downstream and condititional PDFs.ipynb: Makes figures for the results from the D5 script.

The code documentation is limited to inline documentation, feel free to reach out if questions arise.

## Python version
Used (and tested) with Python 3.10. Uses a range of Python modules that can be installed using conda or pip:
- numpy, matplotlib, pandas, geopandas, rasterio, pysheds, emcee, and these modules' dependencies.
- anduryl is used for processing expert judgments, which can be installed with: pip install anduryl.
- matlatzinca is used for processing the dependencies. Can be used by making a check-out from this repository: https://github.com/grongen/Matlatzinca/

## License
[GNU](https://choosealicense.com/licenses/gpl-3.0/)
