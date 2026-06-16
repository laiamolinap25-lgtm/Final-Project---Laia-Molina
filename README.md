# Final Project: Neurotransmitter Architecture and Alzheimer's Disease Biomarkers

This repository contains the code developed for my final degree project, focused on the spatial relationship between normative neurotransmitter receptor/transporter density maps and neuroimaging biomarkers associated with Alzheimer's disease.

The project investigates whether the normal neurochemical architecture of the brain is spatially associated with regional patterns of biomarker alteration across the Alzheimer's disease continuum.

## Project overview

Alzheimer's disease does not affect the brain uniformly. Instead, different cortical and subcortical regions show distinct patterns of metabolic, structural, and microstructural alteration. At the same time, the healthy brain has a heterogeneous neurochemical organization, with different regions showing different densities of neurotransmitter receptors and transporters.

This project explores whether these two spatial patterns are related.

In practical terms, the pipeline correlates regional neurotransmitter receptor/transporter density values with regional ADNI-derived neuroimaging biomarkers, including:

* FDG-PET metabolism
* T1w/T2w ratio
* Grey matter volume
* Cortical thickness

The analyses are performed across cortical and, when available, subcortical brain regions.

## Main research question

Do regions with higher normative neurotransmitter receptor or transporter density show specific patterns of alteration in Alzheimer's disease neuroimaging biomarkers?

This approach does not aim to prove causality. Instead, it evaluates whether the baseline neurochemical organization of the healthy brain may help explain regional vulnerability or preservation patterns observed in Alzheimer's disease.


## Main features

* Regional Spearman correlation analysis
* Support for multiple ADNI biomarkers
* Cortical and subcortical analyses when available
* Neurotransmitter system grouping:

  * Dopamine
  * Acetylcholine
  * GABA
  * Serotonin
  * Glutamate
* Baseline and longitudinal biomarker analyses
* FDR correction for multiple comparisons
* Automated generation of correlation results
* Heatmap visualization of association patterns

## Analysis types

The pipeline supports four main analysis types:

```text
baseline_all
baseline_declining
delta_all
delta_declining
```

Where:

* `baseline_all` analyzes baseline biomarker values across all participants.
* `baseline_declining` analyzes baseline biomarker values in clinically declining participants.
* `delta_all` analyzes annualized biomarker change across all participants.
* `delta_declining` analyzes annualized biomarker change in clinically declining participants.

Longitudinal change is computed as:

```text
delta per year = (last timepoint - baseline timepoint) / years
```

## Neurotransmitter systems

The receptor and transporter groups used in the analysis include:

```text
Dopamine:      D1, D2, DAT
Acetylcholine: VAChT, M1, A4B2
GABA:          GABAa-bz, GABAa
Serotonin:     5HT1a, 5HT1b, 5HT2a, 5HT4, 5HT6, 5HTT
Glutamate:     mGluR5, NMDA
```

## Data

This repository does not include restricted clinical, imaging, or ADNI data.

The code expects regional biomarker files and receptor density tables to be available locally. File paths must be configured in `config.py` according to the user's local directory structure.

Sensitive data, including ADNI-derived files, subject-level information, raw imaging data, and private laboratory resources, are intentionally excluded from this repository.

## Installation

Create a Python environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, the main dependencies include:

```bash
pip install pandas numpy scipy statsmodels plotly openpyxl
```

Additional visualization scripts may require:

```bash
pip install nibabel nilearn matplotlib
```

## Usage

To run the full analysis pipeline:

```bash
python run_all.py
```

This script runs all configured combinations of:

* Biomarkers
* Analysis types
* Neurotransmitter systems

The main output file is:

```text
all_correlations.csv
```

This file stores the correlation results, including:

* Biomarker
* Correlation type
* Neurotransmitter system
* Receptor/transporter
* Region type
* Spearman correlation coefficient
* p-value
* FDR-corrected p-value
* Significance after correction

## Output

Depending on the configuration, the pipeline may generate:

* CSV files with correlation results
* Interactive scatter plots
* HTML heatmaps
* Brain receptor density maps
* Exploratory summaries of the strongest associations

Generated outputs are excluded from version control by default.

## Interpretation

A significant spatial correlation indicates that the regional pattern of an Alzheimer's disease biomarker is associated with the normative distribution of a neurotransmitter receptor or transporter.

A positive correlation suggests that regions with higher receptor/transporter density tend to show higher biomarker values.

A negative correlation suggests that regions with higher receptor/transporter density tend to show lower biomarker values.

These associations should be interpreted as spatial relationships, not as evidence of direct causality.

## Limitations

* The neurotransmitter density maps represent normative receptor/transporter distributions and are not patient-specific.
* Correlation analyses do not establish causal mechanisms.
* Results may be influenced by spatial autocorrelation between brain regions.
* ADNI-derived biomarker data must be interpreted within the limitations of the available cohort, preprocessing pipeline, and regional parcellation.
* The analysis depends on the alignment between receptor maps and biomarker parcellations.

## Project context

This repository was developed as part of a final degree project investigating the relationship between neurotransmitter systems and Alzheimer's disease neuroimaging biomarkers using a computational neuroimaging approach.

## Author

Laia Molina

## License

This repository is intended for academic and research purposes. Please contact the author before reusing or extending the code in another project.
