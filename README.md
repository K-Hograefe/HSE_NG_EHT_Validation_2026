# HSE Unification: From Porous Black Holes to Cosmological Cohesion  
**Hograefe-Singularity-Entropy Framework (v1.0 – v10.5)**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18451598.svg)](https://doi.org/10.5281/zenodo.18451598)

**Author:** Kevin Hograefe  
**Date:** February 1, 2026  
**License:** CC-BY-4.0 (Creative Commons Attribution 4.0 International)  
**Zenodo DOI:** [10.5281/zenodo.18451598](https://zenodo.org/records/18451598)

## Overview

The Hograefe-Singularity-Entropy (HSE) framework replaces the classical point singularity with a thermodynamically consistent porous spacetime structure. Core parameters are calibrated from 209 gravitational-wave events (GWTC-4):

- Porosity factor: ϕ_BH = 0.632 ± 0.011  
- Tsallis non-extensivity: δ = 0.0682 ± 0.0021  

This yields:
- Finite effective horizon: r_eff ≈ 1.277 r_s  
- Entropy reduction: S_HSE ≈ 6.6% S_BH  
- Resolved information paradox via storage in porous Δr skin (Δr/r_s ≈ 0.277)  
- Modified Eddington limit: λ_Edd,max ≈ 23× standard (stable super-Eddington accretion with corona + jets)  
- Cosmological extension: ϕ_cosmo ≈ 0.174 → H₀ = 72.9 ± 1.0 km/s/Mpc (Hubble tension resolved <1σ)

## Key Results & Validations

- **Black-hole scale**: Echo delays Δt ≈ 1.63 ± 0.09 s (testable with ngEHT 2026)  
- **Super-Eddington accretion**: Consistent with ID830 (z=3.4, λ_Edd,X = 12.8 ± 3.9, α_OX = -1.42 ± 0.07)  
- **Cosmology**: Fits to Planck CMB, DESI BAO, Pantheon+ SNe Ia, SH0ES, TDCOSMO (H₀ = 71.6 ± 1.2 km/s/Mpc after porous lensing)  
- **V-Index**: 0.892 (outperforms ΛCDM by ~66% on aggregated datasets)

## Falsifiable Tests (2026–2027)

- ngEHT echo delay: Δt = 1.63 ± 0.09 s at 230 GHz  
- M87* EVPA rotation: Δα = 2.47° ± 0.08° from B_φ/B_z = ϕ_BH × δ  
- ID830 follow-up: Fe Kα peak ~6.4 keV, T_b > 10¹² K at 43 GHz, declining λ_Edd,X over 5 years  
- Long-term: Double-peaked X-ray/optical lags, QPO triplet, systematic α_OX decline

## Repository Contents

- `main.tex` — Full LaTeX source of the paper (v10.5, February 2026)  
- `figures/` — All plots (9-panel test suite, ID830 validation, time evolution)  
- `code/` — Python scripts (MCMC fits, scaling derivations, QPO simulation, time-lag CCF)  
- `data/` — Aggregated GW, CMB, BAO, SNe Ia, TDCOSMO, COSMOS-Web datasets (processed)  
- `requirements.txt` — Python dependencies (numpy, scipy, matplotlib, emcee, corner)  

## Reproducibility

All results are fully reproducible. Run the main analysis scripts in `code/`:

```bash
pip install -r requirements.txt
python code/run_mcmc_fits.py       # GW + cosmology fits
python code/simulate_id830_evolution.py  # Transitional phase prediction
python code/plot_9panel_suite.py   # Generate main figure
