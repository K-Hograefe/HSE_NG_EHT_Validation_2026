# HSE v9 – Hograefe Singularitäts-Entropie
**Unified Entropic Framework: Black Holes and Cosmology**  
Kevin Hograefe – Independent Researcher, Switzerland – ORCID: 0009-0005-0399-2919  
**Release:** 23 November 2025  
DOI: https://doi.org/10.5281/zenodo.17691785  
Paper: [Cosmological_Extension_hse_V9.pdf](Cosmological_Extension_hse_V9.pdf)

## Abstract
The HSE framework replaces the Kerr singularity with an entropic porosity ϕ_BH = 0.632 ± 0.011, yielding a sharp reflective surface at Δr/r_s = 0.277 ± 0.004 and a universal entropy boost ΔS/S = 41.87 ± 4.2 % across 206 GW events (52.34 σ).

HSE v9 extends the same entropic principle to cosmology: a shared parameter ϕ_cosmo = 0.17 ± 0.02 dilutes the effective matter density, yielding H₀ = 72.9 ± 1.0 km s⁻¹ Mpc⁻¹ and resolving the Hubble tension to < 1 σ without additional parameters. Joint Bayesian analysis of GWTC-4 + Planck + SH0ES + DESI BAO strongly favours HSE over ΛCDM (ΔBIC = -27).

All predictions remain falsifiable with NG-EHT 2026 and Euclid/DESI Year-1 data.

**Code & Validation:** https://github.com/K-Hograefe/HSE_NG_EHT_Validation_2026  
**Paper (open access):** https://doi.org/10.5281/zenodo.17691785

## Key Results
- ϕ_BH = 0.632 ± 0.011 → reflective surface at r = 1.277 r_s
- Echo delay in Sgr A* flares ≈ 1.6 ± 0.1 s (NG-EHT 2026)
- ϕ_cosmo = 0.17 ± 0.02 → H₀ = 72.9 ± 1.0 km s⁻¹ Mpc⁻¹
- ΔBIC = -27 vs ΛCDM (decisive evidence)

## Repository Contents
- `HSE_v9.pdf` – final 5-page paper
- `figures/` – corner plots, H(z) comparison, void visualisation
- `code/` – full PyMC/NumPyro MCMC pipeline
- `mcmc_chains/` – complete posterior chains (>47 000 effective samples)
- `environment.yml` – conda environment

## Reproduce the Results
```bash
conda env create -f environment.yml
conda activate hse_v9
python run_joint_analysis.py
