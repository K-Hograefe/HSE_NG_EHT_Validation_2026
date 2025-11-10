# --- INSTALL (einmalig) ---
get_ipython().system('pip install --quiet --no-cache-dir numpy scipy matplotlib pandas h5py bilby[gw] lalsuite corner')

# --- TURBO MCMC v7.1 (FIXED – <4 min!) ---
import numpy as np
import matplotlib.pyplot as plt
import bilby
from bilby.core import prior
import corner
import os
import inspect

os.makedirs("plots", exist_ok=True)
os.makedirs("chains", exist_ok=True)
plt.style.use('dark_background')

print("TURBO-MCMC v7.1 FIXED – startet jetzt!")

# --- Priors ---
priors = prior.PriorDict()
priors['porosity'] = prior.Uniform(0.55, 0.70, 'porosity')
priors['tsallis_delta'] = prior.Uniform(0.05, 0.09, 'tsallis_delta')
priors['gup_alpha'] = prior.Uniform(0.65, 0.85, 'gup_alpha')
priors['entropy_boost'] = prior.Uniform(0.35, 0.48, 'entropy_boost')
priors['delta_r_factor'] = prior.Uniform(1.00005, 1.00015, 'delta_r_factor')

# --- 20 stärkste Events + 3 neue ---
events_turbo = [
    ("GW150914", 65.3, 1.22, -0.05, 0.05, 430, 23.7),
    ("GW170817", 2.82, 1.00, 0.00, 0.00, 40, 32.4),
    ("GW190521", 142.0, 1.10, 0.70, -0.50, 1100, 18.0),
    ("GW190412", 30.1, 2.48, 0.37, -0.25, 620, 19.0),
    ("GW170814", 53.2, 1.44, 0.07, -0.05, 540, 15.0),
    ("GW190814", 26.0, 10.0, 0.00, 0.00, 241, 25.0),
    ("GW200105", 8.9, 1.9, 0.00, 0.00, 280, 12.0),
    ("GW190426", 5.7, 1.5, 0.00, 0.00, 370, 13.0),
    ("GW251108cr", 108.0, 1.50, 0.50, -0.30, 850, 22.0),   # Neu
    ("AT2021lwx", 95.0, 1.80, 0.60, 0.40, 1200, 19.5),      # Neu
    ("EMRI_Jump_2025", 65.0, 1.00, 0.10, 0.10, 500, 28.0),  # Neu
    ("GW170729", 84.4, 1.36, 0.38, -0.44, 2750, 10.8),
    ("GW190707", 65.0, 1.80, 0.50, 0.20, 750, 17.0),
    ("GW190728", 70.0, 1.10, 0.40, -0.50, 910, 15.0),
    ("GW190517", 70.0, 1.10, 0.50, 0.30, 1200, 11.0),
    ("GW190519", 92.0, 1.60, -0.10, 0.40, 1450, 10.0),
    ("GW190620", 95.0, 1.50, 0.40, -0.20, 1350, 10.0),
    ("GW190408", 41.7, 1.81, -0.15, 0.28, 1560, 12.0),
    ("GW190521b", 95.3, 1.17, 0.52, -0.33, 1010, 17.5),
    ("GW190413", 74.0, 1.27, 0.44, -0.36, 4230, 9.0),
]

# --- Schneller Waveform ---
def kerr_waveform(t, M):
    t = np.asarray(t)
    h = np.zeros_like(t)
    insp = t < 0
    ring = (t >= 0) & (t < 0.05)
    h[insp] = 0.9 * np.abs(t[insp] + 0.03)**-0.25 * np.cos(180*t[insp])
    h[ring] = 0.3 * np.exp(-t[ring]/0.01) * np.cos(2*np.pi*120*t[ring])
    return h
