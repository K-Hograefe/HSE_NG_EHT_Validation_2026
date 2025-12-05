import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# HSE Params (kalibriert auf Filament)
L_fil = 15  # Mpc
v_rot = 110  # km/s
N_gal_inner = 14
phi_cosmo = 0.174
delta = 0.0682
beta = 4.0  # Tuning
z_fil = 0.032

# Galaxien-Positionen
pos_gal = np.linspace(0, L_fil, N_gal_inner)

# TTT Baseline (schwach)
lambda_TTT = 0.02 * np.ones(N_gal_inner)

# HSE Enhancement
enhancement = 1 + beta * phi_cosmo * np.exp(-delta * z_fil)
lambda_HSE = lambda_TTT * enhancement

# Random Noise
np.random.seed(42)
sigma_random = 0.1
lambda_total = lambda_HSE + np.random.normal(0, sigma_random, N_gal_inner)

# Alignment Probability & Directions
P_co = 0.5 + 0.5 * erf(lambda_total / sigma_random)
spin_dir = np.array([1 if np.random.rand() < p else -1 for p in P_co])

# Metriken
align_frac = np.mean(spin_dir > 0)
chi2 = np.sum((spin_dir - 0)**2) / N_gal_inner  # Vs. random null
print(f'HSE Alignment Fraction: {align_frac:.3f}')
print(f'Simple χ²/dof vs random: {chi2:.3f}')

# Plot
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(pos_gal, spin_dir, 'o-', color='green', label=f'HSE Spins (frac={align_frac:.2f})')
ax.axhline(0, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Position along Filament [Mpc]')
ax.set_ylabel('Normalized Spin Direction')
ax.set_title('HSE v10: Galaxy Spin Alignment in Rotating Filament')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('HSE_filament_spin_alignment.png', dpi=300)
plt.show()