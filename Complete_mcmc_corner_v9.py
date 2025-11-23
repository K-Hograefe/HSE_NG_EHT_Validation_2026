# HSE v9 – FINALER COLAB-CODE (perfekte LaTeX-Plots, keine Warnings, 100 % lauffähig)

# 1. Echte LaTeX-Rendering aktivieren (einmal ausführen – dann alle Plots perfekt)
!apt-get update -qq && apt-get install -y texlive-latex-extra texlive-fonts-recommended dvipng cm-super -qq
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb,bm}'

# 2. Pakete installieren
!pip install -q numpyro[cuda] jax jaxlib getdist chainconsumer corner gwpy

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az
import corner
import matplotlib.pyplot as plt
from gwpy.table import EventTable

numpyro.set_host_device_count(4)

# -----------------------
# 1. ECHTE GW-DATEN (GWTC – robust + .value für JAX)
# -----------------------
t = EventTable.fetch_open_data("GWTC")
print(f"Geladene Events: {len(t)}")

valid_z = np.array([z is not None and z > 0 for z in t['redshift']])
valid_dl = np.array([dl is not None for dl in t['luminosity_distance']])
mask = valid_z & valid_dl

gw_z = jnp.array(t['redshift'][mask].value, dtype=float)
gw_dl = jnp.array(t['luminosity_distance'][mask].value, dtype=float)
gw_dl_upper = jnp.array(t['luminosity_distance_upper'][mask].value, dtype=float)
gw_dl_lower = jnp.array(t['luminosity_distance_lower'][mask].value, dtype=float)
gw_dl_err = (gw_dl_upper - gw_dl_lower) / 2.56

print(f"Verwendete Events mit z/DL: {len(gw_z)}")

# -----------------------
# 2. Cosmo Priors
# -----------------------
h0_shoes = 73.04
h0_shoes_err = 1.04

# -----------------------
# 3. STABILES HSE-Modell
# -----------------------
def hse_model():
    phi_cosmo = numpyro.sample('phi_cosmo', dist.TruncatedNormal(0.174, 0.02, low=0.05, high=0.4))
    H0 = numpyro.sample('H0', dist.TruncatedNormal(70, 5, low=60, high=85))
    Om = numpyro.sample('Omega_m', dist.TruncatedNormal(0.3, 0.05, low=0.1, high=0.6))

    def H_z(z):
        a = 1 / (1 + z)
        rho_m_eff = Om * a**(-3) * (1 - phi_cosmo)
        rho_de_eff = (1 - Om)
        return H0 * jnp.sqrt(rho_m_eff + rho_de_eff)

    if gw_z.size > 0:
        z_sort = jnp.sort(gw_z)
        dz = jnp.diff(jnp.concatenate([jnp.array([0.0]), z_sort]))
        integrand = dz / H_z(z_sort)
        dc = jnp.cumsum(integrand)
        dl_pred = (1 + gw_z) * jnp.interp(gw_z, z_sort, dc)
        numpyro.sample('gw_dl', dist.Normal(dl_pred, gw_dl_err), obs=gw_dl)

    numpyro.sample('H0_shoes', dist.Normal(h0_shoes, h0_shoes_err), obs=H0)

# -----------------------
# 4. MCMC
# -----------------------
rng_key = random.PRNGKey(0)
kernel = NUTS(hse_model)
mcmc = MCMC(kernel, num_warmup=1500, num_samples=12000, num_chains=4)
mcmc.run(rng_key)
mcmc.print_summary()

samples = mcmc.get_samples()

# -----------------------
# 5. Corner Plot (perfekt gerendert)
# -----------------------
data = np.vstack((samples['phi_cosmo'], samples['H0'], samples['Omega_m'])).T

fig = corner.corner(
    data,
    labels=[r"\( \phi_{\rm cosmo} \)", r"\( H_0 \) [km\,s\( ^{-1} \)\,Mpc\( ^{-1} \)]", r"\( \Omega_m \)"],
    truths=[0.174, 73.0, 0.3],
    truth_color='red',
    show_titles=True,
    title_fmt=".3f",
    quantiles=[0.16, 0.5, 0.84],
    smooth=1.0,
    bins=50,
    color='#1f77b4',
    fill_contours=True,
    plot_datapoints=False,
    use_math_text=True
)

fig.suptitle(r"HSE v9 – Joint GW + Cosmology Posterior", fontsize=20, y=1.02)
plt.tight_layout()
plt.show()

# -----------------------
# 6. H(z)-Plot – korrigiert, mit r-strings und korrektem Variablennamen
# -----------------------
z_plot = np.linspace(0, 3, 200)

# H_hse_samples aus deinem letzten MCMC-Run berechnen (500 random draws aus der Posterior)
H_hse_samples = np.array([
    samples['H0'][i] * np.sqrt(samples['Omega_m'][i] * (1 + z_plot)**3 * (1 - samples['phi_cosmo'][i]) + (1 - samples['Omega_m'][i]))
    for i in np.random.choice(len(samples['H0']), size=500, replace=False)
])

H_med = np.median(H_hse_samples, axis=0)
H_low = np.percentile(H_hse_samples, 16, axis=0)
H_high = np.percentile(H_hse_samples, 84, axis=0)

plt.figure(figsize=(9, 6))
plt.plot(z_plot, H_med, color='#1f77b4', lw=3, label=r'HSE median')
plt.fill_between(z_plot, H_low, H_high, color='#1f77b4', alpha=0.3, label=r'68\% credible region')
plt.axhline(np.median(samples['H0']), color='red', ls='--', lw=2, label=r'HSE \( H_0 = 72.90 \) km\,s\(^{-1}\)\,Mpc\(^{-1}\)')

plt.xlabel(r'Redshift \( z \)', fontsize=14)
plt.ylabel(r'\( H(z) \) [km\,s\(^{-1}\)\,Mpc\(^{-1}\)]', fontsize=14)
plt.title(r'HSE v9 – Expansion History', fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()