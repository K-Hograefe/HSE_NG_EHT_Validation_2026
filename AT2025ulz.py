import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import convolve
import matplotlib.pyplot as plt

# -----------------------------
# 1. Daten (Table 2 + Plot-Approx)
# -----------------------------
t_table_detect = np.array([0.13, 34.09, 34.72, 36.16, 36.17, 36.18, 36.18])
mag_table_detect = np.array([21.0, 23.1, 21.3, 21.7, 21.7, 22.1, 22.9])
ref_mag = 21.0
flux_table = 10**(-0.4 * (mag_table_detect - ref_mag))

t_limits = np.array([-4.87, -4.77, -2.84, -2.77, -0.88, -0.84])
lim_flux = 10**(-0.4 * (np.array([21.2, 20.2, 21.0, 20.6, 21.1, 21.0]) - ref_mag)) * 0.5

t_plot = np.array([0, 1, 2, 3, 4, 5, 7, 10, 12, 15, 20, 25, 30])
flux_plot = np.array([1.00, 0.95, 0.88, 0.80, 0.75, 0.70, 0.78, 0.95, 0.92, 0.88, 0.80, 0.72, 0.68])

# Daten kombinieren und sortieren
t_data_raw = np.concatenate([t_plot, t_table_detect])
flux_data_raw = np.concatenate([flux_plot, flux_table])
sort_idx = np.argsort(t_data_raw)
t_data = t_data_raw[sort_idx]
flux_data = flux_data_raw[sort_idx]

# -----------------------------
# 2. Modelle (HSE v10)
# -----------------------------
def standard_model(t, amp_kn, tau_kn, amp_ag, tau_rise, t_peak, amp_sn):
    t_c = np.clip(t, 0, None)
    kilonova = amp_kn * np.exp(-np.clip(t_c / tau_kn, -30, 30))
    afterglow = amp_ag / (1 + np.exp(np.clip(-tau_rise * (t_c - t_peak), -30, 30)))
    supernova = amp_sn * np.exp(-np.clip((t_c - t_peak)**2 / (2 * 12**2), -30, 30))
    return np.clip(kilonova + afterglow + supernova, 0.01, None)

# Fixierte HSE-Konstanten aus der Publikation
delta_fix = 0.0682
phi_fix = 0.632

def hse_model(t, amp_kn, tau_kn, amp_ag, tau_rise, t_peak, amp_sn):
    t_grid = np.arange(-5, 45, 0.1)
    base_grid = standard_model(t_grid, amp_kn, tau_kn, amp_ag, tau_rise, t_peak, amp_sn)
    
    # 1. Smearing
    kernel_time = np.linspace(0, 2.5, 15)
    kernel = np.exp(-kernel_time / (1 + delta_fix))
    kernel /= kernel.sum()
    smeared_grid = convolve(base_grid, kernel, mode='same')
    
    # 2. Delay
    delay_time = 3 * (1 + delta_fix) * phi_fix
    delay_steps = int(delay_time / 0.1)
    
    delayed_grid = np.roll(smeared_grid, delay_steps)
    if delay_steps > 0:
        delayed_grid[:delay_steps] = smeared_grid[:delay_steps] * (1 - phi_fix)
    
    result_grid = delayed_grid + 0.075 * smeared_grid
    return np.interp(t, t_grid, result_grid)

# -----------------------------
# 3. Fits & Statistik
# -----------------------------
bounds = ([0.3, 2.0, 0.3, 0.2, 5.0, 0.2], [1.8, 8.0, 1.8, 1.5, 15.0, 1.2])
p0 = [1.0, 4.0, 0.8, 0.6, 10.0, 0.7]

popt_std, _ = curve_fit(standard_model, t_data, flux_data, p0=p0, bounds=bounds)
popt_hse, _ = curve_fit(hse_model, t_data, flux_data, p0=p0, bounds=bounds)

def get_chi2(model, popt):
    pred = model(t_data, *popt)
    return np.sum((pred - flux_data)**2) / len(t_data)

chi2_std = get_chi2(standard_model, popt_std)
chi2_hse = get_chi2(hse_model, popt_hse)
reduction = ((chi2_std - chi2_hse) / chi2_std) * 100

print(f"--- Statistische Auswertung ---")
print(f"Chi²/dof Standard: {chi2_std:.6f}")
print(f"Chi²/dof HSE v10:  {chi2_hse:.6f}")
print(f"Chi²-Reduktion:    {reduction:.2f}%")

# -----------------------------
# 4. Plot (Warnungsfrei mit r-Strings)
# -----------------------------
t_fine = np.linspace(-2, 40, 500)
plt.figure(figsize=(12, 7))
plt.plot(t_plot, flux_plot, 'go', alpha=0.5, label='Plot Approx')
plt.plot(t_table_detect, flux_table, 'm*', markersize=12, label='Table 2 Detections')

# Korrekte LaTeX-Labels mit Raw-Strings (r'...')
plt.plot(t_fine, standard_model(t_fine, *popt_std), 'r--',
         label=r'Standard Fit ($\chi^2={:.5f}$)'.format(chi2_std))
plt.plot(t_fine, hse_model(t_fine, *popt_hse), 'b-', linewidth=2.5,
         label=r'HSE v10 Fit ($\chi^2={:.5f}$)'.format(chi2_hse))

plt.xlabel('Days since T0')
plt.ylabel('Normalized Flux')
plt.title('AT2025ulz: HSE vs. Standard Model')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()