#!/usr/bin/env python3
"""
Bayesian Latent-State Production Model for GLF Essay
Corrected from PDF extraction.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import norm, beta as beta_dist

rng = np.random.default_rng(20260521)

# ------------------------------------------------------------
# 1. National data
#    columns: year, grain_output, procurement, retained_pc,
#             rural_labor, area_sown, draft_animals,
#             machinery_hp, fertilizer
# ------------------------------------------------------------
data = [
    (1952, 164, 33, 260, 173, 124, 76, 0.3, 0.08),
    (1953, 167, 47, 242, 177, 127, 81, 0.4, 0.12),
    (1954, 170, 51, 228, 182, 129, 85, 0.5, 0.16),
    (1955, 184, 48, 256, 186, 130, 88, 0.8, 0.24),
    (1956, 193, 40, 284, 185, 136, 88, 1.1, 0.33),
    (1957, 195, 46, 273, 193, 134, 84, 1.7, 0.37),
    (1958, 200, 52, 268, 155, 128, 78, 2.4, 0.55),
    (1959, 170, 64, 193, 163, 116, 79, 3.4, 0.54),
    (1960, 143, 47, 182, 170, 122, 73, 5.0, 0.66),
    (1961, 148, 37, 209, 197, 121, 69, 7.1, 0.45),
    (1962, 160, 32, 229, 213, 122, 70, 10,  0.63),
    (1963, 170, 37, 231, 220, 121, 75, 12,  1.0),
    (1964, 188, 40, 256, 228, 122, 79, 13,  1.3),
    (1965, 195, 39, 261, 234, 120, 84, 15,  1.9),
    (1966, 214, 41, 282, 243, 121, 87, 17,  2.3),
    (1967, 218, 41, 281, 252, 119, 90, 20,  2.4),
    (1968, 209, 40, 261, 261, 116, 92, 22,  2.7),
    (1969, 211, 38, 259, 271, 118, 92, 26,  3.1),
    (1970, 240, 46, 282, 278, 119, 94, 29,  3.4),
    (1971, 250, 44, 293, 284, 121, 95, 38,  3.8),
    (1972, 241, 39, 298, 283, 121, 96, 50,  4.3),
    (1973, 265, 48, 293, 289, 121, 97, 65,  4.8),
    (1974, 275, 47, 303, 292, 121, 98, 81,  5.4),
    (1975, 285, 53, 304, 295, 121, 97, 102, 6.0),
    (1976, 286, 49, 306, 294, 121, 95, 117, 6.8),
    (1977, 283, 48, 300, 293, 120, 94, 140, 7.6),
]

cols = ['year', 'grain_output', 'procurement', 'retained_pc',
        'rural_labor', 'area_sown', 'draft_animals',
        'machinery_hp', 'fertilizer']
df = pd.DataFrame(data, columns=cols)
T = len(df)

# ------------------------------------------------------------
# 2. Derived policy-stress variables
# ------------------------------------------------------------
df['proc_rate'] = df.procurement / df.grain_output
df['glf'] = ((df.year >= 1958) & (df.year <= 1961)).astype(float)

base_proc = df.loc[df.year <= 1957, 'proc_rate'].mean()
base_labor = float(df.loc[df.year == 1957, 'rural_labor'].iloc[0])
base_land_worker = float(
    df.loc[df.year == 1957, 'area_sown'].iloc[0] /
    df.loc[df.year == 1957, 'rural_labor'].iloc[0]
)

df['proc_excess'] = np.maximum(0, df.proc_rate - base_proc)
df['labor_diversion'] = df.glf * np.maximum(
    0, (base_labor - df.rural_labor) / base_labor)
df['land_intensity'] = df.glf * np.maximum(
    0, (df.area_sown / df.rural_labor) / base_land_worker - 1)

def standardize(v):
    v = np.asarray(v, dtype=float)
    return (v - v.mean()) / v.std()

stress_base = np.vstack([
    df.glf.to_numpy(float),
    standardize(df.proc_excess),
    standardize(df.labor_diversion),
    standardize(df.land_intensity)
]).T

stress_names = ['GLF dummy', 'excess procurement',
                'labor diversion', 'land intensity']

# Production inputs: standardized logs
for c in ['rural_labor', 'area_sown', 'draft_animals',
          'machinery_hp', 'fertilizer']:
    df['log_' + c] = np.log(df[c])

input_cols = ['log_rural_labor', 'log_area_sown',
              'log_draft_animals', 'log_machinery_hp',
              'log_fertilizer']
X = df[input_cols].to_numpy(float)
Xz = (X - X.mean(axis=0)) / X.std(axis=0)

y_log = np.log(df.grain_output.to_numpy(float))
y_mean, y_sd = y_log.mean(), y_log.std()
yz = (y_log - y_mean) / y_sd

# ------------------------------------------------------------
# 3. Bayesian latent ecological-vulnerability production model
# ------------------------------------------------------------
pdim = 1 + 5 + 4 + 1 + 1 + 1  # 13 parameters

def unpack(theta):
    i = 0
    alpha = theta[i]; i += 1
    beta = np.exp(theta[i:i+5]); i += 5
    load = np.exp(theta[i:i+4]); i += 4
    rho = expit(theta[i]); i += 1
    penalty = np.exp(theta[i]); i += 1
    sigma = np.exp(theta[i]); i += 1
    return alpha, beta, load, rho, penalty, sigma

def raw_soil(load, rho, stress_matrix):
    stress = stress_matrix @ load
    D = np.zeros(T)
    for t in range(T):
        D[t] = stress[t] + (rho * D[t-1] if t > 0 else 0.0)
    return D

def soil_index(load, rho, stress_matrix=stress_base, reference=None):
    D = raw_soil(load, rho, stress_matrix)
    if reference is None:
        reference = raw_soil(load, rho, stress_base)
    return (D - reference.mean()) / reference.std()

def log_prior(theta):
    alpha, beta, load, rho, penalty, sigma = unpack(theta)
    lp = norm.logpdf(alpha, 0, 1)
    lp += (np.log(2) + norm.logpdf(beta, 0, 0.45) + np.log(beta)).sum()
    lp += (np.log(2) + norm.logpdf(load, 0, 0.60) + np.log(load)).sum()
    lp += (beta_dist.logpdf(rho, 6, 2) + np.log(rho)
           + np.log1p(-rho))
    lp += (np.log(2) + norm.logpdf(penalty, 0, 0.50) + np.log(penalty))
    lp += np.log(2) - 2 * sigma + np.log(sigma)
    return lp

def log_likelihood(theta):
    alpha, beta, load, rho, penalty, sigma = unpack(theta)
    D = soil_index(load, rho)
    mu = alpha + Xz @ beta - penalty * D
    return norm.logpdf(yz, mu, sigma).sum()

def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

def objective(theta):
    val = log_posterior(theta)
    return 1e50 if not np.isfinite(val) else -val

# ------------------------------------------------------------
# 4. Optimization
# ------------------------------------------------------------
init = np.zeros(pdim)
init[1:6]  = np.log([0.10, 0.15, 0.05, 0.15, 0.05])
init[6:10] = np.log([0.50, 0.80, 0.25, 0.15])
init[10]   = logit(0.85)
init[11]   = np.log(0.30)
init[12]   = np.log(0.25)

res = minimize(objective, init, method='BFGS',
               options={'maxiter': 20000, 'gtol': 1e-6})
print('Optimization success:', res.success, '|', res.message)
print('Negative log posterior at MAP:', round(res.fun, 3))

# ------------------------------------------------------------
# 5. Laplace posterior simulation
# ------------------------------------------------------------
map_theta = res.x
cov = np.asarray(res.hess_inv)
cov = (cov + cov.T) / 2
vals, vecs = np.linalg.eigh(cov)
vals = np.clip(vals, 1e-8, 10.0)
cov = vecs @ np.diag(vals) @ vecs.T

# In production, use more draws (e.g., 20000)
n_draws = 2000
samples = rng.multivariate_normal(map_theta, cov, size=n_draws)

# Transform posterior draws
parts = [unpack(th) for th in samples]
alpha_s   = np.array([p[0] for p in parts])
beta_s    = np.array([p[1] for p in parts])
load_s    = np.array([p[2] for p in parts])
rho_s     = np.array([p[3] for p in parts])
penalty_s = np.array([p[4] for p in parts])
sigma_s   = np.array([p[5] for p in parts])

def row(name, values):
    q = np.quantile(values, [0.05, 0.50, 0.95])
    return {'parameter': name,
            'mean': values.mean(),
            'q05': q[0], 'median': q[1], 'q95': q[2]}

rows = [row('soil_output_penalty', penalty_s),
        row('soil_persistence_rho', rho_s),
        row('sigma', sigma_s)]
for j, name in enumerate(stress_names):
    rows.append(row('stress_loading_' + name, load_s[:, j]))
for j, name in enumerate(input_cols):
    rows.append(row('input_elasticity_' + name, beta_s[:, j]))

summary = pd.DataFrame(rows)
print('\nPosterior summary (standardized log-output units; 90% interval):')
print(summary.to_string(
    index=False,
    formatters={'mean': '{:.3f}'.format, 'q05': '{:.3f}'.format,
                'median': '{:.3f}'.format, 'q95': '{:.3f}'.format}
))

# ------------------------------------------------------------
# 6. Latent vulnerability table
# ------------------------------------------------------------
D_samples = np.array([
    soil_index(load_s[i], rho_s[i]) for i in range(len(samples))
])
latent = pd.DataFrame({
    'year': df.year,
    'soil_index_mean': D_samples.mean(axis=0),
    'q05': np.quantile(D_samples, 0.05, axis=0),
    'q95': np.quantile(D_samples, 0.95, axis=0)
})
print('\nLatent ecological vulnerability index (z-score; 90% interval):')
print(latent.to_string(
    index=False,
    formatters={'soil_index_mean': '{:.2f}'.format,
                'q05': '{:.2f}'.format, 'q95': '{:.2f}'.format}
))

# ------------------------------------------------------------
# 7. Counterfactual prediction
# ------------------------------------------------------------
def predict(theta, glf_vec, proc_excess_vec,
            labor_div_vec, land_intensity_vec):
    alpha, beta, load, rho, penalty, sigma = unpack(theta)
    proc_z = (proc_excess_vec - df.proc_excess.mean()) / df.proc_excess.std()
    labor_z = (labor_div_vec - df.labor_diversion.mean()) / df.labor_diversion.std()
    land_z = (land_intensity_vec - df.land_intensity.mean()) / df.land_intensity.std()
    stress_cf = np.vstack([glf_vec, proc_z, labor_z, land_z]).T
    D = soil_index(load, rho, stress_cf,
                   reference=raw_soil(load, rho, stress_base))
    mu = alpha + Xz @ beta - penalty * D
    return np.exp(mu * y_sd + y_mean)

obs_pred = predict(map_theta,
                   df.glf.to_numpy(float),
                   df.proc_excess.to_numpy(float),
                   df.labor_diversion.to_numpy(float),
                   df.land_intensity.to_numpy(float))

glf_cf = df.glf.to_numpy(float).copy()
glf_cf[df.glf == 1] = 0
proc_cf = df.proc_excess.to_numpy(float).copy()
proc_cf[df.glf == 1] = 0
labor_cf = df.labor_diversion.to_numpy(float).copy()
labor_cf[df.glf == 1] = 0
land_cf = df.land_intensity.to_numpy(float).copy()
land_cf[df.glf == 1] = 0

cf_pred = predict(map_theta, glf_cf, proc_cf, labor_cf, land_cf)

cf = pd.DataFrame({
    'year': df.year,
    'observed_output': df.grain_output,
    'model_fit': obs_pred,
    'no_GLF_policy_counterfactual': cf_pred,
    'policy_gap_million_tons': cf_pred - df.grain_output
})

print('\nCounterfactual policy-removal scenario, MAP prediction:')
print(cf[(cf.year >= 1958) & (cf.year <= 1962)].to_string(
    index=False,
    formatters={
        'model_fit': '{:.1f}'.format,
        'no_GLF_policy_counterfactual': '{:.1f}'.format,
        'policy_gap_million_tons': '{:.1f}'.format
    }
))

# ------------------------------------------------------------
# 8. Posterior interval for national gap
# ------------------------------------------------------------
skip = max(1, len(samples) // 20)  # adapt to n_draws
avg_gaps = []
for th in samples[::skip]:
    p = predict(th, glf_cf, proc_cf, labor_cf, land_cf)
    actual = df.loc[(df.year >= 1959) & (df.year <= 1961),
                    'grain_output'].to_numpy()
    avg_gaps.append(
        np.mean(p[(df.year >= 1959) & (df.year <= 1961)] - actual)
    )
avg_gaps = np.array(avg_gaps)
print(
    '\nAverage 1959-61 national policy gap '
    '(posterior, million tons/year): '
    'mean={:.1f}, 90% interval [{:.1f}, {:.1f}]'.format(
        avg_gaps.mean(), *np.quantile(avg_gaps, [0.05, 0.95])
    )
)

# ------------------------------------------------------------
# 9. Sichuan sensitivity module
# ------------------------------------------------------------
scenarios = []
for lam in [1.0, 1.5, 2.0, 2.5]:
    vals = avg_gaps * lam
    scenarios.append({
        'lambda': lam,
        'mean_gap_index': vals.mean(),
        'q05': np.quantile(vals, 0.05),
        'q95': np.quantile(vals, 0.95)
    })

print('\nSichuan multiplier sensitivity:')
print(pd.DataFrame(scenarios).to_string(
    index=False,
    formatters={
        'lambda': '{:.1f}'.format,
        'mean_gap_index': '{:.1f}'.format,
        'q05': '{:.1f}'.format,
        'q95': '{:.1f}'.format
    }
))

# Keep window open when double-clicked
input('\nPress Enter to exit...')
