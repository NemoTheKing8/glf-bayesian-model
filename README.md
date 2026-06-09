[README.md](https://github.com/user-attachments/files/28745690/README.md)
[DISCLAIMER]
 An AI assistant was used in the preparation of this repository:
 documentation (README), code comments, and grammar revision. All research
 design, model specification, data analysis are the
 author's original work.

Bayesian Latent-State Model of Ecological Vulnerability in Agricultural Production

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Replication and extension of agricultural production analysis under policy stress, China 1952–1977.

Overview

This project replicates the empirical framework of Houser, Sands & Xiao (2009)
in my junior year and extends it with a Bayesian latent ecological vulnerability model. The core
question is how do policy shocks degrade the ecological basis of agricultural
production, and how long does that damage persist after the policy ends? Especially challenging  is lack of environmental data, like SOM levels. Therefore, a Bayesian latent model is necessarily in this case.

The model treats soil degradation as an autoregressive latent variable — a
quantity that was never systematically measured during the study period, but
whose effects can be inferred from observed production outcomes.

Approach

Latent-state production function: $S_t = \rho \cdot S_{t-1} + \omega' \cdot P_t$, where $P_t$ is a vector of policy-stress variables and $\rho$ captures ecological persistence
Inference: MAP estimation via BFGS + Laplace approximation of the posterior
Output: Counterfactual simulation estimating production had the policy stress been absent
Sensitivity analysis: Regional multiplier analysis for high-extraction provinces

Data

National agricultural panel (1952–1977), 8 variables:

| Variable | Description |
|----------|-------------|
| 'grain_output' | National grain output (million tons) |
| 'procurement' | State grain procurement (million tons) |
| 'retained_pc' | Per-capita retained grain (kg) |
| 'rural_labor' | Rural labor force (million) |
| 'area_sown' | Grain sown area (million hectares) |
| 'draft_animals' | Draft animals (million head) |
| 'machinery_hp' | Agricultural machinery (million horsepower) |
| 'fertilizer' | Chemical fertilizer (million tons) |

Constructed from published replication datasets in the peer-reviewed literature
(see Houser, Sands & Xiao 2009, *Journal of Development Economics*;
Meng, Qian & Yared 2010, NBER).


Key findings

- Persistence parameter $\rho \approx 0.94$: ecological degradation outlasts the policy shock by over a decade — the system does not automatically recover when the policy ends
- Counterfactual output gap: removing the policy stress recovers an estimated 43 million tons/year of grain output (90% CI: [28.3, 55.9]) during the peak stress period

Limitations

This is a high school independent research project, not peer-reviewed work.
Specific limitations acknowledged in the model:

1. The latent variable estimates the inferred aggregate effect of ecological degradation — not a direct measurement of soil organic matter loss (SOM data was not systematically collected during the study period)
2. A complete province-level model (e.g., for Sichuan) would require county-level panel data that are not publicly available
3. The latent variable cannot establish soil degradation in the laboratory-science sense; it captures the pattern in production residuals consistent with ecological damage
4. The model uses 26 annual observations ($T = 26$) to estimate 13 parameters — a small-$T$ setting where Laplace approximation is preferred over MCMC

Technological References

Houser, D., Sands, B., & Xiao, E. (2009). Three parts natural, seven parts man-made: A Bayesian analysis of the Great Leap Forward famine. *Journal of Development Economics*.
Meng, X., Qian, N., & Yared, P. (2010). The institutional causes of China's Great Famine, 1959–1961. *NBER Working Paper*.
Gelman, A., Carlin, J. B., Stern, H. S., & Rubin, D. B. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.
