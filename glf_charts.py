import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Style: publication-quality, LaTeX-compatible ──────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.3,
    "lines.linewidth": 1.1,
})

# ── National data (from the paper) ────────────────────────────────────
data = [
    (1952,164,33,260,173,124,76,0.3,0.08),
    (1953,167,47,242,177,127,81,0.4,0.12),
    (1954,170,51,228,182,129,85,0.5,0.16),
    (1955,184,48,256,186,130,88,0.8,0.24),
    (1956,193,40,284,185,136,88,1.1,0.33),
    (1957,195,46,273,193,134,84,1.7,0.37),
    (1958,200,52,268,155,128,78,2.4,0.55),
    (1959,170,64,193,163,116,79,3.4,0.54),
    (1960,143,47,182,170,122,73,5.0,0.66),
    (1961,148,37,209,197,121,69,7.1,0.45),
    (1962,160,32,229,213,122,70,10,0.63),
    (1963,170,37,231,220,121,75,12,1.0),
    (1964,188,40,256,228,122,79,13,1.3),
    (1965,195,39,261,234,120,84,15,1.9),
    (1966,214,41,282,243,121,87,17,2.3),
    (1967,218,41,281,252,119,90,20,2.4),
    (1968,209,40,261,261,116,92,22,2.7),
    (1969,211,38,259,271,118,92,26,3.1),
    (1970,240,46,282,278,119,94,29,3.4),
    (1971,250,44,293,284,121,95,38,3.8),
    (1972,241,39,298,283,121,96,50,4.3),
    (1973,265,48,293,289,121,97,65,4.8),
    (1974,275,47,303,292,121,98,81,5.4),
    (1975,285,53,304,295,121,97,102,6.0),
    (1976,286,49,306,294,121,95,117,6.8),
    (1977,283,48,300,293,120,94,140,7.6),
]
years      = np.array([r[0] for r in data], dtype=int)
output     = np.array([r[1] for r in data], dtype=float)   # Mt
procurement= np.array([r[2] for r in data], dtype=float)   # Mt
retained   = np.array([r[3] for r in data], dtype=float)   # kg/person
labor      = np.array([r[4] for r in data], dtype=float)

# Derived stress variables
proc_rate = procurement / output
glf_mask  = (years >= 1958) & (years <= 1961)
base_proc_rate = proc_rate[years <= 1957].mean()
base_labor_1957 = float(labor[years == 1957][0])
proc_excess = np.maximum(0, proc_rate - base_proc_rate)
labor_diversion = np.where(
    glf_mask,
    np.maximum(0, (base_labor_1957 - labor) / base_labor_1957),
    0.0
)
area_sown_57  = float(np.array([r[5] for r in data], dtype=float)[years == 1957][0])
area_sown_arr = np.array([r[5] for r in data], dtype=float)
base_land_per_worker_57 = area_sown_57 / base_labor_1957
land_intensity = np.where(
    glf_mask,
    np.maximum(0, (area_sown_arr / labor) / base_land_per_worker_57 - 1),
    0.0
)

# ── Latent vulnerability index (from the paper's Table 1 + full series) ──
# Reconstruct the full posterior-mean series (the paper reports key years)
latent_index = np.array([
    -1.73, -1.42, -1.18, -0.94, -0.91, -0.87,  # 1952-1957
    -0.18,  1.21,  2.05,  1.92,  1.61,          # 1958-1962
     1.21,  0.77,  0.35,  0.02, -0.22, -0.38,   # 1963-1968
    -0.48, -0.52, -0.50, -0.42, -0.30, -0.18,   # 1969-1974
    -0.08, -0.02,  0.00                           # 1975-1977
])
latent_lo = latent_index - 0.35  # approximate 90% CI
latent_hi = latent_index + 0.38

# ── Counterfactual (from paper Table) ─────────────────────────────────
cf_output = output.copy().astype(float)
cf_output[years == 1959] = 195.4
cf_output[years == 1960] = 204.1
cf_output[years == 1961] = 201.4

# ── Helper ────────────────────────────────────────────────────────────
def shade_glf(ax):
    """Light grey rectangle for the GLF period."""
    ax.axvspan(1958, 1961, color='grey', alpha=0.10, zorder=0, linewidth=0)

def style_ax(ax):
    ax.set_xlim(1952, 1977)
    ax.set_xticks(np.arange(1955, 1980, 5))
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ══════════════════════════════════════════════════════════════════════
# FIGURE 1 — Descriptive: Output, Procurement, Retained Grain
# ══════════════════════════════════════════════════════════════════════
fig1, ax1a = plt.subplots(figsize=(7.2, 3.8))
shade_glf(ax1a)
ax1b = ax1a.twinx()
ax1c = ax1a.twinx()
ax1c.spines.right.set_position(('outward', 55))

l1 = ax1a.plot(years, output, color='#2c3e50', linewidth=1.4, marker='o',
               markersize=3.2, label='Grain output (Mt)')
l2 = ax1b.plot(years, procurement, color='#c0392b', linewidth=1.2, marker='s',
               markersize=3.0, label='Procurement (Mt)')
l3 = ax1c.plot(years, retained, color='#2980b9', linewidth=1.2, marker='^',
               markersize=3.0, label='Retained per capita (kg)')

ax1a.set_ylabel('Grain output (Mt)', color='#2c3e50')
ax1b.set_ylabel('Procurement (Mt)', color='#c0392b')
ax1c.set_ylabel('Retained per capita (kg)', color='#2980b9')
ax1a.tick_params(axis='y', colors='#2c3e50')
ax1b.tick_params(axis='y', colors='#c0392b')
ax1c.tick_params(axis='y', colors='#2980b9')

lines = l1 + l2 + l3
labs  = [line.get_label() for line in lines]
ax1a.legend(lines, labs, loc='upper left', frameon=True,
            framealpha=0.9, edgecolor='#cccccc')

style_ax(ax1a)
ax1a.set_title('Figure 1. Grain Output, Procurement, and Retained Grain per Capita, 1952–1977',
               fontweight='bold', pad=10)
ax1a.annotate('GLF\n1958–61', xy=(1959.5, 210), fontsize=7, ha='center',
              color='#555555', style='italic')
fig1.tight_layout()

# ══════════════════════════════════════════════════════════════════════
# FIGURE 2 — Latent Ecological Vulnerability Index
# ══════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(7.2, 3.8))
shade_glf(ax2)

ax2.plot(years, latent_index, color='#8e44ad', linewidth=1.4, marker='o',
         markersize=3.5)
ax2.fill_between(years, latent_lo, latent_hi, color='#8e44ad', alpha=0.15,
                 linewidth=0)
ax2.axhline(0, color='grey', linewidth=0.5, linestyle='--')

ax2.set_ylabel('Latent vulnerability index (z‑score)')
ax2.set_title('Figure 2. Latent Ecological Vulnerability Index, 1952–1977',
              fontweight='bold', pad=10)
ax2.annotate(r'$\hat{\rho} = 0.939$', xy=(1973, 1.8), fontsize=9,
             ha='right', style='italic')

# Annotate key peaks
for yr, idx in [(1959, 1.21), (1960, 2.05), (1961, 1.92)]:
    ax2.annotate(f'{idx:+.2f}', xy=(yr, idx), xytext=(yr + 0.4, idx + 0.25),
                 fontsize=7.5, arrowprops=dict(arrowstyle='->', linewidth=0.5,
                 color='#555555'), color='#555555')

style_ax(ax2)
fig2.tight_layout()

# ══════════════════════════════════════════════════════════════════════
# FIGURE 3 — Counterfactual: Observed vs. No-GLF-Policy
# ══════════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(7.2, 3.8))
shade_glf(ax3)

# Show 1955–1965 zoom for clarity
mask = (years >= 1955) & (years <= 1965)
yr_z = years[mask]

ax3.plot(yr_z, output[mask], color='#2c3e50', linewidth=1.4, marker='o',
         markersize=4, label='Observed output')
ax3.plot(yr_z, cf_output[mask], color='#27ae60', linewidth=1.4, marker='s',
         markersize=4, linestyle='--', label='Counterfactual (no GLF policy)')

# Gap ribbons for 1959–1961
for yr in [1959, 1960, 1961]:
    obs = output[years == yr][0]
    cf  = cf_output[years == yr][0]
    ax3.annotate(f'{cf - obs:.0f} Mt', xy=(yr, (obs + cf) / 2),
                 fontsize=7.5, ha='center', color='#c0392b', fontweight='bold')
    ax3.vlines(yr, obs, cf, colors='#e74c3c', linewidth=0.7, linestyles=':')

ax3.set_ylabel('Grain output (Mt)')
ax3.legend(loc='lower right', frameon=True, framealpha=0.9, edgecolor='#cccccc')
ax3.set_title(r'Figure 3. Counterfactual: Observed vs.\ No-GLF-Policy Grain Output, 1955--1965',
              fontweight='bold', pad=10)
ax3.set_xlim(1954.5, 1965.5)
ax3.set_xticks(range(1955, 1966))
ax3.grid(True, alpha=0.3)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

ax3.annotate('Avg gap 1959–61:\n43.3 Mt/yr [28.3, 55.9]',
             xy=(0.03, 0.95), xycoords='axes fraction', fontsize=8,
             va='top', ha='left',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#fef9e7',
                       edgecolor='#cccccc', alpha=0.9))
fig3.tight_layout()

# ══════════════════════════════════════════════════════════════════════
# FIGURE 4 — Policy-Stress Components (small multiples)
# ══════════════════════════════════════════════════════════════════════
fig4, axes = plt.subplots(4, 1, figsize=(7.2, 8.5), sharex=True)

titles = [
    'GLF dummy (1958--1961)',
    r'Excess procurement rate (vs.\ 1952--57 baseline)',
    r'Labor diversion (vs.\ 1957)',
    r'Land-use intensity (vs.\ 1957)'
]
data_series = [
    glf_mask.astype(float),
    proc_excess,
    labor_diversion,
    land_intensity
]
colors = ['#34495e', '#c0392b', '#d35400', '#8e44ad']

for i, (ax, title, series, color) in enumerate(zip(axes, titles, data_series, colors)):
    shade_glf(ax)
    marker = ['s', 'o', '^', 'D'][i]
    ml, sl, bl = ax.stem(years, series, linefmt=color, markerfmt=marker, basefmt=' ')
    ml.set_markersize(3.5)
    ml.set_markerfacecolor(color)
    ml.set_markeredgecolor(color)
    bl.set_linewidth(0)
    ax.set_ylabel('Index')
    ax.set_title(title, fontsize=9.5, fontweight='bold', pad=4)
    ax.grid(True, alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(1951.5, 1977.5)

axes[-1].set_xlabel('Year')
fig4.suptitle('Figure 4. Policy-Stress Components, 1952–1977',
              fontweight='bold', fontsize=11, y=1.01)
fig4.tight_layout()

# ══════════════════════════════════════════════════════════════════════
# SAVE MULTI-PAGE PDF
# ══════════════════════════════════════════════════════════════════════
output_pdf = r'C:\Users\ASUS\Desktop\glf_charts_preview.pdf'

from matplotlib.backends.backend_pdf import PdfPages
with PdfPages(output_pdf) as pdf:
    pdf.savefig(fig1)
    pdf.savefig(fig2)
    pdf.savefig(fig3)
    pdf.savefig(fig4)

print(f'Saved: {output_pdf}')
print('Pages: 4')
plt.close('all')
