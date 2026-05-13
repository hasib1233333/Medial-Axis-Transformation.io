"""
Statistical Significance Tests — EGF-MAT vs All Competing Methods
=================================================================
Tests run on all 13,797 images (1,797 NIST + 12,000 MPEG-7).

Methods compared against EGF:
  From public_benchmark_results.json : EDT, Blum
  From alt_mat_benchmark_results.json: VorMAT, HJSkel, MorphS, AFMM

Tests:
  1. Wilcoxon Signed-Rank Test  (recommended — non-parametric, paired)
  2. Paired t-test              (parametric, assumes normality)
  3. McNemar's Test             (binary outcomes: connectivity, 1px-width)

Reference:
  Wilcoxon, F. (1945). "Individual comparisons by ranking methods."
  Biometrics Bulletin, 1(6), 80–83.
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

HERE = Path(__file__).parent

# ── Load data ──────────────────────────────────────────────────────────────────
with open(HERE / 'public_benchmark_results.json') as f:
    pub = json.load(f)

with open(HERE / 'outputs' / 'alt_mat_benchmark_results.json') as f:
    alt = json.load(f)

assert len(pub) == len(alt) == 13797, "Row count mismatch between JSONs"

# ── Extract per-image scores ───────────────────────────────────────────────────
egf_q    = np.array([r['q_EGF']  for r in pub])
egf_sp   = np.array([r['sp_EGF'] for r in pub])
egf_conn = np.array([1 if r['nc_EGF'] == 1 else 0 for r in pub])
egf_w1   = np.array([r['w1_EGF'] for r in pub])

competitors = {
    # name : (q_scores, sp_scores, conn_binary, w1_binary)
    'EDT'   : (np.array([r['q_EDT']  for r in pub]),
               np.array([r['sp_EDT'] for r in pub]),
               np.array([1 if r['nc_EDT'] == 1  else 0 for r in pub]),
               np.array([r['w1_EDT']  for r in pub])),

    'Blum'  : (np.array([r['q_Blum']  for r in pub]),
               np.array([r['sp_Blum'] for r in pub]),
               np.array([1 if r['nc_Blum'] == 1 else 0 for r in pub]),
               np.array([r['w1_Blum']  for r in pub])),

    'VorMAT': (np.array([r['q_VorMAT']  for r in alt]),
               np.array([r['sp_VorMAT'] for r in alt]),
               np.array([1 if r['nc_VorMAT'] == 1 else 0 for r in alt]),
               np.array([r['w1_VorMAT']  for r in alt])),

    'HJSkel': (np.array([r['q_HJSkel']  for r in alt]),
               np.array([r['sp_HJSkel'] for r in alt]),
               np.array([1 if r['nc_HJSkel'] == 1 else 0 for r in alt]),
               np.array([r['w1_HJSkel']  for r in alt])),

    'MorphS': (np.array([r['q_MorphS']  for r in alt]),
               np.array([r['sp_MorphS'] for r in alt]),
               np.array([1 if r['nc_MorphS'] == 1 else 0 for r in alt]),
               np.array([r['w1_MorphS']  for r in alt])),

    'AFMM'  : (np.array([r['q_AFMM']  for r in alt]),
               np.array([r['sp_AFMM'] for r in alt]),
               np.array([1 if r['nc_AFMM'] == 1 else 0 for r in alt]),
               np.array([r['w1_AFMM']  for r in alt])),
}

def sig(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

SEP  = '─' * 90
SEP2 = '═' * 90

# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — Wilcoxon Signed-Rank Test  (Quality & Spurious)
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP2}')
print('TEST 1 — Wilcoxon Signed-Rank Test  (non-parametric, paired, n=13,797)')
print('H0: no difference in median score between EGF and competitor')
print(SEP2)
print(f"  {'Comparison':<20} {'Metric':<12} {'EGF mean':>10} {'Comp mean':>10} "
      f"{'W-stat':>12} {'p-value':>12} {'Sig':>5}")
print(f'  {SEP}')

for name, (cq, csp, cc, cw) in competitors.items():
    # Quality
    w, p = stats.wilcoxon(egf_q, cq, alternative='two-sided')
    print(f"  EGF vs {name:<13} {'Quality':<12} {egf_q.mean():>10.3f} {cq.mean():>10.3f} "
          f"{w:>12.1f} {p:>12.4e} {sig(p):>5}")
    # Spurious
    w, p = stats.wilcoxon(egf_sp, csp, alternative='two-sided')
    print(f"  {'':19} {'Spurious':<12} {egf_sp.mean():>10.3f} {csp.mean():>10.3f} "
          f"{w:>12.1f} {p:>12.4e} {sig(p):>5}")
    print(f'  {"-"*88}')

print(f'  Significance: *** p<0.001   ** p<0.01   * p<0.05   ns = not significant')

# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Paired t-test  (Quality)
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP2}')
print('TEST 2 — Paired t-test  (parametric, n=13,797)')
print('H0: mean quality difference between EGF and competitor = 0')
print(SEP2)
print(f"  {'Comparison':<20} {'Mean Diff':>10} {'t-stat':>10} {'p-value':>12} {'95% CI':>22} {'Sig':>5}")
print(f'  {SEP}')

for name, (cq, csp, cc, cw) in competitors.items():
    diff = egf_q - cq
    t, p = stats.ttest_rel(egf_q, cq)
    ci   = stats.t.interval(0.95, df=len(diff)-1,
                             loc=diff.mean(), scale=stats.sem(diff))
    print(f"  EGF vs {name:<13} {diff.mean():>+10.3f} {t:>10.3f} {p:>12.4e} "
          f"  [{ci[0]:+.3f}, {ci[1]:+.3f}]  {sig(p):>5}")

print(f'\n  Significance: *** p<0.001   ** p<0.01   * p<0.05   ns = not significant')
print(f'  Positive mean diff = EGF quality is higher')

# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — McNemar's Test  (Connectivity & 1-pixel width — binary outcomes)
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP2}')
print("TEST 3 — McNemar's Test  (binary outcomes: Connectivity & 1-Pixel Width)")
print('H0: no difference in proportion of connected/1px skeletons between EGF and competitor')
print(SEP2)
print(f"  {'Comparison':<20} {'Metric':<14} {'EGF %':>8} {'Comp %':>8} "
      f"{'χ²':>10} {'p-value':>12} {'Sig':>5}")
print(f'  {SEP}')

for name, (cq, csp, cc, cw) in competitors.items():
    for metric, egf_bin, comp_bin in [('Connectivity', egf_conn, cc),
                                       ('1-Pix Width',  egf_w1,   cw)]:
        # Build 2x2 contingency table
        b = int(np.sum((egf_bin == 1) & (comp_bin == 0)))  # EGF yes, comp no
        c = int(np.sum((egf_bin == 0) & (comp_bin == 1)))  # EGF no,  comp yes
        table = [[int(np.sum((egf_bin==1)&(comp_bin==1))), b],
                 [c, int(np.sum((egf_bin==0)&(comp_bin==0)))]]
        result = mcnemar(table, exact=False, correction=True)
        egf_pct  = egf_bin.mean() * 100
        comp_pct = comp_bin.mean() * 100
        print(f"  EGF vs {name:<13} {metric:<14} {egf_pct:>7.1f}% {comp_pct:>7.1f}% "
              f"{result.statistic:>10.3f} {result.pvalue:>12.4e} {sig(result.pvalue):>5}")
    print(f'  {"-"*88}')

print(f'  Significance: *** p<0.001   ** p<0.01   * p<0.05   ns = not significant')

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY FOR PAPER
# ══════════════════════════════════════════════════════════════════════════════
print(f'\n{SEP2}')
print('PAPER SUMMARY — Statistical Significance (n=13,797 images)')
print(SEP2)
for name, (cq, csp, cc, cw) in competitors.items():
    _, p_w = stats.wilcoxon(egf_q, cq, alternative='two-sided')
    _, p_t = stats.ttest_rel(egf_q, cq)
    diff   = egf_q.mean() - cq.mean()
    print(f'  EGF vs {name:<8}  ΔQ={diff:+.3f}pp  '
          f'Wilcoxon p={p_w:.4e} {sig(p_w)}  '
          f't-test p={p_t:.4e} {sig(p_t)}')
print(SEP2)
print()
