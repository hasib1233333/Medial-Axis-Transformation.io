"""
EGF Public Benchmark — Full Analysis Figures  (legend-safe version)
====================================================================
Fixes all legend / caption overlap issues.

Figures produced (saved to V4/outputs/):
  fig_analysis_1.png  — Fig. 1: MPEG-7 per-category Quality & Spurious (A, B)
  fig_analysis_2.png  — Fig. 2: NIST per-digit + per-source summary + Conn/1px (C, D, E)
  fig_analysis_3.png  — Fig. 3: Benchmark summary table + radar chart
  fig_analysis_4.png  — Fig. 4: Complete per-source metrics grid (A–F)
"""

import warnings; warnings.filterwarnings('ignore')
import json, os as _os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pathlib import Path

_HERE = Path(_os.path.dirname(_os.path.abspath(__file__)))

# ── Import EGF skeleton from egf.py (same directory, cached in sys.modules) ───
import sys, importlib.util as _ilu
if 'egf' not in sys.modules:
    _egf_spec = _ilu.spec_from_file_location('egf', str(_HERE / 'egf.py'))
    _egf_mod  = _ilu.module_from_spec(_egf_spec)
    _egf_spec.loader.exec_module(_egf_mod)
    sys.modules['egf'] = _egf_mod
egf_skeleton = sys.modules['egf'].egf_100conn_skeleton

# ── Academic serif style ──────────────────────────────────────────────────────
SERIF = ["Times New Roman", "DejaVu Serif", "Palatino", "serif"]
plt.rcParams.update({
    "font.family": "serif", "font.serif": SERIF,
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "figure.titlesize": 9,
    "lines.linewidth": 1.0, "axes.linewidth": 0.8,
    "figure.dpi": 100, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

DPI = 300
OUT = _HERE / 'outputs'
OUT.mkdir(exist_ok=True)

# Hatch / colour scheme (white bars with patterns — matches reference)
H_EDT  = 'ZZ'; H_BLUM = ''; H_EGF = '////'
C_EDT  = 'white'; C_BLUM = 'white'; C_EGF = 'white'
EC_EDT = '#111111'; EC_BLUM = '#111111'; EC_EGF = '#111111'
LW = 0.6

ALGO_LABELS  = ['EDT', 'Blum', 'EGF']
ALGO_HATCHES = [H_EDT, H_BLUM, H_EGF]
ALGO_COLORS  = [C_EDT, C_BLUM, C_EGF]
ALGO_ECS     = [EC_EDT, EC_BLUM, EC_EGF]

def algo_legend_handles():
    return [mpatches.Patch(facecolor=c, edgecolor=ec, hatch=h, linewidth=LW, label=lbl)
            for lbl, h, c, ec in zip(ALGO_LABELS, ALGO_HATCHES, ALGO_COLORS, ALGO_ECS)]

# ── Load data ─────────────────────────────────────────────────────────────────
json_path = _HERE / 'public_benchmark_results.json'
with open(json_path) as f:
    raw = json.load(f)

ALL  = raw
NIST = [r for r in raw if r['source'] == 'NIST_Digits']
MPEG = [r for r in raw if r['source'] == 'MPEG7_Equiv']

# Pre-group rows by (source, cat) — avoids repeated full-list filtering in cat_means
from collections import defaultdict
_grp: dict = defaultdict(list)
for _r in raw:
    _grp[(_r['source'], _r['cat'])].append(_r)

def cat_means(rows, metric_fmt):
    """O(N) instead of O(cats × algos × N): uses pre-grouped lookup."""
    source = rows[0]['source']
    cats   = sorted({r['cat'] for r in rows})
    keys   = {a: metric_fmt.format(a) for a in ALGO_LABELS}
    return {
        cat: {a: float(np.mean([r[keys[a]] for r in _grp[(source, cat)]]))
              for a in ALGO_LABELS}
        for cat in cats
    }

def overall_mean(rows, metric_fmt):
    """Single pass over rows for all algos."""
    keys = {a: metric_fmt.format(a) for a in ALGO_LABELS}
    n    = len(rows)
    sums = {a: 0.0 for a in ALGO_LABELS}
    for r in rows:
        for a in ALGO_LABELS:
            sums[a] += r[keys[a]]
    return {a: sums[a] / n for a in ALGO_LABELS}

SOURCES = {'NIST Digits': NIST, 'MPEG-7 Equiv': MPEG, 'Overall': ALL}

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1  —  MPEG-7 per-category  Quality (A)  &  Spurious (B)
# ═══════════════════════════════════════════════════════════════════════════════
def make_fig1():
    mpeg_cats = sorted(set(r['cat'] for r in MPEG))
    n_cats    = len(mpeg_cats)
    q_by_cat  = cat_means(MPEG, 'q_{}')
    sp_by_cat = cat_means(MPEG, 'sp_{}')
    q_means   = overall_mean(MPEG, 'q_{}')
    sp_means  = overall_mean(MPEG, 'sp_{}')

    fig, (ax_q, ax_sp) = plt.subplots(1, 2, figsize=(16, 5.5))
    fig.suptitle(
        'Fig. 1.  Quality & Spurious — MPEG-7 CE-Shape-1 Equivalent (Latecki et al. 1999)\n'
        f'60 categories × 200 augmented images = 12,000 images  |  '
        f'EGF >= Blum on {sum(1 for c in mpeg_cats if q_by_cat[c]["EGF"] >= q_by_cat[c]["Blum"])}/60 cats',
        fontsize=8.5, fontweight='bold'
    )

    x = np.arange(n_cats)
    w = 0.26

    # ── (A) Quality ───────────────────────────────────────────────────────────
    ax = ax_q
    for i, (algo, hatch, color, ec) in enumerate(zip(ALGO_LABELS, ALGO_HATCHES, ALGO_COLORS, ALGO_ECS)):
        vals = [q_by_cat[c][algo] for c in mpeg_cats]
        ax.bar(x + (i - 1)*w, vals, w, label=algo,
               facecolor=color, edgecolor=ec, hatch=hatch, linewidth=LW)

    ax.axhline(q_means['EGF'],  color='black', lw=0.9, ls='-')
    ax.axhline(q_means['Blum'], color='black', lw=0.9, ls='--')
    ax.axhline(q_means['EDT'],  color='black', lw=0.9, ls=':')

    ax.set_xticks(x)
    ax.set_xticklabels(mpeg_cats, rotation=90, fontsize=5.5)
    ax.set_ylabel('Quality Score (%)')
    ax.set_ylim(20, 105)
    ax.set_title('(A)  Quality — MPEG-7 Equivalent (60 categories, n=200 each)', fontsize=8, pad=6)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.grid(axis='y', which='major', lw=0.3, alpha=0.5)

    # Legend: bar patches + line handles — placed OUTSIDE at upper left above x-axis
    bar_handles = algo_legend_handles()
    line_handles = [
        mpatches.Patch(color='none', label=f'EGF mean({q_means["EGF"]:.2f}%)'),
        mpatches.Patch(color='none', label=f'Blum mean({q_means["Blum"]:.2f}%)'),
        mpatches.Patch(color='none', label=f'EDT({q_means["EDT"]:.2f}%)'),
    ]
    ax.legend(handles=bar_handles + line_handles,
              loc='lower left', bbox_to_anchor=(0.01, 0.01),
              ncol=2, framealpha=0.92, fontsize=5.8,
              edgecolor='#aaaaaa', borderpad=0.5)

    # ── (B) Spurious ─────────────────────────────────────────────────────────
    ax = ax_sp
    for i, (algo, hatch, color, ec) in enumerate(zip(ALGO_LABELS, ALGO_HATCHES, ALGO_COLORS, ALGO_ECS)):
        vals = [sp_by_cat[c][algo] for c in mpeg_cats]
        ax.bar(x + (i - 1)*w, vals, w, label=algo,
               facecolor=color, edgecolor=ec, hatch=hatch, linewidth=LW)

    ax.axhline(sp_means['EGF'],  color='black', lw=0.9, ls='-')
    ax.axhline(sp_means['EDT'],  color='black', lw=0.9, ls=':')
    ax.axhline(sp_means['Blum'], color='black', lw=0.9, ls='--')

    ax.set_xticks(x)
    ax.set_xticklabels(mpeg_cats, rotation=90, fontsize=5.5)
    ax.set_ylabel('Spurious Count  (lower is better)')
    ax.set_title('(B)  Spurious Branches — MPEG-7 Equivalent', fontsize=8, pad=6)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.grid(axis='y', which='major', lw=0.3, alpha=0.5)

    # Legend outside in upper right corner — bbox_to_anchor pushes it clear of tall bars
    sp_handles = algo_legend_handles() + [
        mpatches.Patch(color='none', label=f'EGF mean {sp_means["EGF"]:.2f} (lower)'),
        mpatches.Patch(color='none', label=f'EDT {sp_means["EDT"]:.2f} (higher)'),
        mpatches.Patch(color='none', label=f'Blum {sp_means["Blum"]:.2f}'),
    ]
    ax.legend(handles=sp_handles,
              loc='upper right', bbox_to_anchor=(0.99, 0.99),
              ncol=1, framealpha=0.92, fontsize=5.8,
              edgecolor='#aaaaaa', borderpad=0.5)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = OUT / 'fig_analysis_1.png'
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  saved {path.name}  ({path.stat().st_size//1024} KB)')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2  —  NIST per-digit (C)  +  Per-source summary (D)  +  Conn/1px (E)
# ═══════════════════════════════════════════════════════════════════════════════
def make_fig2():
    digit_cats = sorted(set(r['cat'] for r in NIST))
    q_digit    = cat_means(NIST, 'q_{}')
    q_src      = {s: overall_mean(rows, 'q_{}') for s, rows in SOURCES.items()}
    conn_src   = {s: {a: np.mean([1 if r[f'nc_{a}']==1 else 0 for r in rows])*100
                      for a in ALGO_LABELS}
                  for s, rows in SOURCES.items()}
    w1_src     = {s: {a: np.mean([r[f'w1_{a}'] for r in rows])*100 for a in ALGO_LABELS}
                  for s, rows in SOURCES.items()}

    n_nist = len(NIST); n_mpeg = len(MPEG); n_all = len(ALL)

    fig = plt.figure(figsize=(16, 5.2), constrained_layout=False)
    # Give each panel a bit more horizontal room; allocate extra left margin for (C)
    gs = fig.add_gridspec(1, 3, left=0.06, right=0.97,
                          bottom=0.16, top=0.86,
                          wspace=0.42)
    ax_c = fig.add_subplot(gs[0])
    ax_d = fig.add_subplot(gs[1])
    ax_e = fig.add_subplot(gs[2])

    fig.suptitle(
        f'Fig. 2.  NIST Digits, Per-Source Summary & Key Properties — {n_all:,} Public Images\n'
        f'Sources: sklearn load_digits [D1] + MPEG-7 CE-Shape-1 shapes [D2] (Latecki et al. 1999)',
        fontsize=8.5, fontweight='bold', y=0.99
    )

    w = 0.26
    x10 = np.arange(len(digit_cats))

    # ── (C) NIST per-digit quality ────────────────────────────────────────────
    ax = ax_c
    for i, (algo, hatch, color, ec) in enumerate(zip(ALGO_LABELS, ALGO_HATCHES, ALGO_COLORS, ALGO_ECS)):
        vals = [q_digit[c][algo] for c in digit_cats]
        bars = ax.bar(x10 + (i-1)*w, vals, w,
                      facecolor=color, edgecolor=ec, hatch=hatch, linewidth=LW)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                    f'{v:.0f}', ha='center', va='bottom', fontsize=4.2, rotation=90)

    ax.set_xticks(x10)
    ax.set_xticklabels([f'Digit\n{c.split("_")[1]}' for c in digit_cats], fontsize=6.5)
    ax.set_ylabel('Quality (%)')
    ax.set_ylim(30, 122)   # extra headroom so value labels don't overlap title
    ax.set_title(f'(C)  NIST Digits [D1] — Quality by Digit\n'
                 f'({n_nist:,} real hand-written images, Public Domain)', fontsize=7.5, pad=5)
    ax.legend(handles=algo_legend_handles(),
              loc='lower right', bbox_to_anchor=(0.99, 0.01),
              fontsize=6, framealpha=0.92, edgecolor='#aaaaaa')
    ax.grid(axis='y', lw=0.3, alpha=0.5)

    # ── (D) Per-source quality summary ────────────────────────────────────────
    ax = ax_d
    src_labels = list(SOURCES.keys())
    src_ns     = [n_nist, n_mpeg, n_all]
    x3 = np.arange(3)
    for i, (algo, hatch, color, ec) in enumerate(zip(ALGO_LABELS, ALGO_HATCHES, ALGO_COLORS, ALGO_ECS)):
        vals = [q_src[s][algo] for s in src_labels]
        bars = ax.bar(x3 + (i-1)*w, vals, w,
                      facecolor=color, edgecolor=ec, hatch=hatch, linewidth=LW)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=6, fontweight='bold')

    ax.set_xticks(x3)
    ax.set_xticklabels(
        [f'{s}\n[D{"1" if "NIST" in s else "2" if "MPEG" in s else "1+2"}]  n={n:,}'
         for s, n in zip(src_labels, src_ns)],
        fontsize=6.5)
    ax.set_ylabel('Mean Quality (%)')
    ax.set_ylim(40, 112)
    ax.set_title('(D)  Per-Source Quality Summary\nEGF >= Blum on both datasets', fontsize=7.5, pad=5)
    ax.legend(handles=algo_legend_handles(),
              loc='lower right', bbox_to_anchor=(0.99, 0.01),
              fontsize=6, framealpha=0.92, edgecolor='#aaaaaa')
    ax.grid(axis='y', lw=0.3, alpha=0.5)

    # ── (E) Connectivity & 1-pixel width (overall only) ───────────────────────
    ax = ax_e
    conn_vals = [conn_src['Overall'][a] for a in ALGO_LABELS]
    w1_vals   = [w1_src['Overall'][a]   for a in ALGO_LABELS]
    bar_w = 0.38
    b1 = ax.bar(x3 - bar_w/2, conn_vals, bar_w,
                facecolor='white', edgecolor='black', hatch='', linewidth=LW,
                label='Connectivity %')
    b2 = ax.bar(x3 + bar_w/2, w1_vals, bar_w,
                facecolor='white', edgecolor='black', hatch='....', linewidth=LW,
                label='1-Pixel Width %')

    for bar, v in zip(list(b1) + list(b2), conn_vals + w1_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f'{v:.1f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')

    ax.set_xticks(x3)
    ax.set_xticklabels(ALGO_LABELS, fontsize=8)
    ax.set_ylabel('Rate (%)')
    c_edt = conn_src['Overall']['EDT']; w1_egf = w1_src['Overall']['EGF']
    ax.set_title(f'(E)  Connectivity & 1-Pixel Width\n'
                 f'EDT: {c_edt:.1f}% conn (fails {100-c_edt:.1f}%)  EGF 1px={w1_egf:.1f}%',
                 fontsize=7.5, pad=5)
    ax.set_ylim(0, 120)
    # Legend below the chart — no overlap with bars
    ax.legend(fontsize=6.5, framealpha=0.92, edgecolor='#aaaaaa',
              loc='upper left', bbox_to_anchor=(0.01, 0.99))
    ax.grid(axis='y', lw=0.3, alpha=0.5)

    path = OUT / 'fig_analysis_2.png'
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  saved {path.name}  ({path.stat().st_size//1024} KB)')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3  —  Benchmark summary table + radar chart
# ═══════════════════════════════════════════════════════════════════════════════
def make_fig3():
    # Single pass over ALL for all summary metrics
    n = len(ALL)
    sums = {k: 0.0 for a in ALGO_LABELS
            for k in (f'q_{a}', f'sp_{a}', f'w1_{a}', f'ms_{a}', f'conn_{a}')}
    for r in ALL:
        for a in ALGO_LABELS:
            sums[f'q_{a}']    += r[f'q_{a}']
            sums[f'sp_{a}']   += r[f'sp_{a}']
            sums[f'w1_{a}']   += r[f'w1_{a}']
            sums[f'ms_{a}']   += r[f'ms_{a}']
            sums[f'conn_{a}'] += 1 if r[f'nc_{a}'] == 1 else 0
    q_all    = {a: sums[f'q_{a}']    / n              for a in ALGO_LABELS}
    sp_all   = {a: sums[f'sp_{a}']   / n              for a in ALGO_LABELS}
    w1_all   = {a: sums[f'w1_{a}']   / n * 100        for a in ALGO_LABELS}
    ms_all   = {a: sums[f'ms_{a}']   / n              for a in ALGO_LABELS}
    conn_all = {a: sums[f'conn_{a}'] / n * 100        for a in ALGO_LABELS}

    fig = plt.figure(figsize=(13, 5.5))
    gs  = fig.add_gridspec(1, 2, left=0.02, right=0.98, bottom=0.08, top=0.88,
                           wspace=0.12)
    ax_tbl = fig.add_subplot(gs[0])
    ax_tbl.axis('off')

    fig.suptitle('Fig. 3.  Overall Benchmark Summary — EGF-MAT vs EDT vs Blum',
                 fontsize=9, fontweight='bold', y=0.97)

    # ── Table ─────────────────────────────────────────────────────────────────
    rows_data = [
        ['Metric',            'EDT',                    'Blum',                   'EGF',                    'Winner'],
        ['Quality (%)',
         f"{q_all['EDT']:.3f}", f"{q_all['Blum']:.3f}", f"{q_all['EGF']:.3f}",
         'EGF' if q_all['EGF'] >= max(q_all['EDT'], q_all['Blum']) else 'Blum'],
        ['Spurious branches',
         f"{sp_all['EDT']:.3f}", f"{sp_all['Blum']:.3f}", f"{sp_all['EGF']:.3f}",
         'EGF' if sp_all['EGF'] <= min(sp_all['EDT'], sp_all['Blum']) else 'Blum'],
        ['Connectivity (%)',
         f"{conn_all['EDT']:.1f}", f"{conn_all['Blum']:.1f}", f"{conn_all['EGF']:.1f}",
         'EGF' if conn_all['EGF'] >= max(conn_all['EDT'], conn_all['Blum']) else 'Blum'],
        ['1-Pixel Width (%)',
         f"{w1_all['EDT']:.1f}", f"{w1_all['Blum']:.1f}", f"{w1_all['EGF']:.1f}",
         'EDT' if w1_all['EDT'] >= max(w1_all['Blum'], w1_all['EGF']) else 'EGF'],
        ['Speed (ms)',
         f"{ms_all['EDT']:.3f}", f"{ms_all['Blum']:.3f}", f"{ms_all['EGF']:.3f}",
         'Blum' if ms_all['Blum'] <= min(ms_all['EDT'], ms_all['EGF']) else 'EDT'],
        ['Total images', f'{len(ALL):,}', f'{len(ALL):,}', f'{len(ALL):,}', '—'],
    ]
    col_x     = [0.03, 0.34, 0.50, 0.66, 0.82]
    n_rows    = len(rows_data)
    row_h     = 1.0 / n_rows
    for ri, row in enumerate(rows_data):
        y = 1 - (ri + 0.5) * row_h
        bg = '#dddddd' if ri == 0 else ('#f5f5f5' if ri % 2 == 0 else 'white')
        # Row background rectangle
        ax_tbl.axhspan(1 - (ri+1)*row_h, 1 - ri*row_h,
                       xmin=0, xmax=1, facecolor=bg, alpha=0.6, linewidth=0)
        for ci, (cell, cx) in enumerate(zip(row, col_x)):
            weight = 'bold' if ri == 0 or ci == 0 else 'normal'
            color  = '#005500' if ('EGF' in str(cell) and ri > 0 and ci == 4) else \
                     '#550000' if ('Blum' in str(cell) and ri > 0 and ci == 4 and
                                   'Speed' in rows_data[ri][0]) else 'black'
            ax_tbl.text(cx, y, cell,
                        transform=ax_tbl.transAxes,
                        ha='left', va='center', fontsize=9,
                        fontweight=weight, color=color,
                        fontfamily='serif')
    # Horizontal separator after header
    ax_tbl.plot([0, 1], [1 - row_h, 1 - row_h], color='#888888', lw=0.8,
                transform=ax_tbl.transAxes, clip_on=False)
    ax_tbl.set_xlim(0, 1); ax_tbl.set_ylim(0, 1)
    ax_tbl.set_title('Summary Table', fontsize=9, fontweight='bold', pad=6)

    # ── Radar chart ───────────────────────────────────────────────────────────
    cats_radar = ['Quality', 'Low Spur', 'Connected', '1-Pixel', 'Speed']
    N      = len(cats_radar)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

    max_sp = max(sp_all.values()); max_ms = max(ms_all.values())
    def radar_vals(algo):
        return [
            q_all[algo],
            max(0, 100 - sp_all[algo] / max_sp * 100) if max_sp > 0 else 100,
            conn_all[algo],
            w1_all[algo],
            max(0, 100 - ms_all[algo] / max_ms * 100) if max_ms > 0 else 100,
        ]

    ax_rad = fig.add_subplot(gs[1], polar=True)
    styles = [('EDT', '#cc3333', '--'), ('Blum', '#3366cc', ':'), ('EGF', '#22aa22', '-')]
    for algo, col, ls in styles:
        rv = radar_vals(algo)
        ax_rad.plot(angles, rv + [rv[0]], ls, color=col, lw=2.0, label=algo)
        ax_rad.fill(angles, rv + [rv[0]], alpha=0.07, color=col)

    ax_rad.set_xticks(angles[:-1])
    ax_rad.set_xticklabels(cats_radar, fontsize=8)
    ax_rad.set_ylim(0, 105)
    ax_rad.set_title('Performance Radar\n(all metrics normalised 0–100)',
                     fontsize=8, pad=16)
    # Place legend outside the polar plot so it doesn't overlap the spider-web
    ax_rad.legend(loc='upper right', bbox_to_anchor=(1.38, 1.12), fontsize=8,
                  framealpha=0.92, edgecolor='#aaaaaa')

    path = OUT / 'fig_analysis_3.png'
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  saved {path.name}  ({path.stat().st_size//1024} KB)')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4  —  Complete Per-Source Metrics  A–F
# ═══════════════════════════════════════════════════════════════════════════════
def make_fig4():
    src_labels = list(SOURCES.keys())
    src_ns     = [len(NIST), len(MPEG), len(ALL)]
    x3 = np.arange(3); w = 0.26

    def src_metric(fmt):
        return {s: overall_mean(rows, fmt) for s, rows in SOURCES.items()}

    Q   = src_metric('q_{}')
    SP  = src_metric('sp_{}')
    MS  = src_metric('ms_{}')
    CON = {s: {a: np.mean([1 if r[f'nc_{a}']==1 else 0 for r in rows])*100 for a in ALGO_LABELS}
           for s, rows in SOURCES.items()}
    W1  = {s: {a: np.mean([r[f'w1_{a}'] for r in rows])*100 for a in ALGO_LABELS}
           for s, rows in SOURCES.items()}

    mpeg_cats = sorted(set(r['cat'] for r in MPEG))
    q_cat     = cat_means(MPEG, 'q_{}')
    egf_adv   = {c: q_cat[c]['EGF'] - q_cat[c]['Blum'] for c in mpeg_cats}
    top20     = sorted(egf_adv, key=lambda c: abs(egf_adv[c]), reverse=True)[:20]

    n_all = len(ALL); n_nist = len(NIST); n_mpeg = len(MPEG)

    fig = plt.figure(figsize=(17, 10))
    # explicit margins — extra bottom for rotated x-labels in (F), extra top for suptitle
    gs  = fig.add_gridspec(2, 3,
                           left=0.07, right=0.97,
                           bottom=0.12, top=0.90,
                           hspace=0.58, wspace=0.38)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    fig.suptitle(
        f'Fig. 4.  Complete Per-Source Metrics — {n_all:,} Images\n'
        f'NIST Digits (sklearn, Public Domain) + MPEG-7 CE-Shape-1 Equivalent (Latecki et al. 1999)',
        fontsize=9, fontweight='bold', y=0.97
    )

    xtick_labels = [f'{s}\n({n:,})' for s, n in zip(src_labels, src_ns)]

    def grouped_bars(ax, data_dict, ylabel, title, ylim, val_fmt='{:.1f}',
                     legend_loc='lower right'):
        for i, (algo, hatch, color, ec) in enumerate(
                zip(ALGO_LABELS, ALGO_HATCHES, ALGO_COLORS, ALGO_ECS)):
            vals = [data_dict[s][algo] for s in src_labels]
            bars = ax.bar(x3 + (i-1)*w, vals, w,
                          facecolor=color, edgecolor=ec, hatch=hatch, linewidth=LW)
            pad = (ylim[1] - ylim[0]) * 0.012
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + pad,
                        val_fmt.format(v), ha='center', va='bottom',
                        fontsize=5.8, fontweight='bold')
        ax.set_xticks(x3)
        ax.set_xticklabels(xtick_labels, fontsize=6.5)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_title(title, fontsize=7.8, pad=5)
        ax.set_ylim(*ylim)
        ax.legend(handles=algo_legend_handles(),
                  fontsize=5.8, framealpha=0.92, edgecolor='#aaaaaa',
                  loc=legend_loc,
                  bbox_to_anchor=(0.99, 0.01) if 'right' in legend_loc else (0.01, 0.99))
        ax.grid(axis='y', lw=0.3, alpha=0.5)

    grouped_bars(axes[0], Q,   'Quality Score (%)',      '(A)  Quality',
                 ylim=(40, 112), legend_loc='lower right')
    grouped_bars(axes[1], SP,  'Spurious Count  (lower is better)', '(B)  Spurious Branches',
                 ylim=(0, 11),  legend_loc='upper right')
    grouped_bars(axes[2], CON, 'Connectivity (%)',        '(C)  Connectivity Rate',
                 ylim=(0, 115), legend_loc='lower right')
    grouped_bars(axes[3], W1,  '1-Pixel Width (%)',       '(D)  Strict 1-Pixel Width',
                 ylim=(85, 116), legend_loc='lower right')
    ms_max = max(v for s in MS.values() for v in s.values())
    grouped_bars(axes[4], MS,  'Processing Time (ms)',    '(E)  Processing Time',
                 ylim=(0, ms_max * 1.45), val_fmt='{:.2f}', legend_loc='upper right')

    # (F) EGF quality advantage — top 20 categories
    ax = axes[5]
    adv_vals   = [egf_adv[c] for c in top20]
    bar_colors = ['#2d7a2d' if v >= 0 else '#aa2222' for v in adv_vals]
    ax.bar(np.arange(20), adv_vals, color=bar_colors, edgecolor='black', linewidth=LW)
    mean_adv = np.mean(list(egf_adv.values()))
    ax.axhline(mean_adv, color='black', lw=1.0, ls='--',
               label=f'Mean Delta={mean_adv:+.2f}pp')
    ax.axhline(0, color='black', lw=0.6, ls='-')
    ax.set_xticks(np.arange(20))
    ax.set_xticklabels(top20, rotation=90, fontsize=5.5)
    ax.set_ylabel('EGF - Blum Quality (pp)', fontsize=7)
    ax.set_title('(F)  EGF Quality Advantage per Category\n(Top 20 by absolute difference)',
                 fontsize=7.8, pad=5)
    # Legend in upper right — no bars reach the top
    ax.legend(fontsize=6, framealpha=0.92, edgecolor='#aaaaaa',
              loc='upper right', bbox_to_anchor=(0.99, 0.99))
    ax.grid(axis='y', lw=0.3, alpha=0.5)

    path = OUT / 'fig_analysis_4.png'
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  saved {path.name}  ({path.stat().st_size//1024} KB)')


# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# TABLE 2  —  Overall results across all 13,797 images  (paper Table 2 style)
# ══════════════════════════════════════════════════════════════════════════════
def print_results_table():
    """
    Prints Table 2 from the paper:
    'Overall Results Across All 13,797 Images'
    Metrics: Q(%), Sp, 1px(%), Conn(%), Time(ms)  — per dataset split + overall.
    """
    METHODS = ['EDT', 'Blum', 'EGF']
    REFS    = {'EDT': '[70]', 'Blum': '[1]', 'EGF': ''}
    SPLITS  = [
        ('NIST Digits (1,797)',  NIST),
        ('MPEG-7 Equiv (12,000)', MPEG),
        ('Overall (13,797)',      ALL),
    ]

    def stats(rows, m):
        q    = np.mean([r[f'q_{m}']  for r in rows])
        sp   = np.mean([r[f'sp_{m}'] for r in rows])
        w1   = np.mean([r[f'w1_{m}'] for r in rows]) * 100
        conn = np.mean([1 if r[f'nc_{m}'] == 1 else 0 for r in rows]) * 100
        ms   = np.mean([r[f'ms_{m}'] for r in rows])
        return q, sp, w1, conn, ms

    SEP  = '─' * 82
    SEP2 = '═' * 82

    print(f'\n{SEP2}')
    print('  TABLE 2.  Overall Results — EGF-MAT vs EDT vs Blum  '
          f'({len(ALL):,} images: {len(NIST):,} NIST + {len(MPEG):,} MPEG-7)')
    print(SEP2)

    hdr = f"  {'Method':<12} {'Q (%)':>8} {'Sp':>7} {'1px (%)':>9} {'Conn (%)':>10} {'Time (ms)':>11}"
    print(hdr)

    for split_name, rows in SPLITS:
        print(f'{SEP}')
        print(f'  ▶  {split_name}')
        print(f'  {"─"*78}')
        best_q  = max(stats(rows, m)[0] for m in METHODS)
        best_sp = min(stats(rows, m)[1] for m in METHODS)
        best_w1 = max(stats(rows, m)[2] for m in METHODS)
        best_cn = max(stats(rows, m)[3] for m in METHODS)

        for m in METHODS:
            q, sp, w1, conn, ms = stats(rows, m)
            ref   = REFS[m]
            label = f'{m}{ref}'
            # Star markers for best per column
            sq = ' ★' if abs(q  - best_q)  < 0.01 else '  '
            ss = ' ★' if abs(sp - best_sp) < 0.01 else '  '
            sw = ' ★' if abs(w1 - best_w1) < 0.01 else '  '
            sc = ' ★' if abs(conn - best_cn) < 0.01 else '  '
            print(f"  {label:<12} {q:>7.3f}{sq} {sp:>6.3f}{ss} {w1:>8.1f}{sw} "
                  f"{conn:>9.1f}{sc} {ms:>10.3f}")

    print(SEP2)

    # ── Advantage summary ─────────────────────────────────────────────────────
    q_e, sp_e, w1_e, cn_e, ms_e   = stats(ALL, 'EDT')
    q_b, sp_b, w1_b, cn_b, ms_b   = stats(ALL, 'Blum')
    q_g, sp_g, w1_g, cn_g, ms_g   = stats(ALL, 'EGF')

    print(f'\n  EGF vs Blum  │  ΔQ = +{q_g-q_b:.3f}pp  '
          f'│  ΔSp = −{sp_b-sp_g:.3f} ({(sp_b-sp_g)/sp_b*100:.1f}% fewer)  '
          f'│  ΔConn = +{cn_g-cn_b:.1f}pp  '
          f'│  Δ1px = +{w1_g-w1_b:.1f}pp')
    print(f'  EGF vs EDT   │  ΔQ = +{q_g-q_e:.3f}pp  '
          f'│  ΔSp = −{sp_e-sp_g:.3f} ({(sp_e-sp_g)/sp_e*100:.1f}% fewer)  '
          f'│  ΔConn = +{cn_g-cn_e:.1f}pp  '
          f'│  Δ1px = +{w1_g-w1_e:.1f}pp')
    print(f'  EGF time     │  {ms_g:.3f} ms/img  '
          f'({ms_g/ms_b:.1f}× Blum,  {ms_g/ms_e:.1f}× EDT)')
    print(f'{SEP2}\n')


# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════════════════════════════════════
print('='*60)
print('EGF Public Benchmark — Analysis Figures')
print(f'Dataset: {len(ALL):,} images  ({len(NIST):,} NIST + {len(MPEG):,} MPEG-7)')
print('='*60)

print('\n[1/4] Fig 1: MPEG-7 per-category Quality & Spurious ...')
make_fig1()

print('[2/4] Fig 2: NIST per-digit + Per-source + Conn/1px ...')
make_fig2()

print('[3/4] Fig 3: Summary table + Radar chart ...')
make_fig3()

print('[4/4] Fig 4: Complete per-source metrics grid (A-F) ...')
make_fig4()

print(f'\n{"="*60}')
print('All 4 figures saved to:')
for nm in ['fig_analysis_1.png','fig_analysis_2.png','fig_analysis_3.png','fig_analysis_4.png']:
    p = OUT / nm
    if p.exists():
        print(f'  {nm}  ({p.stat().st_size//1024} KB)')
print('='*60)

# ── Print benchmark results table ─────────────────────────────────────────────
print_results_table()
