#!/usr/bin/env python3
"""Draw the poster's data figures from the recorded runs (light backgrounds, print friendly).

    python3 scripts/make_figures.py            (from the poster/ folder; needs numpy + matplotlib)

  figures/mission_map.pdf   forest mission of 2026-11-24 (run claude_full1): path, frogs, position errors
  figures/drift.pdf         one recorded drive, three ways of estimating the robot's position
  figures/call.pdf          spectrogram of the simulated coqui call (ugv_audio.audio_core)

Every number comes from data/: nothing here is illustrative.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Polygon
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG, DATA = os.path.join(ROOT, 'figures'), os.path.join(ROOT, 'data')
ORANGE, TEAL, GRAY, INK, MUTED = '#F76902', '#007FA3', '#9AA3A8', '#1E1E1E', '#4A4A4A'
plt.rcParams.update({'font.family': ['TeX Gyre Heros', 'Nimbus Sans', 'Helvetica', 'Arial', 'DejaVu Sans'],
                     'pdf.fonttype': 3, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': 1.2, 'axes.edgecolor': '#8A8A8A', 'text.color': INK,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Frogs of run claude_full1 as graded by eval_survey: recorded position, label, horizontal error.
FROGS = [(40.52, -2.51, 'in a bush', '3 cm'), (37.50, -9.28, 'open glade', '22 cm'),
         (35.60, -2.52, 'tall grass', '13 cm'), (24.36, -5.90, 'far west', '37 cm'),
         (47.33, -9.18, 'up a tree', '7 cm'), (39.38, 8.49, 'behind a trunk', '3 cm')]


def mission_map(w=8.0, h=5.35, band=0.42):
    s = json.load(open(os.path.join(DATA, 'forest_shapes.json')))
    tr = np.loadtxt(os.path.join(DATA, 'path_forest_mission.csv'), delimiter=',')
    x0, x1, yc = 8.9, 54.9, 1.0
    half = (x1 - x0) * (h - band) / w / 2
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, band / h, 1, 1 - band / h])
    ax.set_xlim(x0, x1), ax.set_ylim(yc - half, yc + half), ax.set_aspect('equal'), ax.axis('off')
    ax.add_patch(Polygon(s['trail'], fc='#F1E4D0', ec='none', zorder=1))
    for g in s['grass']:
        ax.add_patch(Polygon(g, fc='#DDEBCF', ec='none', zorder=2))
    for b in s['bush']:
        ax.add_patch(Polygon(b, fc='#B9D6AC', ec='none', zorder=3, joinstyle='round'))
    for t in s['trunk']:
        ax.add_patch(Circle(t, 0.36, fc='#A9825F', ec='white', lw=0.8, zorder=4))
    ax.plot(tr[:, 1], tr[:, 2], color='white', lw=5.6, solid_capstyle='round', zorder=5)
    ax.plot(tr[:, 1], tr[:, 2], color=ORANGE, lw=3.2, solid_capstyle='round', solid_joinstyle='round', zorder=6)
    off = {'in a bush': (1.25, 0.0, 'left'), 'open glade': (-1.2, -1.0, 'right'), 'tall grass': (-1.2, 0.5, 'right'),
           'far west': (0.0, -2.4, 'center'), 'up a tree': (0.0, 2.3, 'center'), 'behind a trunk': (-1.2, 0.8, 'right')}
    box = dict(boxstyle='round,pad=0.10', fc='white', ec='none', alpha=0.8)
    for x, y, name, err in FROGS:
        ax.add_patch(Circle((x, y), 0.85, fc=TEAL, ec='white', lw=2.4, zorder=8))
        dx, dy, ha = off[name]
        ax.text(x + dx, y + dy + 0.72, name, ha=ha, va='center', zorder=9, fontsize=16.5, fontweight='bold', bbox=box)
        ax.text(x + dx, y + dy - 0.72, f'{err} off', ha=ha, va='center', zorder=9, fontsize=16.5, bbox=box)
    ax.plot(48, 2.1, marker='s', ms=17, color=ORANGE, mec='white', mew=2.4, zorder=9)
    ax.text(49.1, 2.1, 'start =\nfinish', va='center', ha='left', fontsize=16.5, fontweight='bold', zorder=9, bbox=box)
    # key below the map: text in ink, colour carried by the marks
    k = fig.add_axes([0, 0, 1, band / h])
    k.set_xlim(0, w), k.set_ylim(0, band), k.axis('off')
    y = band * 0.42
    for kx, c, r, lab in ((1.62, '#B9D6AC', 0.12, 'bush'), (2.52, '#DDEBCF', 0.12, 'tall grass'),
                          (3.95, '#A9825F', 0.085, 'trunk'), (6.45, TEAL, 0.14, 'frog found')):
        k.add_patch(Circle((kx, y), r, fc=c, ec='none'))
        k.text(kx + 0.22, y, lab, va='center', fontsize=15.5)
    k.plot([4.95, 5.35], [y, y], color=ORANGE, lw=3.6)
    k.text(5.45, y, 'robot', va='center', fontsize=15.5)
    m = 5.0 * w / (x1 - x0)
    k.plot([0.15, 0.15 + m], [y - 0.1, y - 0.1], color=INK, lw=3)
    k.text(0.15 + m / 2, y - 0.04, '5 m', ha='center', va='bottom', fontsize=14)
    fig.savefig(os.path.join(FIG, 'mission_map.pdf'), facecolor='white')
    plt.close(fig)


def drift(w=6.4, h=5.35):
    d = np.loadtxt(os.path.join(DATA, 'drift_same_drive.csv'), delimiter=',', skiprows=2)
    x = d[:, 1]
    fig, ax = plt.subplots(figsize=(w, h))
    for col, c, lw in ((2, GRAY, 3.0), (3, TEAL, 3.4), (4, ORANGE, 4.6)):
        ax.plot(x, d[:, col], color=c, lw=lw, solid_capstyle='round', solid_joinstyle='round', zorder=3 + col)
    ax.set_xlim(0, 146), ax.set_ylim(0, 5)
    ax.set_xlabel('distance driven (m)', fontsize=18.5), ax.set_ylabel('position error (m)', fontsize=18.5)
    ax.tick_params(labelsize=17, length=0)
    ax.set_xticks([0, 50, 100, 140]), ax.set_yticks(range(6))
    ax.grid(axis='y', color='#E3E3E3', lw=1.0), ax.set_axisbelow(True)
    k = dict(fontsize=18, va='center', ha='left', zorder=20, bbox=dict(boxstyle='round,pad=0.08', fc='white', ec='none'))
    for y, c, lw, name, val in ((4.72, GRAY, 3.0, 'flat-world odometry', 'up to 4.7 m'),
                                (3.86, TEAL, 3.4, '3D odometry', 'up to 1.2 m'),
                                (3.00, ORANGE, 4.6, '+ tree matching', 'up to 0.4 m')):
        ax.plot([4, 14], [y, y], color=c, lw=lw + 1, solid_capstyle='round', zorder=20)
        ax.text(18, y, name, fontweight='bold', **k)
        ax.text(18, y - 0.37, val, **k)
    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(FIG, 'drift.pdf'), facecolor='white')
    plt.close(fig)
    print('drift max (m): 2D %.2f  3D %.2f  +trees %.2f' % tuple(d[:, 2:5].max(0)))


def call(w=8.0, h=3.2):
    sys.path.insert(0, os.path.join(ROOT, '..', 'src', 'ugv_main', 'ugv_audio'))
    from ugv_audio import audio_core as ac
    fs = ac.SAMPLE_RATE
    n = ac.FRAME_SAMPLES
    sig = ac.synthesize_coqui_call_frame(n, fs)
    sig = sig + np.random.default_rng(0).normal(0, 0.012, n)
    win, hop, nfft = 256, 8, 2048
    pad = np.pad(sig, win // 2, mode='reflect')
    S = np.array([np.abs(np.fft.rfft(pad[i:i + win] * np.hanning(win), nfft)) for i in range(0, n, hop)]).T
    db = 20 * np.log10(S / S.max() + 1e-6)
    f = np.fft.rfftfreq(nfft, 1 / fs) / 1000.0
    m = (f > 0.6) & (f < 2.8)
    cm = LinearSegmentedColormap.from_list('w', ['#FFFFFF', '#FEE3CC', '#FBA463', ORANGE, '#B04A00'])
    fig, ax = plt.subplots(figsize=(w, h))
    ax.imshow(db[m], origin='lower', aspect='auto', extent=[0, 1000 * n / fs, f[m][0], f[m][-1]], cmap=cm, vmin=-34, vmax=0,
              interpolation='bicubic')
    ax.set_yticks([1, 2]), ax.set_yticklabels(['1 kHz', '2 kHz']), ax.set_xticks([])
    ax.tick_params(labelsize=21, length=0)
    ax.text(28, 1.48, '“co”', ha='center', va='center', fontsize=38, fontweight='bold')
    ax.text(74, 1.48, '“quí!”', ha='center', va='center', fontsize=38, fontweight='bold')
    ax.set_xlabel('one call in the simulator: about 0.1 s', fontsize=20)
    ax.spines['left'].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIG, 'call.pdf'), facecolor='white', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    mission_map()
    drift()
    call()
