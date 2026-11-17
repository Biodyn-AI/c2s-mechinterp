#!/usr/bin/env python3
"""Draw vector schematics for the manuscript and graphical abstract.

The icons illustrate the analysis design; their counts and coordinates are not
experimental measurements. All output paths are relative to this source file.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

OUTPUT = Path(__file__).resolve().parent / 'figures'
COLORS = {
    'ink': '#263747', 'muted': '#657786', 'line': '#c1cbd3',
    'blue': '#3277ad', 'purple': '#7565ad', 'teal': '#228777',
    'coral': '#c35b4c', 'gold': '#b18530', 'pale': '#f5f8fa',
}
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
})


def label(ax, x, y, words, size=8, color='ink', weight='normal', **kwargs):
    return ax.text(x, y, words, ha='center', va='center', fontsize=size,
                   color=COLORS.get(color, color), fontweight=weight, **kwargs)


def rounded(ax, x, y, width, height, color='line', fill='white', radius=.012, lw=.8):
    patch = FancyBboxPatch((x, y), width, height,
                          boxstyle=f'round,pad=0,rounding_size={radius}',
                          edgecolor=COLORS.get(color, color),
                          facecolor=COLORS.get(fill, fill), linewidth=lw)
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color='muted', width=.9, head=8, **kwargs):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
                                mutation_scale=head, linewidth=width,
                                color=COLORS.get(color, color), **kwargs))


def cell(ax, x, y):
    # Expression bars sit inside a cell outline: no measured profile is implied.
    ax.add_patch(Ellipse((x, y), .115, .17, fc='#edf5fb', ec=COLORS['blue'], lw=1.2))
    ax.add_patch(Ellipse((x-.026, y+.045), .025, .04, fc='#c9deef', ec='none'))
    for i, height in enumerate([.063, .046, .035, .023]):
        ax.add_patch(Rectangle((x-.034+i*.019, y-.047), .012, height,
                               facecolor=COLORS['blue'], edgecolor='none'))


def gene_list(ax, x, y, size):
    for i, gene in enumerate(['ACTB', 'GAPDH', 'RPLP0']):
        label(ax, x, y+.056-i*.052, gene, size=size,
              color='blue', family='DejaVu Sans Mono')
    arrow(ax, (x+.057, y+.073), (x+.057, y-.075), 'blue', head=5)


def tokens(ax, x, y):
    # Blocks depict sub-word pieces without asserting a particular segmentation.
    rows = [(.015, .035, .023), (.027, .020), (.018, .025, .018)]
    for row, widths in enumerate(rows):
        left = x-sum(widths)/2-(len(widths)-1)*.003/2
        for j, width in enumerate(widths):
            rounded(ax, left, y+.043-row*.052, width, .032,
                    color='blue', fill=['#c9deef', '#e4edf5', '#accce5'][j % 3],
                    radius=.003, lw=.5)
            left += width+.003


def transformer(ax, x, y):
    for row in range(4):
        rounded(ax, x-.054, y-.071+row*.044, .108, .032,
                color='purple', fill='#f0edf7', radius=.005)
    # The vertical line denotes the residual stream; a horizontal tap is pooled.
    ax.plot([x, x], [y-.093, y+.106], color=COLORS['purple'], lw=2.0)
    arrow(ax, (x+.006, y+.004), (x+.137, y+.004), 'purple', width=1.3, head=7)
    ax.add_patch(Circle((x, y+.004), .006, color=COLORS['purple']))


def dictionary_icon(ax, x, y):
    for row in range(3):
        for col in range(5):
            active = (row, col) in {(0, 1), (1, 3), (2, 0)}
            rounded(ax, x-.058+col*.025, y+.052-row*.048, .018, .032,
                    color='teal' if active else 'line',
                    fill='teal' if active else '#f0f4f4', radius=.003, lw=.6)


def atlas_icon(ax, x, y):
    center = (x-.016, y+.004)
    for dx, dy in [(-.046, .036), (-.046, -.035), (.03, .045)]:
        ax.plot([center[0], x+dx], [center[1], y+dy], color=COLORS['teal'], lw=.8)
        ax.add_patch(Circle((x+dx, y+dy), .008, fc='#cfe7e1', ec=COLORS['teal'], lw=.6))
    ax.add_patch(Circle(center, .012, fc=COLORS['teal'], ec='none'))
    rounded(ax, x+.024, y-.042, .064, .034, color='teal', fill='#edf7f4', radius=.006)
    label(ax, x+.056, y-.025, 'GO', size=7, color='teal')
    ax.plot([center[0]+.012, x+.024], [center[1], y-.025], color=COLORS['teal'], lw=.8)


def circuit_icon(ax, x, y):
    for column in [-1, 0, 1]:
        for row in [-1, 1]:
            ax.add_patch(Circle((x+column*.056, y+row*.035), .010,
                                fc='white', ec=COLORS['purple'], lw=.9))
    for start, end in [((-.046, .035), (-.010, .035)),
                       ((.010, .035), (.046, -.035)),
                       ((-.046, -.035), (-.010, .035))]:
        arrow(ax, (x+start[0], y+start[1]), (x+end[0], y+end[1]), 'purple', head=5)
    ax.plot([x-.066, x-.046], [y+.020, y+.050], color=COLORS['coral'], lw=1.3)
    ax.plot([x-.066, x-.046], [y+.050, y+.020], color=COLORS['coral'], lw=1.3)


def steering_icon(ax, x, y):
    cloud = np.array([[-.055, -.033], [-.045, .015], [-.026, -.008],
                      [-.010, -.044], [.012, -.006], [.022, .036],
                      [.040, .020], [.053, .045], [.060, -.020]])
    ax.scatter(x+cloud[:, 0], y+cloud[:, 1], s=7, c='#cbd5dc', linewidths=0)
    arrow(ax, (x-.049, y-.039), (x+.055, y+.047), 'gold', width=1.8, head=8,
          connectionstyle='arc3,rad=-.18')
    ax.add_patch(Circle((x-.049, y-.039), .007, fc=COLORS['blue'], ec='none'))
    ax.add_patch(Circle((x+.055, y+.047), .009, fc='white', ec=COLORS['gold'], lw=1))


def browser_icon(ax, x, y):
    rounded(ax, x-.073, y-.057, .146, .115, color='teal', radius=.005)
    ax.plot([x-.073, x+.073], [y+.032, y+.032], color=COLORS['teal'], lw=.7)
    for i in range(3):
        ax.add_patch(Circle((x-.059+i*.012, y+.045), .0035, fc=COLORS['teal'], ec='none'))
    ax.plot([x-.031, x-.031], [y-.057, y+.032], color=COLORS['line'], lw=.6)
    for i in range(3):
        ax.plot([x-.062, x-.043], [y+.009-i*.021]*2, color=COLORS['line'], lw=1.5)
    for row in range(2):
        for col in range(3):
            rounded(ax, x-.018+col*.027, y-.041+row*.030, .020, .021,
                    color='line', fill='#e1f0eb', radius=.002, lw=.4)


def draw_schematic(stem, graphical=False):
    height = 3.15 if graphical else 4.0
    fig, ax = plt.subplots(figsize=(7.2, height), dpi=300)
    fig.subplots_adjust(left=.012, right=.988, bottom=.018, top=.982)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    top = .766 if graphical else .742
    title_y = .956
    font = 7.5 if graphical else 8.3
    title = 'C2S-Scale: from cell sentences to controlled tests'
    label(ax, .5, title_y, title, size=10.3 if graphical else 11.0, weight='bold')

    # A single left-to-right path makes tokenization and the residual tap explicit.
    positions = [.075, .260, .425, .620, .875]
    cell(ax, positions[0], top)
    gene_list(ax, positions[1], top, font)
    tokens(ax, positions[2], top)
    transformer(ax, positions[3], top)
    dictionary_icon(ax, positions[4], top)
    for start, end in [(.143, .194), (.326, .366), (.483, .554)]:
        arrow(ax, (start, top+.004), (end, top+.004), head=7)
    arrow(ax, (.759, top+.004), (.802, top+.004), 'purple', width=1.2, head=7)
    label(ax, .728, top+.061, 'Mean pool', size=6.8, color='purple')
    label(ax, .618, top+.137, 'Residual stream', size=6.8, color='purple')
    label(ax, .875, top+.137, 'TopK SAE', size=6.8, color='teal')
    for x, words in zip(positions, ['Expression', 'Ranked genes', 'Tokens', 'Transformer', 'Sparse features']):
        label(ax, x, top-.139, words, size=font, weight='bold')

    # One branch rail leads from the sparse feature representation to four uses.
    rail_y = .486 if graphical else .482
    centers = [.128, .376, .624, .872]
    ax.plot([.875, .875], [top-.166, rail_y], color=COLORS['line'], lw=.9)
    ax.plot([centers[0], centers[-1]], [rail_y, rail_y], color=COLORS['line'], lw=.9)
    card_bottom = .164
    card_height = .259 if graphical else .274
    cards = [('Feature atlas', atlas_icon, 'teal'),
             ('Causal circuits', circuit_icon, 'purple'),
             ('Geometry + steering', steering_icon, 'gold'),
             ('Public atlas', browser_icon, 'teal')]
    for x, (title, icon, color) in zip(centers, cards):
        rounded(ax, x-.112, card_bottom, .224, card_height, fill='pale', color='line', lw=.65)
        arrow(ax, (x, rail_y), (x, card_bottom+card_height+.006), color='line', head=5)
        label(ax, x, card_bottom+card_height-.046, title, size=font-.1, weight='bold')
        icon(ax, x, card_bottom+.096)

    # These three controls describe comparisons, not the outcomes of experiments.
    controls = ['Input-matched baselines', 'Co-expression nulls', 'Matched feature capacity']
    for x, words in zip([.172, .5, .828], controls):
        rounded(ax, x-.155, .035, .31, .067, color='blue', fill='#edf4fa', radius=.012, lw=.65)
        label(ax, x, .069, words, size=font-.4, color='blue', weight='bold')
    OUTPUT.mkdir(exist_ok=True)
    for extension in ['pdf', 'svg', 'png']:
        fig.savefig(OUTPUT/f'{stem}.{extension}', dpi=300, facecolor='white')
    if graphical:
        # Keep the existing submission-format TIFF consistent with vector outputs.
        fig.savefig(OUTPUT/f'{stem}.tiff', dpi=300, facecolor='white',
                    pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    print(stem, 'exported as PDF, SVG, PNG' + (' and TIFF' if graphical else ''))


if __name__ == '__main__':
    draw_schematic('Figure_1_schematic')
    draw_schematic('graphical_abstract', graphical=True)
