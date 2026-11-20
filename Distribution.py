#!/usr/bin/env python3
"""Analyse a leaf data set: pie chart and bar chart for each plant type."""
import argparse
import os
import sys

import leaf_utils as lu


def parse_args():
    parser = argparse.ArgumentParser(
        description='Display a pie chart and a bar chart of the number '
                    'of images of each disease, for each plant type found '
                    'in DIRECTORY.')
    parser.add_argument('directory',
                        help='data set directory (e.g. ./Apple or ./images)')
    parser.add_argument('-o', '--output', metavar='DIR',
                        help='also save the charts as PNG files in DIR')
    return parser.parse_args()


def draw(plant, counts):
    """Build the figure of one plant type."""
    import matplotlib.pyplot as plt
    labels = list(counts.keys())
    values = list(counts.values())
    colors = plt.cm.tab10(range(len(labels)))
    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle('%s class distribution' % plant, fontsize=14)
    ax_pie.pie(values, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90)
    ax_pie.set_title('Share of each class')
    bars = ax_bar.bar(labels, values, color=colors)
    ax_bar.bar_label(bars)
    ax_bar.set_title('Number of images per class')
    ax_bar.set_ylabel('Images')
    ax_bar.tick_params(axis='x', rotation=30)
    for tick in ax_bar.get_xticklabels():
        tick.set_ha('right')
    fig.tight_layout()
    return fig


def main():
    args = parse_args()
    lu.check_dir(args.directory)
    groups = lu.group_by_plant(args.directory)
    if not groups:
        raise lu.LeafError('no image found in %s' % args.directory)

    import matplotlib
    if args.output is None and not lu.can_display():
        matplotlib.use('Agg')
        print('(no display available: charts not shown)', file=sys.stderr)
    import matplotlib.pyplot as plt

    if args.output:
        os.makedirs(args.output, exist_ok=True)
    for plant, counts in groups:
        print('%s: %d images' % (plant, sum(counts.values())))
        for name, num in counts.items():
            print('  %-30s %d' % (name, num))
        fig = draw(plant, counts)
        if args.output:
            fig.savefig(os.path.join(args.output, '%s_distribution.png'
                                     % plant), dpi=120)
    if lu.can_display():
        plt.show()


if __name__ == '__main__':
    try:
        main()
    except lu.LeafError as err:
        sys.exit('Error: %s' % err)
    except KeyboardInterrupt:
        sys.exit(130)
