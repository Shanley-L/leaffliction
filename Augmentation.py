#!/usr/bin/env python3
"""Data augmentation: six augmented copies of an image, or a balanced
data set when a directory is given."""
import argparse
import os
import shutil
import sys
from multiprocessing import Pool

import numpy as np

import leaf_utils as lu


def parse_args():
    parser = argparse.ArgumentParser(
        description='With an image: create the 6 augmentations (Flip, '
                    'Rotate, Skew, Shear, Crop, Distortion) next to it, '
                    'named <name>_<Type>.<ext>, and display them. With a '
                    'directory: balance every class (sub directory) up to '
                    'the size of the largest one.')
    parser.add_argument('path', help='image file or data set directory')
    parser.add_argument('-o', '--output', metavar='DIR',
                        help='(directory mode) write the balanced data set '
                             'in DIR (e.g. augmented_directory) instead of '
                             'modifying the data set in place')
    parser.add_argument('--seed', type=int, default=None,
                        help='seed of the random generator')
    return parser.parse_args()


def augment_file(path, seed):
    """Create the 6 augmentations of one image, return [(type, path, img)]."""
    img = lu.read_image(path)
    rng = np.random.default_rng(seed)
    directory, name = os.path.split(path)
    stem, ext = os.path.splitext(name)
    results = []
    for kind in lu.AUGMENTATIONS:
        new = lu.augment(img, kind, rng)
        target = lu.free_name(directory, stem, kind, ext)
        lu.write_image(target, new)
        results.append((kind, target, new))
    return img, results


def _job(task):
    """Worker: create one augmented image."""
    src, dst, kind, seed = task
    try:
        img = lu.read_image(src)
    except lu.LeafError:
        return False
    rng = np.random.default_rng(seed)
    lu.write_image(dst, lu.augment(img, kind, rng))
    return True


def balance(root, output, seed):
    """Balance the data set located in root (copied to output if given)."""
    lu.check_dir(root)
    if output:
        if os.path.exists(output):
            raise lu.LeafError('%s already exists' % output)
        shutil.copytree(root, output)
        root = output
    classes = lu.class_dirs(root, skip_augmented=True)
    if not classes:
        raise lu.LeafError('no image found in %s' % root)
    # images from previous runs count as members of their class
    for cdir in classes:
        classes[cdir] = [os.path.join(cdir, f)
                         for f in sorted(os.listdir(cdir))
                         if lu.is_image(f)]
    sources = {c: [f for f in fs if not lu.is_augmented(f)]
               for c, fs in classes.items()}
    target, tasks = plan_balance(classes, sources, seed)
    with Pool() as pool:
        done = sum(pool.imap_unordered(_job, tasks, chunksize=16))
    print('%d classes balanced to %d images (%d images created) in %s'
          % (len(classes), target, done, root))
    return root


def plan_balance(classes, sources, seed):
    """List the (src, dst, kind, seed) jobs balancing every class.

    classes: {dir: all images}, sources: {dir: original images}."""
    rng = np.random.default_rng(seed)
    target = max(len(v) for v in classes.values())
    kinds = list(lu.AUGMENTATIONS)
    tasks = []
    for cdir, files in classes.items():
        origins = sources[cdir]
        taken = set(os.listdir(cdir))
        for i in range(target - len(files)):
            src = origins[i % len(origins)]
            kind = kinds[int(rng.integers(len(kinds)))]
            stem, ext = os.path.splitext(os.path.basename(src))
            name = '%s_%s%s' % (stem, kind, ext)
            n = 1
            while name in taken:
                n += 1
                name = '%s_%s%d%s' % (stem, kind, n, ext)
            taken.add(name)
            tasks.append((src, os.path.join(cdir, name), kind,
                          int(rng.integers(2 ** 31))))
    return target, tasks


def main():
    args = parse_args()
    if os.path.isdir(args.path):
        balance(args.path, args.output, args.seed)
        return
    if not os.path.isfile(args.path):
        raise lu.LeafError('%s: no such file or directory' % args.path)
    img, results = augment_file(args.path, args.seed)
    for kind, target, _ in results:
        print('saved %s' % target)
    items = [('Original', img)] + [(k, im) for k, _, im in results]
    lu.show_images(items, title=os.path.basename(args.path), cols=4)


if __name__ == '__main__':
    try:
        main()
    except lu.LeafError as err:
        sys.exit('Error: %s' % err)
    except KeyboardInterrupt:
        sys.exit(130)
