#!/usr/bin/env python3
"""Image transformations of a leaf, based on PlantCV."""
import argparse
import os
import sys
from multiprocessing import Pool

import leaf_utils as lu

SAVED = [n for n in lu.TRANSFORM_NAMES if n != 'Original']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Image transformations (Gaussian blur, Mask, Roi '
                    'objects, Analyze object, Pseudolandmarks, Color '
                    'histogram). With an IMAGE they are displayed. With '
                    '-src and -dst the transformations of every image of '
                    'the source directory are saved in the destination '
                    'directory.',
        epilog='examples:\n'
               '  ./Transformation.py "Apple/apple_healthy/image (1).JPG"\n'
               '  ./Transformation.py -src Apple/apple_healthy/ '
               '-dst dst_directory -mask',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('image', nargs='?',
                        help='image to transform and display')
    parser.add_argument('-src', metavar='DIR',
                        help='source directory containing the images')
    parser.add_argument('-dst', metavar='DIR',
                        help='destination directory of the transformed '
                             'images (created if missing)')
    parser.add_argument('-mask', action='store_true',
                        help='save only the masked images (leaf on a white '
                             'background) instead of every transformation')
    parser.add_argument('--original', action='store_true',
                        help='also copy the original image in DST')
    args = parser.parse_args()
    if args.image and (args.src or args.dst):
        parser.error('give either an image or -src/-dst, not both')
    if bool(args.src) != bool(args.dst):
        parser.error('-src and -dst must be used together')
    if not args.image and not args.src:
        parser.error('an image or -src/-dst is required')
    return args


def process_file(job):
    """Transform one image and save the results in the destination."""
    path, dst, mask_only, keep = job
    try:
        img = lu.read_image(path)
        out = lu.transformations(img)
    except (lu.LeafError, Exception) as err:  # noqa: BLE001
        return '%s: %s' % (path, err)
    stem, ext = os.path.splitext(os.path.basename(path))
    ext = ext if ext.lower() in ('.jpg', '.jpeg', '.png') else '.JPG'
    names = ['Mask'] if mask_only else SAVED
    if keep:
        names = ['Original'] + names
    for name in names:
        lu.write_image(os.path.join(dst, '%s_%s%s' % (stem, name, ext)),
                       out[name])
    return None


def transform_directory(args):
    lu.check_dir(args.src)
    files = []
    for cur, dirs, names in os.walk(args.src):
        dirs.sort()
        files += [os.path.join(cur, n) for n in sorted(names)
                  if lu.is_image(n) and not lu.is_augmented(n)]
    if not files:
        raise lu.LeafError('no image found in %s' % args.src)
    os.makedirs(args.dst, exist_ok=True)
    jobs = [(f, args.dst, args.mask, args.original) for f in files]
    with Pool() as pool:
        errors = [e for e in pool.imap_unordered(process_file, jobs, 8)
                  if e]
    for err in errors:
        print('skipped ' + err, file=sys.stderr)
    print('%d image(s) transformed in %s' % (len(files) - len(errors),
                                             args.dst))


def show_image(path):
    img = lu.read_image(path)
    out = lu.transformations(img)
    items = [(n.replace('_', ' '), out[n]) for n in lu.TRANSFORM_NAMES]
    if not lu.can_display():
        print('transformations computed: ' + ', '.join(
            n for n, _ in items), file=sys.stderr)
        return
    lu.show_images(items[:-1], title=os.path.basename(path), cols=3,
                   block=False)
    lu.show_images(items[-1:], cols=1)


def main():
    args = parse_args()
    if args.image:
        if not os.path.isfile(args.image):
            raise lu.LeafError('%s: no such file' % args.image)
        show_image(args.image)
    else:
        transform_directory(args)


if __name__ == '__main__':
    try:
        main()
    except lu.LeafError as err:
        sys.exit('Error: %s' % err)
    except KeyboardInterrupt:
        sys.exit(130)
