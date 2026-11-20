#!/usr/bin/env python3
"""Predict the disease of a leaf with the learnings of train.py."""
import argparse
import os
import sys
import zipfile

import cv2
import numpy as np

import leaf_model as lm
import leaf_utils as lu


def parse_args():
    parser = argparse.ArgumentParser(
        description='Display the original and the transformed image of a '
                    'leaf and print its predicted disease. With a '
                    'directory, every image is classified and the '
                    'accuracy is computed from the sub directory names.')
    parser.add_argument('path', nargs='?',
                        help='image (or directory) to classify')
    parser.add_argument('-m', '--model', default=None, metavar='ZIP',
                        help='learnings archive (default: learnings.zip)')
    parser.add_argument('--validate', action='store_true',
                        help='compute the accuracy on the validation set '
                             'stored in the archive')
    args = parser.parse_args()
    if not args.path and not args.validate:
        parser.error('an image, a directory or --validate is required')
    return args


def find_model(given):
    if given:
        return given
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (os.getcwd(), here):
        cand = os.path.join(base, 'learnings.zip')
        if os.path.isfile(cand):
            return cand
    raise lu.LeafError('learnings.zip not found: run train.py first or '
                       'give --model')


def classify(learn, paths, batch=128):
    """Return a list of (class index, confidence) or None per path."""
    results = [None] * len(paths)
    todo, imgs = [], []
    for i, p in enumerate(paths):
        img = lm.preprocess_file(p)
        if img is not None:
            todo.append(i)
            imgs.append(img)
    for s in range(0, len(imgs), batch):
        idx, conf = learn.predict_batch(imgs[s:s + batch])
        for j, c, f in zip(todo[s:s + batch], idx, conf):
            results[j] = (c, f)
    return results


def report(learn, paths, truths):
    """Print the accuracy of the predictions against known classes."""
    res = classify(learn, paths)
    ok = total = 0
    for p, r, t in zip(paths, res, truths):
        if r is None:
            print('skipped %s' % p, file=sys.stderr)
            continue
        total += 1
        ok += learn.classes[r[0]] == t
    if total:
        print('accuracy: %.2f%% (%d/%d images)' % (ok * 100.0 / total, ok,
                                                   total))


def validate(learn, model_path):
    """Accuracy on the validation images stored in the archive."""
    if not learn.validation:
        raise lu.LeafError('no validation set in %s' % model_path)
    with zipfile.ZipFile(model_path) as zf:
        imgs, truths = [], []
        for item in learn.validation:
            data = np.frombuffer(zf.read(item['file']), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                continue
            imgs.append(lm.preprocess(img, learn.size))
            truths.append(item['class'])
    pred = []
    for s in range(0, len(imgs), 128):
        pred += learn.predict_batch(imgs[s:s + 128])[0]
    ok = sum(learn.classes[p] == t for p, t in zip(pred, truths))
    print('validation set: %d images, accuracy %.2f%%'
          % (len(truths), ok * 100.0 / len(truths)))


def predict_dir(learn, root):
    paths, truths = [], []
    for cur, imgs in lu.class_dirs(root).items():
        for p in imgs:
            paths.append(p)
            truths.append(os.path.basename(cur))
    if not paths:
        raise lu.LeafError('no image found in %s' % root)
    known = set(learn.classes)
    sel = [i for i, t in enumerate(truths) if t in known]
    if not sel:
        print('(directory names are not class names: no accuracy)')
        for p, r in zip(paths, classify(learn, paths)):
            if r:
                print('%s: %s (%.1f%%)' % (p, learn.classes[r[0]],
                                           r[1] * 100))
        return
    report(learn, [paths[i] for i in sel], [truths[i] for i in sel])


def predict_image(learn, path):
    img = lu.read_image(path)
    transformed = lm.preprocess(img, learn.size)
    idx, conf = learn.predict_batch([transformed])
    label = learn.classes[idx[0]]
    print('Class predicted: %s (confidence %.1f%%)' % (label,
                                                       conf[0] * 100))
    if not lu.can_display():
        return
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.5))
    for ax, (name, im) in zip(axes, (('Original', img),
                                     ('Transformed', transformed))):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(name)
        ax.axis('off')
    fig.suptitle('Class predicted: %s' % label, fontsize=14,
                 color='green')
    fig.tight_layout()
    plt.show()


def main():
    args = parse_args()
    model_path = find_model(args.model)
    if not os.path.isfile(model_path):
        raise lu.LeafError('%s: no such file' % model_path)
    learn = lm.Learnings(model_path)
    if args.validate:
        validate(learn, model_path)
    if not args.path:
        return
    if os.path.isdir(args.path):
        predict_dir(learn, args.path)
    elif os.path.isfile(args.path):
        predict_image(learn, args.path)
    else:
        raise lu.LeafError('%s: no such file or directory' % args.path)


if __name__ == '__main__':
    try:
        main()
    except lu.LeafError as err:
        sys.exit('Error: %s' % err)
    except KeyboardInterrupt:
        sys.exit(130)
