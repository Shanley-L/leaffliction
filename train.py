#!/usr/bin/env python3
"""Train a leaf disease classifier and save the learnings in a .zip."""
import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import zipfile
from multiprocessing import Pool

import numpy as np
import torch
from torch import nn

import Augmentation
import leaf_model as lm
import leaf_utils as lu


def parse_args():
    parser = argparse.ArgumentParser(
        description='Learn the diseases of the leaves found in DIRECTORY '
                    '(one sub directory per class). The images are '
                    'split in a training and a validation set, the '
                    'training set is balanced with data augmentation, '
                    'transformed (background removal) and used to train a '
                    'network. Learnings and images are saved in a .zip.')
    parser.add_argument('directory', help='data set (e.g. ./Apple/)')
    parser.add_argument('-o', '--output', default='learnings.zip',
                        help='archive to create (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='part of each class kept for validation')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 0.05 <= args.val_ratio <= 0.5:
        parser.error('--val-ratio must be between 0.05 and 0.5')
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    return args


def class_names(classes):
    """Unique class name of each directory."""
    names = {}
    for cdir in classes:
        name = os.path.basename(cdir)
        if name in names.values():
            name = '%s_%s' % (os.path.basename(os.path.dirname(cdir)), name)
        names[cdir] = name
    return names


def split(classes, ratio, seed):
    """Stratified split of the original images."""
    rng = random.Random(seed)
    train, val = {}, {}
    for cdir, files in classes.items():
        files = sorted(files)
        rng.shuffle(files)
        nval = min(max(1, round(len(files) * ratio)), len(files) - 1)
        val[cdir], train[cdir] = files[:nval], files[nval:]
    return train, val


def build_workdir(work, names, train, val, seed):
    """Copy the images, balance the training set. Returns the class list."""
    for cdir, name in names.items():
        for part, sets in (('train', train), ('validation', val)):
            dest = os.path.join(work, part, name)
            os.makedirs(dest)
            for f in sets[cdir]:
                shutil.copy2(f, dest)
    tr_classes = {os.path.join(work, 'train', n): sorted(
        os.path.join(work, 'train', n, os.path.basename(f))
        for f in train[c]) for c, n in names.items()}
    target, tasks = Augmentation.plan_balance(tr_classes, tr_classes, seed)
    with Pool() as pool:
        done = sum(pool.imap_unordered(Augmentation._job, tasks, 16))
    print('training set balanced to %d images per class (%d augmented)'
          % (target, done))


def load_set(work, part, classes):
    """Transform every image of work/part. Returns (images, labels)."""
    paths, labels = [], []
    for idx, name in enumerate(classes):
        cdir = os.path.join(work, part, name)
        for f in sorted(os.listdir(cdir)):
            paths.append(os.path.join(cdir, f))
            labels.append(idx)
    with Pool() as pool:
        imgs = pool.map(lm.preprocess_file, paths, chunksize=32)
    keep = [i for i, im in enumerate(imgs) if im is not None]
    return (np.stack([imgs[i] for i in keep]),
            np.array([labels[i] for i in keep]),
            [paths[i] for i in keep])


@torch.no_grad()
def evaluate(model, x, y, dev):
    model.eval()
    pred = []
    for i in range(0, len(x), 256):
        xb = lm.to_tensor(x[i:i + 256]).to(dev)
        pred.append(model(xb).argmax(1).cpu())
    pred = torch.cat(pred).numpy()
    return pred, float((pred == y).mean())


def fit(xtr, ytr, xva, yva, classes, epochs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = lm.device()
    model = lm.LeafNet(len(classes)).to(dev)
    xt = torch.from_numpy(xtr).to(dev)
    yt = torch.from_numpy(ytr).long().to(dev)
    bs = 64
    steps = (len(xt) + bs - 1) // bs
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=4e-3, total_steps=epochs * steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), device=dev)
        total = 0.0
        for i in range(steps):
            idx = perm[i * bs:(i + 1) * bs]
            xb = lm.to_tensor(xt[idx])
            if dev.type == 'cuda':
                xb = xb.to(dev)
            if random.random() < 0.5:
                xb = xb.flip(3)
            if random.random() < 0.5:
                xb = xb.flip(2)
            with torch.autocast(dev.type, enabled=dev.type == 'cuda'):
                loss = loss_fn(model(xb), yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        _, acc = evaluate(model, xva, yva, dev)
        print('epoch %2d/%d  loss %.4f  validation accuracy %.2f%%'
              % (epoch + 1, epochs, total / steps, acc * 100))
    return model


def write_zip(path, work, model_blob, classes, val_files):
    """Archive: model, class names, training and validation images."""
    if os.path.exists(path):
        os.remove(path)
    meta = [{'file': os.path.relpath(f, work), 'class': classes[c]}
            for f, c in val_files]
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(lm.MODEL_FILE, model_blob)
        zf.writestr('classes.json', json.dumps(classes, indent=1))
        zf.writestr('validation.json', json.dumps(meta, indent=1))
        for cur, dirs, files in os.walk(work):
            dirs.sort()
            for f in sorted(files):
                full = os.path.join(cur, f)
                zf.write(full, os.path.relpath(full, work),
                         compress_type=zipfile.ZIP_STORED)


def sha1(path):
    h = hashlib.sha1()
    with open(path, 'rb') as fd:
        for chunk in iter(lambda: fd.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    args = parse_args()
    lu.check_dir(args.directory)
    found = lu.class_dirs(args.directory, skip_augmented=True)
    if len(found) < 2:
        raise lu.LeafError('at least two classes (sub directories with '
                           'images) are needed in %s' % args.directory)
    names = class_names(found)
    classes = [names[c] for c in found]
    train, val = split(found, args.val_ratio, args.seed)
    nval = sum(len(v) for v in val.values())
    print('%d classes, %d images: %d for training, %d for validation'
          % (len(classes), sum(len(v) for v in found.values()),
             sum(len(v) for v in train.values()), nval))
    if nval < 100:
        print('warning: fewer than 100 validation images', file=sys.stderr)

    work = tempfile.mkdtemp(prefix='leaffliction_')
    try:
        build_workdir(work, names, train, val, args.seed)
        print('transforming the images...')
        xtr, ytr, _ = load_set(work, 'train', classes)
        xva, yva, vpaths = load_set(work, 'validation', classes)
        print('training on %s' % lm.device())
        model = fit(xtr, ytr, xva, yva, classes, args.epochs, args.seed)
        pred, acc = evaluate(model, xva, yva, lm.device())
        print('\nvalidation set: %d images, accuracy %.2f%%'
              % (len(yva), acc * 100))
        for idx, name in enumerate(classes):
            sel = yva == idx
            print('  %-30s %.2f%% (%d images)'
                  % (name, (pred[sel] == idx).mean() * 100, sel.sum()))
        if acc <= 0.9:
            print('warning: accuracy is not above 90%', file=sys.stderr)
        blob = lm.save_model(model, classes, lm.SIZE)
        val_files = [(p, y) for p, y in zip(vpaths, yva.tolist())]
        write_zip(args.output, work, blob, classes, val_files)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print('\nlearnings saved in %s' % args.output)
    print('%s  %s' % (sha1(args.output), args.output))


if __name__ == '__main__':
    try:
        main()
    except lu.LeafError as err:
        sys.exit('Error: %s' % err)
    except KeyboardInterrupt:
        sys.exit(130)
