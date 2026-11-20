"""Neural network, preprocessing and learnings archive of Leaffliction."""
import io
import json
import zipfile

import cv2
import numpy as np
import torch
from torch import nn

import leaf_utils as lu

MODEL_FILE = 'model.pt'
SIZE = 128


def preprocess(img, size=SIZE):
    """Transformed image: leaf isolated on a white background, resized.

    Returns the transformed image (BGR, size x size)."""
    mask = lu.leaf_mask(img)
    out = lu.apply_mask(img, mask)
    return cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)


def preprocess_file(path):
    """Worker friendly version: path -> transformed image or None."""
    try:
        return preprocess(lu.read_image(path))
    except (lu.LeafError, cv2.error):
        return None


def to_tensor(batch):
    """uint8 BGR (N, H, W, 3) array/tensor -> normalised float tensor."""
    t = torch.as_tensor(batch)
    t = t.permute(0, 3, 1, 2).flip(1)  # NHWC BGR -> NCHW RGB
    return (t.float() / 255.0 - 0.5) / 0.5


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2))


class LeafNet(nn.Module):
    """Small convolutional network."""

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            _block(3, 32), _block(32, 64), _block(64, 128),
            _block(128, 256), _block(256, 256))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(256, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_model(model, classes, size):
    buf = io.BytesIO()
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({'state': state, 'classes': classes, 'size': size}, buf)
    return buf.getvalue()


class Learnings:
    """The model stored in a learnings .zip archive."""

    def __init__(self, path):
        try:
            with zipfile.ZipFile(path) as zf:
                blob = zf.read(MODEL_FILE)
                self.validation = []
                if 'validation.json' in zf.namelist():
                    self.validation = json.loads(
                        zf.read('validation.json'))
        except (OSError, KeyError, zipfile.BadZipFile) as err:
            raise lu.LeafError('invalid learnings archive %s (%s)'
                               % (path, err))
        data = torch.load(io.BytesIO(blob), map_location='cpu',
                          weights_only=True)
        self.classes = data['classes']
        self.size = data['size']
        self.dev = device()
        self.model = LeafNet(len(self.classes))
        self.model.load_state_dict(data['state'])
        self.model.to(self.dev).eval()
        self.path = path

    @torch.no_grad()
    def predict_batch(self, transformed):
        """List of transformed images -> (indices, probabilities)."""
        x = to_tensor(np.stack(transformed)).to(self.dev)
        prob = torch.softmax(self.model(x), dim=1).cpu()
        conf, idx = prob.max(dim=1)
        return idx.tolist(), conf.tolist()
