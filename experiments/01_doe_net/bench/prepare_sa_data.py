"""Reproduce the Sequential Attention (Yasuda et al., ICLR 2023) data pipeline with numpy only.

Reference code (google-research/google-research, sequential_attention/experiments):
  datasets/data_loader.py  -- per-dataset loaders (adapted by the SA authors from LassoNet)
  datasets/dataset.py      -- train/val split: idx = np.random.permutation(train), val = last 12.5 %
  experiments.py           -- seeds [1, 2, 3, 4, 5], val_ratio 0.125, k = 50

The numpy random call order of the reference is reproduced (np.random.seed(seed) ->
loader-internal shuffle for mice/coil -> permutation for the val split), so the
train/val/test splits match the reference implementation for each seed.
Known quirks kept on purpose for comparability with the published numbers:
  * Mice: missing values are filled with -100000 and then replaced by a class-wise mean
    that still includes other -100000 entries (pointed out by IP-CAE, ICML 2024).
  * ISOLET / Activity / MNIST / Fashion / COIL: min-max scaling fit on train+test.

Output: <out_dir>/<name>_seed<s>.npz with X_train, y_train, X_val, y_val, X_test, y_test
and <out_dir>/manifest.json (raw-file sha256, shapes, library versions).

Usage: python bench/prepare_sa_data.py --out_dir DATA_DIR [--seeds 1 2 3 4 5]
"""

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import PIL
import sklearn
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

URLS = {
    "mice": ["https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls"],
    "isolet": ["https://archive.ics.uci.edu/ml/machine-learning-databases/isolet/isolet1+2+3+4.data.Z",
               "https://archive.ics.uci.edu/ml/machine-learning-databases/isolet/isolet5.data.Z"],
    "activity": ["https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"],
    "coil20": ["https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip"],
    # identical to tf.keras.datasets.{mnist,fashion_mnist}.load_data()
    "mnist": ["https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"],
    "fashion": [f"https://storage.googleapis.com/tensorflow/tf-keras-datasets/{f}" for f in
                ["train-labels-idx1-ubyte.gz", "train-images-idx3-ubyte.gz",
                 "t10k-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz"]],
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(raw_dir, name):
    d = os.path.join(raw_dir, name)
    os.makedirs(d, exist_ok=True)
    paths = []
    for url in URLS[name]:
        p = os.path.join(d, urllib.request.unquote(url.rsplit("/", 1)[1]))
        if not os.path.exists(p):
            urllib.request.urlretrieve(url, p)
        paths.append(p)
    # unpack
    if name == "mice":
        csv = os.path.join(d, "Data_Cortex_Nuclear.csv")
        if not os.path.exists(csv):  # == xls2csv.py
            pd.read_excel(paths[0]).to_csv(csv, index=None, header=True)
    if name == "isolet":
        for p in paths:
            if not os.path.exists(p[:-2]):
                subprocess.run(["uncompress", "-k", "-f", p], check=True)
    if name in ("activity", "coil20"):
        with zipfile.ZipFile(paths[0]) as z:
            z.extractall(d)
    return d, {os.path.basename(p): sha256(p) for p in paths}


# ---------------------------------------------------------------- loaders (== data_loader.py)
def load_mice(d):
    filling_value = -100000
    path = os.path.join(d, "Data_Cortex_Nuclear.csv")
    with open(path) as fp:
        x = np.genfromtxt(fp.readlines(), delimiter=",", skip_header=1, usecols=range(1, 78),
                          filling_values=filling_value, encoding="UTF-8")
    with open(path) as fp:
        classes = np.genfromtxt(fp.readlines(), delimiter=",", skip_header=1, usecols=range(78, 81),
                                dtype=None, encoding="UTF-8")
    for i, row in enumerate(x):
        for j, val in enumerate(row):
            if val == filling_value:
                x[i, j] = np.mean([x[k, j] for k in range(classes.shape[0])
                                   if np.all(classes[i] == classes[k])])
    y = np.zeros((classes.shape[0]), dtype=np.uint8)
    for i, row in enumerate(classes):
        for j, (val, label) in enumerate(zip(row, ["Control", "Memantine", "C/S"])):
            y[i] += (2 ** j) * (val == label)
    x = MinMaxScaler(feature_range=(0, 1)).fit_transform(x)
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x, y = x[indices], y[indices]
    n = x.shape[0] * 4 // 5
    return x[:n], x[n:], y[:n].astype(np.int64), y[n:].astype(np.int64)


def load_isolet(d):
    tr = np.genfromtxt(os.path.join(d, "isolet1+2+3+4.data"), delimiter=",")
    te = np.genfromtxt(os.path.join(d, "isolet5.data"), delimiter=",")
    x_train, y_train = tr[:, :617], tr[:, 617]
    x_test, y_test = te[:, :617], te[:, 617]
    x = MinMaxScaler(feature_range=(0, 1)).fit_transform(np.concatenate((x_train, x_test)))
    return (x[:len(y_train)], x[len(y_train):],
            (y_train - 1).astype(np.int64), (y_test - 1).astype(np.int64))


def load_activity(d):
    base = os.path.join(d, "UCI HAR Dataset")
    x_train = np.genfromtxt(os.path.join(base, "train", "X_train.txt"))
    x_test = np.genfromtxt(os.path.join(base, "test", "X_test.txt"))
    y_train = np.genfromtxt(os.path.join(base, "train", "y_train.txt"))
    y_test = np.genfromtxt(os.path.join(base, "test", "y_test.txt"))
    x = MinMaxScaler(feature_range=(0, 1)).fit_transform(np.concatenate((x_train, x_test)))
    return (x[:len(y_train)], x[len(y_train):],
            (y_train - 1).astype(np.int64), (y_test - 1).astype(np.int64))


def load_coil(d):
    samples = []
    for i in range(1, 21):
        for image_index in range(72):
            fn = os.path.join(d, "coil-20-proc", "obj%d__%d.png" % (i, image_index))
            with Image.open(fn) as img:
                rescaled = img.resize((20, 20))
                pixels = [float(v) for v in list(rescaled.getdata())]
            samples.append(np.array(pixels + [i]))
    samples = np.array(samples)
    np.random.shuffle(samples)
    data = samples[:, :-1]
    targets = (samples[:, -1] + 0.5).astype(np.int64)
    data = (data - data.min()) / (data.max() - data.min())
    n = data.shape[0] * 4 // 5
    return data[:n], data[n:], targets[:n] - 1, targets[n:] - 1


def _mnist_like(x_train, y_train, x_test, y_test):
    x_train = x_train.reshape(len(x_train), -1).astype(np.float32)
    x_test = x_test.reshape(len(x_test), -1).astype(np.float32)
    x = np.concatenate((x_train, x_test))
    x = (x - x.min()) / (x.max() - x.min())
    return x[:len(y_train)], x[len(y_train):], y_train.astype(np.int64), y_test.astype(np.int64)


def load_mnist(d):
    with np.load(os.path.join(d, "mnist.npz")) as f:
        return _mnist_like(f["x_train"], f["y_train"], f["x_test"], f["y_test"])


def load_fashion(d):
    def lab(fn):
        with gzip.open(os.path.join(d, fn), "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=8)

    def img(fn, n):
        with gzip.open(os.path.join(d, fn), "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=16).reshape(n, 28, 28)
    y_train, y_test = lab("train-labels-idx1-ubyte.gz"), lab("t10k-labels-idx1-ubyte.gz")
    x_train = img("train-images-idx3-ubyte.gz", len(y_train))
    x_test = img("t10k-images-idx3-ubyte.gz", len(y_test))
    return _mnist_like(x_train, y_train, x_test, y_test)


LOADERS = {"mice": load_mice, "isolet": load_isolet, "activity": load_activity,
           "coil20": load_coil, "mnist": load_mnist, "fashion": load_fashion}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--datasets", default="mice,mnist,fashion,isolet,coil20,activity")
    ap.add_argument("--val_ratio", type=float, default=0.125)
    args = ap.parse_args()
    raw_dir = os.path.join(args.out_dir, "raw")
    manifest = {"reference": "google-research/sequential_attention experiments (data_loader.py, dataset.py, experiments.py)",
                "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                             "pillow": PIL.__version__, "scikit-learn": sklearn.__version__},
                "val_ratio": args.val_ratio, "seeds": args.seeds, "datasets": {}}
    for name in args.datasets.split(","):
        d, hashes = fetch(raw_dir, name)
        info = {"raw_sha256": hashes, "splits": {}}
        for seed in args.seeds:
            np.random.seed(seed)                      # == run.py main()
            x_tr, x_te, y_tr, y_te = LOADERS[name](d)  # loader-internal shuffle (mice, coil)
            idx = np.random.permutation(len(x_tr))     # == dataset.py get_dataset()
            val_size = int(np.size(idx) * args.val_ratio)
            va, tr = idx[-val_size:], idx[:-val_size]
            out = os.path.join(args.out_dir, f"{name}_seed{seed}.npz")
            np.savez_compressed(out, X_train=x_tr[tr].astype(np.float32), y_train=y_tr[tr],
                     X_val=x_tr[va].astype(np.float32), y_val=y_tr[va],
                     X_test=x_te.astype(np.float32), y_test=y_te)
            info["splits"][seed] = {"n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(y_te)),
                                    "n_features": int(x_tr.shape[1]),
                                    "n_classes": int(max(y_tr.max(), y_te.max()) + 1),
                                    "test_idx_head": [int(v) for v in idx[:3]]}
            print(f"{name:8s} seed={seed} train={len(tr)} val={len(va)} test={len(y_te)} "
                  f"P={x_tr.shape[1]} C={info['splits'][seed]['n_classes']} "
                  f"Xrange=[{x_tr.min():.3f},{x_tr.max():.3f}] nan={bool(np.isnan(x_tr).any())}", flush=True)
        manifest["datasets"][name] = info
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main()
