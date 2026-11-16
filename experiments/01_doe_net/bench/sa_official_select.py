"""Feature selection with the official Sequential Attention code (Yasuda et al., ICLR 2023).

Runs the selection stage of google-research/sequential_attention `experiments/run.py`
(algorithms: sa = Sequential Attention, seql = Sequential LASSO, gl = Group LASSO) on the
seed-wise splits written by bench/prepare_sa_data.py, with the authors' per-dataset
hyper-parameters (hyperparams_sa.py: batch, lr, decay, selection epochs; deep_layers 67,
batch norm on, as in experiments.py). The selected indices are written as JSONL and then
evaluated by run_fs_benchmark.py (--precomputed) with the common downstream protocol.

Deviations from run.py (see bench/PROTOCOL.md, change log v3):
  1. decay_steps = 0 (mice, fashion) makes tf ExponentialDecay undefined (lr = nan at step 0
     and 0 afterwards); it is interpreted as "no decay" (constant learning rate).
  2. seql / gl: SequentialLASSOModel does not forward `layer_sequence` to MLPModel in the
     official code (TypeError); patched at runtime, the official files are not modified.
  3. The integer learning rate of mice (1) is cast to float (run.py does this via absl flags).
  4. The npz train/val split is used (identical to get_dataset()'s split for the same seed).

Requires TensorFlow (CPU is enough) and tf-keras with TF_USE_LEGACY_KERAS=1.
Usage:
  TF_USE_LEGACY_KERAS=1 CUDA_VISIBLE_DEVICES="" python bench/sa_official_select.py \
      --sa_root PATH/google-research/sequential_attention --data_dir DATA --out sa_sel.jsonl \
      --algos sa,seql,gl --datasets mice,coil20,isolet,activity,mnist,fashion --seeds 1 2 3 4 5
"""

import argparse
import json
import os
import random
import sys
import time

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

KEY = {"coil20": "coil"}  # our dataset names -> hyperparams_sa keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sa_root", required=True, help="path to google-research/sequential_attention")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--algos", default="sa,seql,gl")
    ap.add_argument("--datasets", default="mice,coil20,isolet,activity,mnist,fashion")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--intra", type=int, default=4)
    ap.add_argument("--inter", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=None, help="override (smoke tests only)")
    args = ap.parse_args()

    sys.path.insert(0, args.sa_root)
    import numpy as np
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(args.intra)
    tf.config.threading.set_inter_op_parallelism_threads(args.inter)
    from sequential_attention.experiments import hyperparams_sa as hp
    from sequential_attention.experiments.models import mlp as mlp_mod
    from sequential_attention.experiments.models.mlp_sa import SequentialAttentionModel
    from sequential_attention.experiments.models.mlp_seql import SequentialLASSOModel

    stash = {"layer_sequence": None}          # deviation 2
    orig_init = mlp_mod.MLPModel.__init__

    def init_with_layers(self, layer_sequence=None, **kw):
        return orig_init(self, layer_sequence=layer_sequence or stash["layer_sequence"], **kw)
    mlp_mod.MLPModel.__init__ = init_with_layers

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            done.add((r["method"], r["dataset"], r["seed"]))

    for ds in args.datasets.split(","):
        key = KEY.get(ds, ds)
        for seed in args.seeds:
            for algo in args.algos.split(","):
                if (algo, ds, seed) in done:
                    continue
                epochs = args.epochs or hp.EPOCHS[key]
                batch = hp.BATCH[key]
                lr = float(hp.LEARNING_RATE[key])                      # deviation 3
                decay_steps, decay_rate = hp.DECAY_STEPS[key], hp.DECAY_RATE[key]
                if decay_steps == 0:                                   # deviation 1
                    decay_steps, decay_rate = 1, 1.0
                random.seed(seed)
                np.random.seed(seed)
                tf.random.set_seed(seed)
                tf.keras.backend.clear_session()
                z = np.load(os.path.join(args.data_dir, f"{ds}_seed{seed}.npz"))
                n_cls = int(max(z["y_train"].max(), z["y_val"].max(), z["y_test"].max())) + 1
                n_feat = z["X_train"].shape[1]

                def transform(x, y):
                    return tf.cast(x, tf.float32), tf.one_hot(y, n_cls)
                ds_tr = (tf.data.Dataset.from_tensor_slices((z["X_train"], z["y_train"]))
                         .map(transform).shuffle(100, reshuffle_each_iteration=True)
                         .batch(batch, drop_remainder=True))
                ds_va = (tf.data.Dataset.from_tensor_slices((z["X_val"], z["y_val"]))
                         .map(transform).batch(batch))
                steps = epochs * len(ds_tr)
                mlp_args = dict(layer_sequence=[67], is_classification=True, num_classes=n_cls,
                                learning_rate=lr, decay_steps=decay_steps, decay_rate=decay_rate,
                                alpha=0, batch_norm=True)
                stash["layer_sequence"] = [67]
                if algo == "sa":
                    model = SequentialAttentionModel(num_inputs=n_feat, num_inputs_to_select=args.k,
                                                     num_inputs_to_select_per_step=1,
                                                     num_train_steps=steps, **mlp_args)
                else:
                    model = SequentialLASSOModel(num_inputs=n_feat, num_inputs_to_select=args.k,
                                                 num_inputs_to_select_per_step=(args.k if algo == "gl" else 1),
                                                 num_train_steps=steps, group_lasso_scale=0.01,
                                                 **mlp_args)
                model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy"])
                t0 = time.time()
                hist = model.fit(ds_tr, validation_data=ds_va, epochs=epochs, verbose=0)
                if algo == "sa":
                    sel = tf.math.top_k(model.seqatt.selected_features, k=args.k)[1].numpy().tolist()
                else:
                    sel = model.seql.selected_features_history.numpy().tolist()
                assert len(sel) == args.k, len(sel)
                rec = {"method": algo, "dataset": ds, "seed": seed, "selected": [int(i) for i in sel],
                       "sel_epochs": epochs, "sel_seconds": round(time.time() - t0, 1),
                       "sel_final_loss": float(hist.history["loss"][-1]),
                       "sel_final_val_acc": float(hist.history.get("val_accuracy", [float("nan")])[-1]),
                       "sa_hp": {"batch": batch, "lr": lr, "decay_steps": decay_steps, "decay_rate": decay_rate}}
                with open(args.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"{ds:8s} {algo:4s} seed={seed} val_acc={rec['sel_final_val_acc']:.4f} "
                      f"({rec['sel_seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
