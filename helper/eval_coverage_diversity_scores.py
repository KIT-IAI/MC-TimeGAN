"""eval_coverage_diversity_scores.py - coverage/diversity metrics and raw-label control scores for MC-TimeGAN.

Location: helper/eval_coverage_diversity_scores.py, run from the repo root.

Extends helper/metrics.py with:

1. Control scores: discriminative/predictive scores for synthetic data
   generated with unmodified ordinal labels. Baseline to separate the
   intended label-driven shift from generation deficits.
2. Diversity metrics on flattened day windows: coverage@k, density@k
   (Naeem et al., ICML 2020, arXiv:2002.09797), normalized NN distances,
   diversity_ratio (<1 = repetitive output).

Scale convention (synthetic file suffix = label file suffix, suffix-less =
raw labels): modified exports are on the training scale (sgens: 15 W
dead-band, factor 5), raw-label exports on the original scale (scale 1,
dead-band only). Warns if the synthetic/original max ratio leaves [0.5, 2].

Usage:
    python helper/eval_coverage_diversity_scores.py

Evaluates the load/sgen modified pairs, auto-discovers control exports,
writes helper/slurm/01_eval_coverage_diversity_scores.json. The SLURM
wrapper eval_coverage_diversity_scores_jobs.sh submits one job per pair
(load, pv) and collects logs and per-pair JSONs (_load/_pv suffix) in
helper/slurm/.
"""

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys

import numpy as np

# Make the repository root importable regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except ImportError:
    torch = None

try:
    from helper.metrics import discriminative_score_metrics, predictive_score_metrics

    HAVE_METRICS = True
except ImportError:
    HAVE_METRICS = False  # coverage-only mode works without the repo metrics


def load_array(path, index_col):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        import pandas as pd

        df = pd.read_csv(path, index_col=index_col)
        numeric = df.select_dtypes(include=[np.number])
        print("  using columns: %s" % list(numeric.columns))
        arr = numeric.to_numpy(dtype=np.float64)
    elif ext in (".npy", ".npz"):
        loaded = np.load(path, allow_pickle=False)
        if ext == ".npz":
            loaded = loaded[list(loaded.keys())[0]]
        arr = np.asarray(loaded, dtype=np.float64)
    else:
        sys.exit("Unsupported file type: %s" % path)
    print(
        "  loaded %s  shape=%s  range=[%.4g, %.4g]"
        % (path, arr.shape, arr.min(), arr.max())
    )
    return arr


def to_windows(arr, horizon):
    if arr.ndim == 3:
        return arr
    return np.stack([arr[i : i + horizon] for i in range(len(arr) - horizon)])


def pairwise_dists(a, b):
    sq = (
        np.sum(a * a, axis=1)[:, None]
        + np.sum(b * b, axis=1)[None, :]
        - 2.0 * (a @ b.T)
    )
    np.maximum(sq, 0.0, out=sq)
    return np.sqrt(sq)


def coverage_metrics(ori_flat, syn_flat, k):
    d_oo = pairwise_dists(ori_flat, ori_flat)
    np.fill_diagonal(d_oo, np.inf)
    radii = np.sort(d_oo, axis=1)[:, k - 1]
    d_os = pairwise_dists(ori_flat, syn_flat)
    intra_ori = float(d_oo.min(axis=1).mean())
    d_ss = pairwise_dists(syn_flat, syn_flat)
    np.fill_diagonal(d_ss, np.inf)
    return {
        "coverage": float((d_os.min(axis=1) <= radii).mean()),
        "density": float((d_os <= radii[:, None]).sum(axis=0).mean() / k),
        "nn_dist_ori_to_syn": float(d_os.min(axis=1).mean() / intra_ori),
        "nn_dist_syn_to_ori": float(d_os.min(axis=0).mean() / intra_ori),
        "diversity_ratio": float(d_ss.min(axis=1).mean() / intra_ori),
    }


def evaluate_pair(name, role, ori, syn, syn_scale, syn_deadband, args, meta):
    dim = ori.shape[-1]
    if syn.shape[-1] != dim:
        sys.exit("[%s] feature mismatch" % name)
    if syn_deadband > 0.0:
        syn = np.where(syn < syn_deadband, 0.0, syn)
    if syn_scale != 1.0:
        syn = syn * syn_scale
    ratio = float(syn.max() / max(ori.max(), 1e-12))
    if ratio > 2.0 or ratio < 0.5:
        print(
            "[%s] WARNING: synthetic/original max ratio = %.2f - check the "
            "scale convention of this export" % (name, ratio)
        )
    data_min = ori.reshape(-1, dim).min(axis=0)
    span = ori.reshape(-1, dim).max(axis=0) - data_min
    span = np.where(span == 0.0, 1.0, span)
    ori_win = to_windows((ori - data_min) / span, args.horizon)
    syn_win = to_windows((syn - data_min) / span, args.horizon)
    n = min(len(ori_win), len(syn_win), args.max_windows)
    with_scores = role == "control" and not args.no_scores
    if with_scores and not HAVE_METRICS:
        sys.exit("[%s] helper.metrics not importable - run from the repo root" % name)

    results = {}
    for run in range(args.runs):
        seed = args.seed + run
        rng = np.random.default_rng(seed)
        np.random.seed(seed)
        if torch is not None:
            torch.manual_seed(seed)
        ori_sub = ori_win[rng.choice(len(ori_win), size=n, replace=False)]
        syn_sub = syn_win[rng.choice(len(syn_win), size=n, replace=False)]
        vals = coverage_metrics(
            ori_sub.reshape(n, -1), syn_sub.reshape(n, -1), args.knn
        )
        if with_scores:
            vals["discriminative"] = float(
                discriminative_score_metrics(ori_sub, syn_sub)
            )
            vals["predictive_tstr"] = float(predictive_score_metrics(ori_sub, syn_sub))
            vals["predictive_trtr"] = float(predictive_score_metrics(ori_sub, ori_sub))
        for key, value in vals.items():
            results.setdefault(key, []).append(value)
        print(
            "[%s] run %02d (seed %d): %s"
            % (
                name,
                run + 1,
                seed,
                "  ".join("%s=%.4g" % (k, v) for k, v in vals.items()),
            ),
            flush=True,
        )

    summary = {
        key: {
            "mean": float(np.mean(v)),
            "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "runs": v,
        }
        for key, v in results.items()
    }
    meta["pairs"][name].update(
        {
            "role": role,
            "windows_per_run": int(n),
            "features": int(dim),
            "knn": int(args.knn),
            "synthetic_scale": float(syn_scale),
            "synthetic_deadband": float(syn_deadband),
            "metrics": summary,
        }
    )
    print(
        "[%s] SUMMARY (%s): %s"
        % (
            name,
            role,
            "  ".join(
                "%s=%.4g+/-%.4g" % (k, s["mean"], s["std"]) for k, s in summary.items()
            ),
        )
    )
    if role == "modified":
        print(
            "[%s] note: the label-driven severity shift lowers coverage by "
            "design - judge collapse via diversity_ratio" % name
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        default=None,
        metavar=("NAME", "ORIGINAL", "SYNTHETIC"),
    )
    parser.add_argument(
        "--control-pair",
        nargs=3,
        action="append",
        default=None,
        metavar=("NAME", "ORIGINAL", "SYNTHETIC"),
    )
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--max-windows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=58)
    parser.add_argument("--knn", type=int, default=5)
    parser.add_argument("--index-col", type=int, default=None)
    parser.add_argument("--syn-scale", type=float, default=1.0)
    parser.add_argument("--syn-deadband", type=float, default=0.0)
    parser.add_argument("--control-syn-scale", type=float, default=1.0)
    parser.add_argument("--control-syn-deadband", type=float, default=0.0)
    parser.add_argument("--no-scores", action="store_true")
    parser.add_argument(
        "--out", default="helper/slurm/01_eval_coverage_diversity_scores.json"
    )
    args = parser.parse_args()

    if args.pair is None:
        pairs = [
            [
                "load",
                "modified",
                "helper/data/raw/feeder_loads_4w_data.csv",
                "helper/synthetic_data/2024-07-23_mc_timegan_feeder_loads_n5t1p25_n2.csv",
                1.0,
                0.0,
            ],
            [
                "sgen",
                "modified",
                "helper/data/raw/feeder_sgens_4w_data.csv",
                "helper/synthetic_data/2024-07-23_mc_timegan_feeder_sgens_n5t1_n2.csv",
                5.0,
                1.5e-5,
            ],
        ]
    else:
        pairs = [
            [n, "modified", o, s, args.syn_scale, args.syn_deadband]
            for n, o, s in args.pair
        ]

    if args.control_pair is None and args.pair is None:
        # Raw-label control exports: fixed names first, then the newest
        # dated suffix-less export (naming convention: suffix-less = raw labels).
        import glob

        def newest(*patterns):
            for pat in patterns:
                hits = sorted(glob.glob(pat))
                if hits:
                    return hits[-1]
            return None

        load_ctrl = newest(
            "helper/synthetic_data/mc_timegan_feeder_loads_unmodified.csv",
            "helper/synthetic_data/*_mc_timegan_feeder_loads.csv",
        )
        sgen_ctrl = newest(
            "helper/synthetic_data/mc_timegan_feeder_sgens_unmodified.csv",
            "helper/synthetic_data/*_mc_timegan_feeder_sgens.csv",
        )
        candidates = [
            [
                "load_control",
                "helper/data/raw/feeder_loads_4w_data.csv",
                load_ctrl,
                1.0,
                0.0,
            ],
            [
                "sgen_control",
                "helper/data/raw/feeder_sgens_4w_data.csv",
                sgen_ctrl,
                1.0,
                1.5e-5,
            ],
        ]
        pairs += [
            [n, "control", o, s, sc, db]
            for n, o, s, sc, db in candidates
            if s is not None
        ]
    elif args.control_pair is not None:
        pairs += [
            [n, "control", o, s, args.control_syn_scale, args.control_syn_deadband]
            for n, o, s in args.control_pair
        ]

    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        commit = "unavailable"
    meta = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "git_commit": commit,
        "device": (
            "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        ),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": getattr(torch, "__version__", "not installed"),
        },
        "settings": vars(args),
        "pairs": {},
    }
    print(
        "MC-TimeGAN control/coverage metrics  |  device=%s  commit=%s"
        % (meta["device"], commit[:12])
    )

    for name, role, ori_path, syn_path, syn_scale, syn_deadband in pairs:
        print("\n=== %s (%s) ===" % (name, role))
        ori = load_array(ori_path, args.index_col)
        syn = load_array(syn_path, args.index_col)
        meta["pairs"][name] = {"original_path": ori_path, "synthetic_path": syn_path}
        evaluate_pair(name, role, ori, syn, syn_scale, syn_deadband, args, meta)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, default=str)
    print("\nResults written to %s" % args.out)


if __name__ == "__main__":
    main()
