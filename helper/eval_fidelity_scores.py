"""eval_fidelity_scores.py - TimeGAN discriminative/predictive scores for MC-TimeGAN outputs.

Location: helper/eval_fidelity_scores.py, run from the repo root.

Reuses the repo metrics in helper/metrics.py (Yoon et al., NeurIPS 2019):
discriminative = |0.5 - test accuracy| of a post-hoc GRU classifier (0 =
indistinguishable, max 0.5), predictive = TSTR MAE, reported next to the
TRTR baseline. Preprocessing mirrors Evaluation.prepare_data: min-max
scaling fitted on the original data, sliding windows of --horizon timesteps
(default 96 = one day at 15 min), --max-windows windows per dataset and
run, --runs repetitions with seeds seed..seed+runs-1, mean +/- std.

Scale convention as in eval_coverage_diversity_scores.py: the default pv
pair carries the notebook's 15 W dead-band mask and fivefold scale, the
load pair shares one scale and needs no factor.

Usage:
    python helper/eval_fidelity_scores.py

Evaluates the load and pv pairs of the evaluation notebook and writes
helper/slurm/01_eval_fidelity_scores.json. The SLURM wrapper
eval_coverage_diversity_scores_jobs.sh (all mode) runs fidelity and
coverage per pair and writes 01_eval_fidelity_scores_load.json /
01_eval_fidelity_scores_pv.json instead.
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
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

try:
    import torch
except ImportError:
    torch = None

try:
    from helper.metrics import discriminative_score_metrics, predictive_score_metrics
except ImportError as exc:
    sys.exit(
        "Could not import helper.metrics - place this script in helper/ of the "
        "MC-TimeGAN repository and run it from the repo root.\nImport error: %s" % exc
    )


def load_array(path, index_col):
    """Load a 2D time series (time, features) or 3D window stack from csv/npy/npz."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        import pandas as pd

        df = pd.read_csv(path, index_col=index_col)
        numeric = df.select_dtypes(include=[np.number])
        dropped = [c for c in df.columns if c not in numeric.columns]
        if dropped:
            print("  note: dropped non-numeric columns %s" % dropped)
        print("  using columns: %s" % list(numeric.columns))
        arr = numeric.to_numpy(dtype=np.float64)
    elif ext in (".npy", ".npz"):
        loaded = np.load(path, allow_pickle=False)
        if ext == ".npz":
            keys = list(loaded.keys())
            if len(keys) != 1:
                sys.exit(
                    "%s contains keys %s - expected exactly one array" % (path, keys)
                )
            loaded = loaded[keys[0]]
        arr = np.asarray(loaded, dtype=np.float64)
    else:
        sys.exit("Unsupported file type: %s (expected .csv/.npy/.npz)" % path)
    if arr.ndim not in (2, 3):
        sys.exit(
            "%s has shape %s - expected 2D (time, features) or 3D windows"
            % (path, arr.shape)
        )
    print(
        "  loaded %s  shape=%s  range=[%.4g, %.4g]"
        % (path, arr.shape, arr.min(), arr.max())
    )
    return arr


def to_windows(arr, horizon):
    """Slice a 2D series into sliding windows (stride 1), mirroring Evaluation.prepare_data."""
    if arr.ndim == 3:
        return arr
    if len(arr) <= horizon:
        sys.exit("Series of length %d is shorter than horizon %d" % (len(arr), horizon))
    return np.stack([arr[i : i + horizon] for i in range(len(arr) - horizon)])


def fit_minmax(reference_2d):
    """Column-wise min-max parameters fitted on the original data."""
    data_min = reference_2d.min(axis=0)
    data_max = reference_2d.max(axis=0)
    span = np.where(data_max - data_min == 0.0, 1.0, data_max - data_min)
    return data_min, span


def evaluate_pair(name, ori, syn, syn_scale, syn_deadband, args, meta):
    """Run both scores args.runs times on freshly subsampled windows."""
    dim = ori.shape[-1]
    if syn.shape[-1] != dim:
        sys.exit(
            "[%s] feature mismatch: original %d vs synthetic %d"
            % (name, dim, syn.shape[-1])
        )

    # Mirror the evaluation notebook's synthetic preprocessing: dead-band mask
    # first (values below the threshold set to zero), then the scale factor
    # that lifts the raw generator output onto the scale of the original file.
    if syn_deadband > 0.0:
        syn = np.where(syn < syn_deadband, 0.0, syn)
    if syn_scale != 1.0:
        syn = syn * syn_scale
    if syn_deadband > 0.0 or syn_scale != 1.0:
        print(
            "[%s] synthetic preprocessing: deadband<%g -> 0, scale x%g, "
            "new range=[%.4g, %.4g]"
            % (name, syn_deadband, syn_scale, syn.min(), syn.max())
        )

    # Scale with parameters fitted on the ORIGINAL data so both datasets share
    # the identical transform (statistical fidelity is judged in original space).
    data_min, span = fit_minmax(ori.reshape(-1, dim))
    ori_scaled = (ori - data_min) / span
    syn_scaled = syn if args.synthetic_is_normalized else (syn - data_min) / span

    ori_win = to_windows(ori_scaled, args.horizon)
    syn_win = to_windows(syn_scaled, args.horizon)
    n = min(len(ori_win), len(syn_win), args.max_windows)
    print(
        "[%s] windows: original %d, synthetic %d -> %d per dataset and run, "
        "scaled ranges ori=[%.3f, %.3f] syn=[%.3f, %.3f]"
        % (
            name,
            len(ori_win),
            len(syn_win),
            n,
            ori_win.min(),
            ori_win.max(),
            syn_win.min(),
            syn_win.max(),
        )
    )

    results = {"discriminative": [], "predictive": [], "predictive_baseline_trtr": []}
    for run in range(args.runs):
        seed = args.seed + run
        rng = np.random.default_rng(seed)
        np.random.seed(seed)
        if torch is not None:
            torch.manual_seed(seed)
        ori_sub = ori_win[rng.choice(len(ori_win), size=n, replace=False)]
        syn_sub = syn_win[rng.choice(len(syn_win), size=n, replace=False)]

        disc = float(discriminative_score_metrics(ori_sub, syn_sub))
        pred = float(predictive_score_metrics(ori_sub, syn_sub))
        results["discriminative"].append(disc)
        results["predictive"].append(pred)
        line = "[%s] run %02d (seed %d): discriminative=%.4f  predictive(TSTR)=%.3e" % (
            name,
            run + 1,
            seed,
            disc,
            pred,
        )
        if not args.no_baseline:
            base = float(predictive_score_metrics(ori_sub, ori_sub))
            results["predictive_baseline_trtr"].append(base)
            line += "  baseline(TRTR)=%.3e" % base
        print(line, flush=True)

    summary = {}
    for key, values in results.items():
        if values:
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "runs": values,
            }
    meta["pairs"][name].update(
        {
            "original_windows": int(len(ori_win)),
            "synthetic_windows": int(len(syn_win)),
            "windows_per_run": int(n),
            "features": int(dim),
            "synthetic_scale": float(syn_scale),
            "synthetic_deadband": float(syn_deadband),
            "scores": summary,
        }
    )

    print("[%s] SUMMARY over %d runs:" % (name, args.runs))
    print(
        "[%s]   discriminative score : %.4f +/- %.4f   (0 = indistinguishable, max 0.5)"
        % (name, summary["discriminative"]["mean"], summary["discriminative"]["std"])
    )
    print(
        "[%s]   predictive score TSTR: %.3e +/- %.3e"
        % (name, summary["predictive"]["mean"], summary["predictive"]["std"])
    )
    if "predictive_baseline_trtr" in summary:
        print(
            "[%s]   predictive baseline  : %.3e +/- %.3e   (train-on-real reference)"
            % (
                name,
                summary["predictive_baseline_trtr"]["mean"],
                summary["predictive_baseline_trtr"]["std"],
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        default=None,
        metavar=("NAME", "ORIGINAL", "SYNTHETIC"),
        help="model name plus paths to original and synthetic data (repeat once "
        "per model; defaults to the evaluation-notebook datasets for load and pv)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=96,
        help="window length in timesteps (default 96 = one day at 15 min)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="repetitions per score (default 10, TimeGAN convention)",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=1000,
        help="windows sampled per dataset and run (default 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=58,
        help="base seed, run r uses seed+r (default 58 as in the tutorials)",
    )
    parser.add_argument(
        "--index-col",
        type=int,
        default=None,
        help="csv column index to treat as index (e.g. 0), default none",
    )
    parser.add_argument(
        "--syn-scale",
        type=float,
        default=1.0,
        help="scale factor applied to the synthetic data of every explicit "
        "--pair (the evaluation notebook lifts the raw PV generator output "
        "with factor 5); default 1.0",
    )
    parser.add_argument(
        "--syn-deadband",
        type=float,
        default=0.0,
        help="synthetic values below this threshold are set to zero before "
        "scaling, applied to every explicit --pair (the notebook masks PV "
        "below 15 W = 1.5e-5 MW); default 0",
    )
    parser.add_argument(
        "--synthetic-is-normalized",
        action="store_true",
        help="synthetic data is already min-max normalized to [0, 1]",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="skip the train-on-real predictive baseline",
    )
    parser.add_argument(
        "--out",
        default="helper/slurm/01_eval_fidelity_scores.json",
        help="JSON output path",
    )
    args = parser.parse_args()

    if args.pair is None:
        # The datasets loaded by the evaluation notebook. feeder_sgens_4w_data
        # .csv is the fivefold-scaled re-export while the generator output is
        # on the unscaled training scale - the pv pair therefore carries the
        # notebook's dead-band mask (15 W) and the fivefold scale. The load
        # files share one scale.
        pairs = [
            [
                "load",
                "helper/data/raw/feeder_loads_4w_data.csv",
                "helper/synthetic_data/2024-07-23_mc_timegan_feeder_loads_n5t1p25_n2.csv",
                1.0,
                0.0,
            ],
            [
                "pv",
                "helper/data/raw/feeder_sgens_4w_data.csv",
                "helper/synthetic_data/2024-07-23_mc_timegan_feeder_sgens_n5t1_n2.csv",
                5.0,
                1.5e-5,
            ],
        ]
    else:
        pairs = [
            [name, ori, syn, args.syn_scale, args.syn_deadband]
            for name, ori, syn in args.pair
        ]
    for _, ori_path, syn_path, _, _ in pairs:
        for path in (ori_path, syn_path):
            if not os.path.exists(path):
                sys.exit(
                    "File not found: %s (run from the MC-TimeGAN repo root, "
                    "or pass --pair with your paths)" % path
                )

    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=REPO_ROOT
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
        "settings": {
            "horizon": args.horizon,
            "runs": args.runs,
            "max_windows": args.max_windows,
            "base_seed": args.seed,
            "synthetic_is_normalized": args.synthetic_is_normalized,
        },
        "pairs": {},
    }
    print(
        "MC-TimeGAN fidelity scores  |  device=%s  commit=%s"
        % (meta["device"], commit[:12])
    )

    for name, ori_path, syn_path, syn_scale, syn_deadband in pairs:
        print("\n=== %s ===" % name)
        print("original:")
        ori = load_array(ori_path, args.index_col)
        print("synthetic:")
        syn = load_array(syn_path, args.index_col)
        meta["pairs"][name] = {"original_path": ori_path, "synthetic_path": syn_path}
        evaluate_pair(name, ori, syn, syn_scale, syn_deadband, args, meta)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print("\nResults written to %s" % args.out)


if __name__ == "__main__":
    main()
