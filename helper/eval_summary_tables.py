"""eval_summary_tables.py - plain-text summary tables for the extended evaluation metrics.

Location: helper/eval_summary_tables.py, run from the repo root.

Collects the JSONs written by eval_fidelity_scores.py and
eval_coverage_diversity_scores.py from helper/slurm/ (suffix-less or
_load/_pv) and renders one table per pair: modified-label metrics next to
the raw-label control scores. Missing JSONs leave "-" gaps instead of
failing - each worker of the SLURM wrapper calls this script last, so the
table completes with the last finishing job.

Usage:
    python helper/eval_summary_tables.py [--dir helper/slurm]

writes helper/slurm/00_eval_summary_tables.txt and prints it to stdout.
"""

import argparse
import json
import os

GROUPS = [
    ("load", ["load"], ["load_control"]),
    ("pv", ["pv", "sgen"], ["pv_control", "sgen_control"]),
]


def collect(dir_, stem):
    """Merge the 'pairs' dicts of the suffix-less and per-pair JSON variants."""
    pairs, sources = {}, []
    for suffix in ("", "_load", "_pv"):
        path = os.path.join(dir_, "01_%s%s.json" % (stem, suffix))
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                pairs.update(json.load(handle).get("pairs", {}))
            sources.append(os.path.basename(path))
        except (OSError, ValueError) as exc:
            print("note: skipping %s (%s)" % (path, exc))
    return pairs, sources


def pick(pairs, names):
    for name in names:
        if name in pairs:
            return pairs[name]
    return None


def metric(pair, *keys):
    """Return the first matching mean/std entry from 'metrics' or 'scores'."""
    if pair is None:
        return None
    entries = pair.get("metrics") or pair.get("scores") or {}
    for key in keys:
        if key in entries:
            return entries[key]
    return None


def fmt(entry):
    if entry is None:
        return "-"
    mean, std = entry["mean"], entry.get("std", 0.0)
    if mean != 0.0 and abs(mean) < 1e-2:
        return "%.2e ± %.1e" % (mean, std)
    return "%.3f ± %.3f" % (mean, std)


def fmt_predictive(tstr, trtr):
    if tstr is None:
        return "-"
    text = "%.2e ± %.1e" % (tstr["mean"], tstr.get("std", 0.0))
    if trtr is not None:
        text += "  (TRTR %.2e)" % trtr["mean"]
    return text


def render(title, header, rows):
    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]

    def rule(left, mid, right):
        return left + mid.join("─" * (w + 2) for w in widths) + right

    def line(cells):
        return "│ " + " │ ".join(c.ljust(w) for c, w in zip(cells, widths)) + " │"

    parts = [title, rule("┌", "┬", "┐"), line(header)]
    parts.append(rule("├", "┼", "┤"))
    for i, cells in enumerate(rows):
        parts.append(line(cells))
        parts.append(rule("├", "┼", "┤") if i < len(rows) - 1 else rule("└", "┴", "┘"))
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default="helper/slurm")
    parser.add_argument(
        "--out", default=None, help="default <dir>/00_eval_summary_tables.txt"
    )
    args = parser.parse_args()
    out = args.out or os.path.join(args.dir, "00_eval_summary_tables.txt")

    fid_pairs, fid_sources = collect(args.dir, "eval_fidelity_scores")
    cov_pairs, cov_sources = collect(args.dir, "eval_coverage_diversity_scores")

    blocks = []
    for title, mod_names, ctrl_names in GROUPS:
        cov_mod = pick(cov_pairs, mod_names)
        cov_ctrl = pick(cov_pairs, ctrl_names)
        fid_mod = pick(fid_pairs, mod_names)
        if cov_mod is None and cov_ctrl is None and fid_mod is None:
            continue
        knn = (cov_mod or cov_ctrl or {}).get("knn", 5)
        header = ["Metric", "%s (modified labels)" % title, "%s (raw labels)" % title]
        rows = [
            [
                "coverage@%d" % knn,
                fmt(metric(cov_mod, "coverage")),
                fmt(metric(cov_ctrl, "coverage")),
            ],
            [
                "density@%d" % knn,
                fmt(metric(cov_mod, "density")),
                fmt(metric(cov_ctrl, "density")),
            ],
            [
                "diversity_ratio",
                fmt(metric(cov_mod, "diversity_ratio")),
                fmt(metric(cov_ctrl, "diversity_ratio")),
            ],
            [
                "discriminative",
                fmt(metric(fid_mod, "discriminative")),
                fmt(metric(cov_ctrl, "discriminative")),
            ],
            [
                "predictive (TSTR)",
                fmt_predictive(
                    metric(fid_mod, "predictive", "predictive_tstr"),
                    metric(fid_mod, "predictive_baseline_trtr", "predictive_trtr"),
                ),
                fmt_predictive(
                    metric(cov_ctrl, "predictive_tstr", "predictive"),
                    metric(cov_ctrl, "predictive_trtr", "predictive_baseline_trtr"),
                ),
            ],
        ]
        blocks.append(render(title, header, rows))

    if blocks:
        legend = (
            "modified labels = synthetic export generated with the MODIFIED "
            "ordinal labels\n"
            "                  (the intended congestion shift)\n"
            "raw labels      = control export generated with the UNMODIFIED "
            "ordinal labels\n"
            "                  (baseline that separates the label-driven shift "
            "from generation deficits)"
        )
        sources = fid_sources + cov_sources
        text = (
            "\n\n".join(blocks)
            + "\n\n"
            + legend
            + "\n\nSources: "
            + ", ".join(sources)
            + "\n"
        )
    else:
        text = "No result JSONs found in %s\n" % args.dir

    print(text)
    os.makedirs(args.dir, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(text)
    print("Summary written to %s" % out)


if __name__ == "__main__":
    main()
