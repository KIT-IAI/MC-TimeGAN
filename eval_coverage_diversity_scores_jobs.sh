#!/bin/bash
# Script Name: eval_coverage_diversity_scores_jobs.sh
# Target location: repository root (next to __main__.py), run from there.
# Description: SLURM wrapper for the extended evaluation metrics. Master
# modes submit one sbatch job per worker; logs and JSON results are
# collected in helper/slurm/:
#
#   ./eval_coverage_diversity_scores_jobs.sh        2 jobs: coverage/control
#                                                   (MCTG-cov-load / -pv) via
#                                                   helper/eval_coverage_diversity_scores.py
#   ./eval_coverage_diversity_scores_jobs.sh all    4 jobs: fidelity + coverage
#                                                   (MCTG-fid-* via
#                                                   helper/eval_fidelity_scores.py)
#   ./eval_coverage_diversity_scores_jobs.sh load|pv|fid-load|fid-pv
#                                                   worker: runs one scope directly
#   ./eval_coverage_diversity_scores_jobs.sh summary
#                                                   rebuild the summary tables only
#
# Results per worker in helper/slurm/:
#   slurm-coverage-<pair>-<jobid>.out/.err + 01_eval_coverage_diversity_scores_<pair>.json
#   slurm-fidelity-<pair>-<jobid>.out/.err + 01_eval_fidelity_scores_<pair>.json
# Every worker finishes with helper/eval_summary_tables.py, which condenses
# all JSONs present so far into 00_eval_summary_tables.txt (the last job
# completes the table).
#
# Control exports = synthetic data generated with the RAW ordinal labels
# (training notebook: load the checkpoint, generate with
# helper/data/raw_labels/*_labels_ordinal.csv, export; these exports are on
# the ORIGINAL scale, so the control pairs use scale 1 and dead-band only).
# -----------------------------------------
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:2g.10gb:1
#SBATCH --mail-user=goekhan.demirel@kit.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --job-name=MCTG-metrics

# Repo root = submit directory; workers inherit it via SLURM_SUBMIT_DIR.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
LOG_DIR="${REPO_DIR}/helper/slurm"

MODE="$1"

# ----------------------------------------------------------------------------
# Master modes
# ----------------------------------------------------------------------------
if [ -z "$MODE" ]; then
    mkdir -p "$LOG_DIR"
    echo "Master started at $(date) - submitting one coverage job per pair"
    for p in load pv; do
        sbatch --job-name=MCTG-cov-${p} \
               --output="${LOG_DIR}/slurm-coverage-${p}-%j.out" \
               --error="${LOG_DIR}/slurm-coverage-${p}-%j.err" \
               "$0" "$p"
    done
    echo "Master finished at $(date) - monitor with: squeue -u $USER"
    exit 0
fi

if [ "$MODE" = "all" ]; then
    mkdir -p "$LOG_DIR"
    echo "Master started at $(date) - submitting four parallel jobs (fidelity + coverage, load + pv)"
    for w in fid-load fid-pv load pv; do
        case "$w" in
            fid-*) tag="fidelity-${w#fid-}"; jname="MCTG-${w}" ;;
            *)     tag="coverage-${w}";      jname="MCTG-cov-${w}" ;;
        esac
        sbatch --job-name="${jname}" \
               --output="${LOG_DIR}/slurm-${tag}-%j.out" \
               --error="${LOG_DIR}/slurm-${tag}-%j.err" \
               "$0" "$w"
    done
    echo "Master finished at $(date) - monitor with: squeue -u $USER"
    exit 0
fi

# ----------------------------------------------------------------------------
# Worker helpers
# ----------------------------------------------------------------------------
run_fidelity_load() {
    python helper/eval_fidelity_scores.py \
        --pair load helper/data/raw/feeder_loads_4w_data.csv \
                    helper/synthetic_data/2024-07-23_mc_timegan_feeder_loads_n5t1p25_n2.csv \
        --runs 10 --horizon 96 --max-windows 1000 --seed 58 \
        --out "${LOG_DIR}/01_eval_fidelity_scores_load.json"
}

run_fidelity_pv() {
    python helper/eval_fidelity_scores.py \
        --pair pv helper/data/raw/feeder_sgens_4w_data.csv \
                  helper/synthetic_data/2024-07-23_mc_timegan_feeder_sgens_n5t1_n2.csv \
        --syn-scale 5 --syn-deadband 1.5e-5 \
        --runs 10 --horizon 96 --max-windows 1000 --seed 58 \
        --out "${LOG_DIR}/01_eval_fidelity_scores_pv.json"
}

run_coverage_load() {
    local CONTROL="helper/synthetic_data/mc_timegan_feeder_loads_unmodified.csv"
    local EXTRA=""
    if [ -f "$CONTROL" ]; then
        EXTRA="--control-pair load_control helper/data/raw/feeder_loads_4w_data.csv ${CONTROL}"
    else
        echo "note: ${CONTROL} not found - running coverage metrics only"
    fi
    python helper/eval_coverage_diversity_scores.py \
        --pair load helper/data/raw/feeder_loads_4w_data.csv \
                    helper/synthetic_data/2024-07-23_mc_timegan_feeder_loads_n5t1p25_n2.csv \
        ${EXTRA} \
        --runs 10 --horizon 96 --max-windows 1000 --seed 58 --knn 5 \
        --out "${LOG_DIR}/01_eval_coverage_diversity_scores_load.json"
}

run_coverage_pv() {
    # Suffix-less raw-label export, already on the original scale (see header).
    local CONTROL="helper/synthetic_data/2024-08-01_mc_timegan_feeder_sgens.csv"
    local EXTRA=""
    if [ -f "$CONTROL" ]; then
        EXTRA="--control-pair sgen_control helper/data/raw/feeder_sgens_4w_data.csv ${CONTROL} --control-syn-deadband 1.5e-5"
    else
        echo "note: ${CONTROL} not found - running coverage metrics only"
    fi
    python helper/eval_coverage_diversity_scores.py \
        --pair sgen helper/data/raw/feeder_sgens_4w_data.csv \
                    helper/synthetic_data/2024-07-23_mc_timegan_feeder_sgens_n5t1_n2.csv \
        ${EXTRA} \
        --syn-scale 5 --syn-deadband 1.5e-5 \
        --runs 10 --horizon 96 --max-windows 1000 --seed 58 --knn 5 \
        --out "${LOG_DIR}/01_eval_coverage_diversity_scores_pv.json"
}

# ----------------------------------------------------------------------------
# Worker modes
# ----------------------------------------------------------------------------
cd "$REPO_DIR" || exit 1
mkdir -p "$LOG_DIR"
echo "Worker (${MODE}) started at $(date) on $(hostname)"
VENV="${VENV:-$HOME/mctimegan_env/bin/activate}"
[ -f "$VENV" ] && source "$VENV"
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

if [ "$MODE" = "load" ]; then
    run_coverage_load
elif [ "$MODE" = "pv" ]; then
    run_coverage_pv
elif [ "$MODE" = "fid-load" ]; then
    run_fidelity_load
elif [ "$MODE" = "fid-pv" ]; then
    run_fidelity_pv
elif [ "$MODE" = "summary" ]; then
    :  # summary only - rebuilt below from the JSONs present
else
    echo "Unknown mode '$MODE' - expected 'all', 'load', 'pv', 'fid-load', 'fid-pv' or 'summary'" >&2
    exit 1
fi

python helper/eval_summary_tables.py --dir "$LOG_DIR"
echo "Worker (${MODE}) finished at $(date) - logs and results in ${LOG_DIR}/"
