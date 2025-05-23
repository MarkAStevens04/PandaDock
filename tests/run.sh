#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run.sh – simple PandaDock launcher
#
# Usage:
#   ./run.sh [mode] [additional PandaDock flags]
#
# Modes:
#   default   → auto-select algorithm (fallback if no mode given)
#   fast      → CPU Monte-Carlo quick search
#   ga        → Genetic Algorithm with enhanced scoring
#   hybrid    → Hybrid (GA+SA) GPU search
#   physics   → Physics-based scoring with energy breakdown
#   pocket    → Detect pockets & auto-grid
# ---------------------------------------------------------------------------

set -euo pipefail

# ---- input files (edit as needed) -----------------------------------------
RECEPTOR="receptor.pdb"
LIGAND="ligand.sdf"
GPU="--use-gpu"
# ---------------------------------------------------------------------------

MODE="${1:-default}"
shift || true  # Drop $1 so remaining args pass straight to PandaDock

case "$MODE" in
  fast)
    pandadock -p "$RECEPTOR" -l "$LIGAND" "$GPU" \
      -a monte-carlo --mc-steps 5000 --fast-mode "$@"
    ;;
  ga)
    pandadock -p "$RECEPTOR" -l "$LIGAND" "$GPU"\
      -a genetic --population-size 50 --iterations 300 \
      --enhanced-scoring "$@"
    ;;
  hybrid)
    pandadock -p "$RECEPTOR" -l "$LIGAND" "$GPU"\
      -a hybrid --use-gpu --gpu-id 0 \
      --hybrid-temperature-start 5.0 --hybrid-temperature-end 0.1 \
      --cooling-factor 0.95 "$@"
    ;;
  physics)
    pandadock -p "$RECEPTOR" -l "$LIGAND" "$GPU"\
      --physics-based --energy-decomposition --per-residue-energy \
      --detailed-energy "$@"
    ;;
  pocket)
    pandadock -p "$RECEPTOR" -l "$LIGAND" "$GPU"\
      --detect-pockets --grid-spacing 0.375 --grid-radius 12.0 "$@"
    ;;
  default|*)
    pandadock -p "$RECEPTOR" -l "$LIGAND" "$GPU" --auto-algorithm "$@"
    ;;
esac

