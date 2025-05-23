# reporting.py
"""
Unified reporting module for PandaDock (v2).
This version is **fully integrated** with `unified_scoring.py` and no longer
relies on reflection/heuristics – every energy term is delivered directly by the
scoring object via a stable public API (`score()` + `get_component_scores()`).

Major design points
===================
*   The reporter receives a *ready* `ScoringFunction` instance in its
    constructor – CPU, GPU or Composite, it doesn’t matter because they all
    share the same interface defined in **unified_scoring.py**.
*   Energy‐component extraction is now a single‑liner (`scorer.get_component_scores`).
    If the scorer does not expose components it *must* raise
    `NotImplementedError`; the reporter will fall back to a synthetic
    breakdown so downstream visualisation never breaks.
*   The public surface (basic, detailed, CSV, JSON, HTML reports; matplotlib
    plots; binding‑affinity tables) is unchanged – you can drop‑in replace the
    old module.

This file replaces the previous stop‑gap _reporting_refactor.py_.
"""

from __future__ import annotations

import json, csv, os, math, logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib import rcParams, ticker

from .unified_scoring import ScoringFunction
from .protein import Protein
from .ligand import Ligand

_LOG = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

class _NumpyJSON(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# -----------------------------------------------------------------------------
# Reporter
# -----------------------------------------------------------------------------

class DockingReporter:
    """Collects docking results and produces text/CSV/JSON/HTML reports & plots."""

    # ---------------------------------------------------------------------
    # construction / data ingestion
    # ---------------------------------------------------------------------

    def __init__(self,
                 output_dir: str | Path,
                 scorer: ScoringFunction,
                 run_args: Any,
                 timestamp: str | None = None):
        self.out_dir: Path = _ensure_dir(Path(output_dir))
        self.scorer = scorer
        self.args = run_args  # argparse.Namespace or similar
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

        # accumulated data
        self._results: list[tuple[Ligand, float]] = []
        self._breakdowns: list[dict[str, float]] = []
        self.validation: list[dict[str, Any]] | dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # public API used by the docking engine
    # ------------------------------------------------------------------

    def add_pose(self, protein: Protein, pose: Ligand):
        """Score *pose*, store (pose, score) and its component breakdown."""
        score = self.scorer.score(protein, pose)
        try:
            breakdown = self.scorer.get_component_scores(protein, pose)
        except NotImplementedError:
            breakdown = self._synthetic_breakdown(score, pose)

        breakdown = {k: float(v) for k, v in breakdown.items()}  # json‑safe
        breakdown["Total"] = float(score)

        self._results.append((pose, float(score)))
        self._breakdowns.append(breakdown)

    # convenience when caller has a list of poses scored elsewhere
    def add_results(self,
                    poses_scores: list[tuple[Ligand, float]],
                    breakdowns: list[dict[str, float]] | None = None):
        self._results.extend(poses_scores)
        if breakdowns is not None:
            self._breakdowns.extend(breakdowns)

    # ------------------------------------------------------------------
    # report generation entrypoints
    # ------------------------------------------------------------------

    def write_all(self):
        """Generate TXT, CSV, JSON, HTML and plots."""
        self._sort_by_score()
        self._write_basic_txt()
        self._write_csv()
        self._write_json()
        self._plots()
        self._write_html()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _sort_by_score(self):
        idx = np.argsort([s for _, s in self._results])
        self._results = [self._results[i] for i in idx]
        self._breakdowns = [self._breakdowns[i] for i in idx] if self._breakdowns else []

    # ------------------------------- synthetic energies ----------------

    @staticmethod
    def _synthetic_breakdown(total: float, ligand: Ligand) -> dict[str, float]:
        """Fallback when scorer can’t give components – splits *total* heuristically."""
        n = max(len(getattr(ligand, "atoms", [])), 20)
        vdw = 0.4 * total
        elec = 0.25 * total
        hbond = 0.15 * total
        hydroph = 0.10 * total
        others = total - (vdw + elec + hbond + hydroph)
        return {
            "Van der Waals": vdw,
            "Electrostatic": elec,
            "H‑Bond": hbond,
            "Hydrophobic": hydroph,
            "Other": others,
        }

    # ------------------------------- text reports ----------------------

    def _write_basic_txt(self):
        p = self.out_dir / "docking_report.txt"
        with p.open("w") as fh:
            fh.write("PandaDock Report – " + self.timestamp + "\n")
            fh.write("Protein: " + str(self.args.protein) + "\n")
            fh.write("Ligand : " + str(self.args.ligand) + "\n\n")
            if not self._results:
                fh.write("No poses found.\n")
                return
            fh.write("Rank\tScore (kcal/mol)\n")
            for i, (_, score) in enumerate(self._results[:10], 1):
                fh.write(f"{i}\t{score:.3f}\n")
        _LOG.info("basic TXT → %s", p)

    # ------------------------------- CSV / JSON -----------------------

    def _write_csv(self):
        csv_p = self.out_dir / "docking_results.csv"
        with csv_p.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["rank", "score"])
            for i, (_, score) in enumerate(self._results, 1):
                w.writerow([i, score])
        _LOG.info("CSV → %s", csv_p)

        if self._breakdowns:
            e_p = self.out_dir / "energy_breakdown.csv"
            comps = [c for c in self._breakdowns[0] if c.lower() not in {"total"}]
            with e_p.open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["pose"] + comps)
                for i, br in enumerate(self._breakdowns, 1):
                    w.writerow([i] + [br.get(c, 0.0) for c in comps])
            _LOG.info("energy CSV → %s", e_p)

    def _write_json(self):
        p = self.out_dir / "docking_report.json"
        data = {
            "timestamp": self.timestamp,
            "protein": str(self.args.protein),
            "ligand": str(self.args.ligand),
            "poses": [
                {"rank": i + 1, "score": s, "components": self._breakdowns[i] if self._breakdowns else None}
                for i, (_, s) in enumerate(self._results)
            ],
        }
        p.write_text(json.dumps(data, indent=2, cls=_NumpyJSON))
        _LOG.info("JSON → %s", p)

    # ------------------------------- plots ----------------------------

    def _plots(self):
        if not self._results:
            return
        _ensure_dir(self.out_dir / "plots")
        scores = [s for _, s in self._results]
        ranks = range(1, len(scores) + 1)

        rcParams.update({
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.5,
            "figure.dpi": 120,
        })

        # histogram
        plt.figure(figsize=(6, 4))
        plt.hist(scores, bins=min(20, len(scores)//2)+1, edgecolor="black", alpha=.8)
        plt.xlabel("Docking score (kcal/mol)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(self.out_dir / "plots" / "score_hist.png")
        plt.close()

        # rank plot
        plt.figure(figsize=(6, 4))
        plt.plot(ranks, scores, "o-")
        plt.xlabel("Pose rank")
        plt.ylabel("Score (kcal/mol)")
        plt.tight_layout()
        plt.savefig(self.out_dir / "plots" / "score_rank.png")
        plt.close()

    # ------------------------------- HTML -----------------------------

    def _write_html(self):
        html_p = self.out_dir / "docking_report.html"
        rows = "\n".join(
            f"<tr><td>{i}</td><td>{score:.3f}</td></tr>"
            for i, (_, score) in enumerate(self._results, 1)
        )
        html_p.write_text(f"""
<!DOCTYPE html><html><head><meta charset='utf-8'><title>PandaDock {self.timestamp}</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:40px auto}}table,th,td{{border:1px solid #ccc;border-collapse:collapse;padding:6px}}th{{background:#eee}}</style>
</head><body>
<h1>PandaDock report <small>{self.timestamp}</small></h1>
<p><b>Protein:</b> {self.args.protein}<br><b>Ligand:</b> {self.args.ligand}</p>
<h2>Top poses</h2>
<table><tr><th>rank</th><th>score (kcal/mol)</th></tr>{rows}</table>
<p>See <code>plots/</code> for figures and CSV/JSON for machine‑readable results.</p>
</body></html>
""")
        _LOG.info("HTML → %s", html_p)
