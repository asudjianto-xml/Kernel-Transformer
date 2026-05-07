"""Head-to-head benchmark of seven attention/kernel variants (Chapter 12).

Migrated from ``affinity/ablation_runner.py`` and
``affinity/ablation_aggregate.py``. The full sweep is the 360-cell
falsification grid described in Chapter 12 of *Kernels and Transformers
for Tabular Data* and in §6 of the flagship paper
:cite:`sudjianto2026symmetric`:

    * **20 datasets** (10 regression, 10 classification),
    * **7 variants** -- ``std_attn``, ``baseline_attn`` (asymmetric);
      ``aff_nw``, ``diff_ppr``, ``diff_cheb``, ``diff_multiscale``,
      ``psd_nw`` (symmetric),
    * **18 (head-mode, depth, head-count) configurations** =
      :math:`\\{NW, MLP\\} \\times \\{1, 2, 4\\} \\times \\{1, 4, 8\\}`,
    * 3 seeds per (variant, dataset, configuration) cell.

Per the §6 convention, a *cell* is a (dataset, head-count) entry within
a (head-mode, depth, task) *corner*; there are :math:`30` cells
(:math:`10` datasets :math:`\\times 3` head counts) per corner and
:math:`12` corners per task type, for :math:`360` cells per task type.

The full sweep takes multiple GPU-hours on an NVIDIA GB10. The chapter
notebook therefore loads cached results from
``affinity/book/data/cached_audits/benchmark_360cell.json`` (canonical
name; ported from ``affinity/ablation_results_full.json``) and only runs
a tiny demonstration slice in-line.

This module exposes three public functions:

* :func:`run_360cell_benchmark` -- the runnable scaffold; supports a
  ``slice_only`` mode the notebook calls to demonstrate the API.
* :func:`aggregate_winrates` -- the §6.2-§6.4 aggregator. Operates on a
  fresh report or one loaded from JSON.
* :func:`plot_per_corner_winners` -- the per-cell scatter that becomes
  Figure 12.1.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from .base import AuditReport, save_report, seed_all

# --------------------------------------------------------------------------- #
# Variant taxonomy                                                            #
# --------------------------------------------------------------------------- #

ASYMMETRIC_VARIANTS: tuple[str, ...] = ("std_attn", "baseline_attn")
SYMMETRIC_VARIANTS: tuple[str, ...] = (
    "aff_nw",
    "diff_ppr",
    "diff_cheb",
    "diff_multiscale",
    "psd_nw",
)
ALL_VARIANTS: tuple[str, ...] = ASYMMETRIC_VARIANTS + SYMMETRIC_VARIANTS

#: Std-Attn is the canonical asymmetric reference for the per-cell
#: best-symmetric comparison of §6.3 / §12.3.
STD_REFERENCE: str = "std_attn"

#: ``{NW, MLP} x {L = 1, 2, 4}`` per task-type, the §6 "corner" axis.
DEFAULT_HEAD_MODES: tuple[str, ...] = ("nw", "mlp")
DEFAULT_DEPTHS: tuple[int, ...] = (1, 2, 4)
DEFAULT_HEAD_COUNTS: tuple[int, ...] = (1, 4, 8)


# --------------------------------------------------------------------------- #
# Dataclasses for the API surface                                             #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Corner:
    """One (head_mode, depth, task) corner of the falsification grid.

    Per §6.1 of :cite:`sudjianto2026symmetric` each corner contains
    :math:`10\\,\\text{datasets} \\times 3\\,H = 30` cells.
    """

    head_mode: str
    depth: int
    task: str

    def as_tuple(self) -> tuple[str, int, str]:
        return (self.head_mode, self.depth, self.task)


@dataclass
class CellResult:
    """A single (variant, dataset, H, L, head_mode, seed) result.

    ``loss`` is the held-out loss in the natural metric for the task
    (NLL for classification, MSE for regression). ``raw`` carries the
    per-seed losses when ``n_seeds > 1``.
    """

    variant: str
    dataset: str
    task: str
    H: int
    n_layers: int
    head_mode: str
    loss: float
    std: float = 0.0
    raw: list[float] = field(default_factory=list)
    mean_acc: float | None = None


# --------------------------------------------------------------------------- #
# Public API: run / aggregate / plot                                          #
# --------------------------------------------------------------------------- #


def run_360cell_benchmark(
    datasets: Sequence[Mapping[str, Any]],
    variants: Sequence[str],
    corners: Sequence[Corner],
    *,
    seeds: int = 3,
    head_counts: Sequence[int] = DEFAULT_HEAD_COUNTS,
    epochs: int = 600,
    cache_path: str | Path | None = None,
    slice_only: bool = False,
    train_fn: Any = None,
) -> AuditReport:
    """Run the head-to-head head-to-head sweep and return an :class:`AuditReport`.

    The full 360-cell sweep takes multiple GPU-hours and is gated behind
    the ``slice_only=False`` flag plus an explicit ``train_fn``. The
    chapter notebook always uses the cached-JSON path
    (``benchmark_360cell.json``) plus a small ``slice_only=True`` demo.

    Parameters
    ----------
    datasets:
        A list of dataset descriptors. Each must carry at least
        ``{name, task, n, p, metric}``; ``task`` is one of
        ``"regression"`` or ``"classification"``. The chapter ships with
        the 20 small-tabular datasets of §6.1.
    variants:
        Subset of :data:`ALL_VARIANTS` to evaluate. The headline rates
        are computed against ``STD_REFERENCE`` and the symmetric set
        :data:`SYMMETRIC_VARIANTS`; for partial sweeps only the cells
        that contain all required variants enter the head-to-head.
    corners:
        Iterable of (head_mode, depth, task) :class:`Corner` triples.
        Cells are formed for every (corner, dataset, head-count, seed,
        variant) tuple that matches the corner's task.
    seeds:
        Number of training seeds per (variant, dataset, head-count,
        depth, head-mode) cell. The §6 sweep uses 3.
    head_counts:
        Number of heads :math:`H \\in \\{1, 4, 8\\}` per cell.
    epochs:
        Per-cell training epochs. Forwarded to ``train_fn``.
    cache_path:
        Optional path to dump the resulting report (via
        :func:`tabkernels.audits.base.save_report`).
    slice_only:
        If True, no training is performed: each cell is filled with a
        deterministic dummy loss derived from the per-cell hash. Used by
        the chapter notebook smoke test (``2 datasets x 2 corners x 1
        seed`` runs in :math:`<1` s) and by :func:`pytest`. Production
        runs must pass ``slice_only=False`` and an explicit ``train_fn``.
    train_fn:
        Callable with signature
        ``train_fn(dataset, variant, head_mode, n_layers, H, seed,
        epochs) -> dict`` returning at least ``{loss, mean_acc}``.
        Required when ``slice_only=False``.

    Returns
    -------
    AuditReport
        ``{config, metrics, per_seed, notes}``. ``per_seed`` carries one
        row per :class:`CellResult`; ``metrics`` carries the aggregated
        head-to-head numbers from :func:`aggregate_winrates`.

    Notes
    -----
    The §6 chapter table cites the *full* benchmark numbers
    (78% / 88% / 69% best-symmetric win-rates) which are published in the
    flagship paper :cite:`sudjianto2026symmetric` and replayed from the
    cached JSON. Reproducing them end-to-end requires the full
    architecture stack of :mod:`tabkernels.architectures`, which is out
    of scope for this audit module. For a notebook-tractable scale-up
    path see the TALENT integration documented in
    ``notebooks/12_benchmark.ipynb``.
    """
    if not slice_only and train_fn is None:
        raise ValueError(
            "Production sweep requires an explicit train_fn. "
            "For a smoke run, pass slice_only=True."
        )

    cells: list[dict[str, Any]] = []
    rng_seed_base = 0
    for corner in corners:
        for ds in datasets:
            if ds["task"] != corner.task:
                continue
            for H in head_counts:
                for variant in variants:
                    raw = []
                    accs: list[float] = []
                    for seed in range(seeds):
                        seed_all(rng_seed_base + seed)
                        if slice_only:
                            res = _slice_dummy_result(
                                ds["name"], variant, corner, H, seed
                            )
                        else:
                            res = train_fn(  # type: ignore[misc]
                                dataset=ds,
                                variant=variant,
                                head_mode=corner.head_mode,
                                n_layers=corner.depth,
                                H=H,
                                seed=seed,
                                epochs=epochs,
                            )
                        raw.append(float(res["loss"]))
                        if res.get("mean_acc") is not None:
                            accs.append(float(res["mean_acc"]))
                    mean = float(np.mean(raw))
                    std = float(np.std(raw))
                    mean_acc = float(np.mean(accs)) if accs else None
                    cells.append(
                        dict(
                            variant=variant,
                            dataset=ds["name"],
                            task=corner.task,
                            H=int(H),
                            n_layers=int(corner.depth),
                            head_mode=corner.head_mode,
                            mean=mean,
                            std=std,
                            raw=raw,
                            mean_acc=mean_acc,
                        )
                    )

    report: AuditReport = {
        "config": {
            "n_datasets": len(datasets),
            "n_variants": len(variants),
            "n_corners": len(corners),
            "seeds": seeds,
            "head_counts": list(head_counts),
            "epochs": epochs,
            "slice_only": slice_only,
        },
        "per_seed": cells,
        "notes": (
            "Slice-only run; losses are deterministic stubs."
            if slice_only
            else "Full benchmark run."
        ),
    }
    report["metrics"] = _summary_payload(cells)

    if cache_path is not None:
        save_report(report, cache_path)
    return report


def aggregate_winrates(report: AuditReport | Mapping[str, Any]) -> pd.DataFrame:
    """Aggregate a :class:`AuditReport` into per-corner head-to-head stats.

    The returned DataFrame has one row per (head_mode, depth, task)
    corner plus three roll-up rows (NW overall, MLP overall, all
    corners). Columns:

        ``head_mode, depth, task, n_cells, sym_win_frac, median_delta,
        std_attn_wins, best_sym_wins, best_sym_variant``

    where ``sym_win_frac`` is the §6.3 :math:`\\Pr[\\Delta > 0]` --- the
    fraction of cells in the corner where the best symmetric variant
    beats Std-Attn. ``median_delta`` is the median of
    :math:`\\Delta = L_{\\text{Std-Attn}} - \\min_{v \\in \\mathcal{S}} L_v`,
    in NLL or MSE units. The flagship-paper headline of 78% / 88% / 69%
    appears as the three roll-up rows.

    Parameters
    ----------
    report:
        A report produced by :func:`run_360cell_benchmark` or loaded
        from ``benchmark_360cell.json`` via
        :func:`tabkernels.audits.base.load_report`.
    """
    cells = list(_iter_cells(report))
    if not cells:
        raise ValueError("Report carries no cells.")

    h2h = _headtohead_per_cell(cells)
    rows: list[dict[str, Any]] = []
    head_modes = sorted({r["head_mode"] for r in h2h})
    depths = sorted({r["n_layers"] for r in h2h})
    tasks = sorted({r["task"] for r in h2h})
    for hm in head_modes:
        for L in depths:
            for task in tasks:
                rows.append(_winrate_row(h2h, head_mode=hm, depth=L, task=task))
    # Roll-ups: NW overall, MLP overall, all corners.
    for hm in head_modes:
        rows.append(_winrate_row(h2h, head_mode=hm, depth=None, task=None,
                                 label=f"{hm.upper()} overall"))
    rows.append(_winrate_row(h2h, head_mode=None, depth=None, task=None,
                             label="ALL"))
    df = pd.DataFrame(rows)
    return df


def plot_per_corner_winners(
    report: AuditReport | Mapping[str, Any],
    *,
    figsize: tuple[float, float] = (6.6, 4.8),
) -> matplotlib.figure.Figure:
    """Per-cell win-rate scatter (Figure 12.1).

    Each marker is one cell of the falsification grid, plotted at
    :math:`x = L_{\\text{Std-Attn}}`, :math:`y = \\min_{v \\in
    \\mathcal{S}} L_v`. Markers below the diagonal :math:`y = x`
    correspond to cells where the best symmetric variant beats Std-Attn.
    The MLP-head cells (the FM-relevant regime) are filled; the NW-head
    cells are open.

    Returns the :class:`matplotlib.figure.Figure` so the caller can
    save it under ``figures/fig_12_01_per_cell_winrate.pdf``.
    """
    cells = list(_iter_cells(report))
    h2h = _headtohead_per_cell(cells)
    if not h2h:
        raise ValueError("Report contains no head-to-head-compatible cells.")

    fig, ax = plt.subplots(figsize=figsize)
    for hm, marker, label_prefix in (("mlp", "o", "MLP"), ("nw", "s", "NW")):
        for task, color in (("regression", "#1f77b4"), ("classification", "#d62728")):
            sub = [r for r in h2h if r["head_mode"] == hm and r["task"] == task]
            if not sub:
                continue
            xs = np.array([r["std_loss"] for r in sub])
            ys = np.array([r["best_sym_loss"] for r in sub])
            ax.scatter(
                xs, ys, marker=marker,
                facecolors=color if hm == "mlp" else "none",
                edgecolors=color, s=24, alpha=0.75,
                label=f"{label_prefix}, {task[:3]}",
            )
    lo = min(min(r["std_loss"] for r in h2h),
             min(r["best_sym_loss"] for r in h2h))
    hi = max(max(r["std_loss"] for r in h2h),
             max(r["best_sym_loss"] for r in h2h))
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6, label="$y=x$")
    ax.set_xlabel("Std-Attn held-out loss")
    ax.set_ylabel("Best symmetric held-out loss")
    ax.set_title("Per-cell win-rate: Std-Attn vs best symmetric variant")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Migration helper: ablation_results_full.json -> benchmark_360cell.json      #
# --------------------------------------------------------------------------- #


def migrate_flagship_json(src: str | Path, dst: str | Path) -> AuditReport:
    """Port ``ablation_results_full.json`` to the canonical schema.

    The flagship layout is a list of ``{name, task, cells: [...]}``;
    the canonical :class:`AuditReport` flattens that to a single
    ``per_seed`` list. Hardware / seeding metadata is recorded in
    ``config``; the headline summary is precomputed under ``metrics`` so
    consumers can read 78% / 88% / 69% without re-aggregating.
    """
    import json
    src = Path(src)
    dst = Path(dst)
    with open(src) as f:
        flagship = json.load(f)

    cells: list[dict[str, Any]] = []
    for ds_block in flagship:
        for c in ds_block["cells"]:
            cells.append(
                dict(
                    variant=c["variant"],
                    dataset=ds_block["name"],
                    task=ds_block["task"],
                    H=int(c["H"]),
                    n_layers=int(c["n_layers"]),
                    head_mode=c["head_mode"],
                    mean=float(c["mean"]),
                    std=float(c["std"]),
                    raw=[float(x) for x in c.get("raw", [])],
                    mean_acc=c.get("mean_acc"),
                )
            )

    report: AuditReport = {
        "config": {
            "n_datasets": len({c["dataset"] for c in cells}),
            "n_variants": len({c["variant"] for c in cells}),
            "n_corners": len({(c["head_mode"], c["n_layers"], c["task"])
                              for c in cells}),
            "seeds": 3,
            "head_counts": sorted({c["H"] for c in cells}),
            "epochs": 600,
            "slice_only": False,
            "source_json": str(src.name),
        },
        "per_seed": cells,
        "metrics": _summary_payload(cells),
        "notes": (
            "Ported from affinity/ablation_results_full.json (flagship "
            "paper §6 sweep). Each per_seed row is a (variant, dataset, "
            "H, L, head_mode) cell averaged over 3 seeds; the raw list "
            "carries the per-seed losses."
        ),
    }
    save_report(report, dst)
    return report


# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #


def _iter_cells(report: AuditReport | Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield the per-cell records from a report, accepting either schema."""
    rows: list[dict[str, Any]] = (
        report.get("per_seed") or report.get("cells") or []  # type: ignore[assignment]
    )
    yield from rows


def _slice_dummy_result(
    dataset: str, variant: str, corner: Corner, H: int, seed: int
) -> dict[str, Any]:
    """Deterministic per-cell stub used by the smoke / slice path.

    Symmetric variants are biased to slightly lower loss than Std-Attn
    so the smoke test exercises the same code path the cached JSON
    activates. The numbers are not meant to reproduce the headline
    rates.
    """
    base = abs(hash((dataset, variant, corner.head_mode, corner.depth, H, seed)))
    loss = 0.30 + 0.05 * ((base % 1000) / 1000.0)
    if variant in SYMMETRIC_VARIANTS:
        loss -= 0.02
    if corner.head_mode == "mlp" and variant in SYMMETRIC_VARIANTS:
        loss -= 0.005
    mean_acc = None
    if corner.task == "classification":
        mean_acc = 1.0 - loss
    return {"loss": float(loss), "mean_acc": mean_acc}


def _headtohead_per_cell(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compute :math:`\\Delta = L_{\\text{Std-Attn}} - \\min_v L_v` per cell."""
    grouped: dict[tuple[str, int, int, str, str], dict[str, float]] = defaultdict(dict)
    for r in cells:
        key = (r["dataset"], int(r["H"]), int(r["n_layers"]),
               r["head_mode"], r["task"])
        grouped[key][r["variant"]] = float(r["mean"])

    out: list[dict[str, Any]] = []
    for key, per_v in grouped.items():
        std = per_v.get(STD_REFERENCE)
        sym_pairs = [(v, per_v[v]) for v in SYMMETRIC_VARIANTS
                     if v in per_v and not np.isnan(per_v[v])]
        if std is None or np.isnan(std) or not sym_pairs:
            continue
        best_v, best_loss = min(sym_pairs, key=lambda p: p[1])
        out.append(
            {
                "dataset": key[0],
                "H": key[1],
                "n_layers": key[2],
                "head_mode": key[3],
                "task": key[4],
                "delta": float(std - best_loss),
                "sym_wins": bool(std > best_loss),
                "std_loss": float(std),
                "best_sym_loss": float(best_loss),
                "best_sym_variant": best_v,
            }
        )
    return out


def _winrate_row(
    h2h: Sequence[Mapping[str, Any]],
    *,
    head_mode: str | None,
    depth: int | None,
    task: str | None,
    label: str | None = None,
) -> dict[str, Any]:
    sub = [r for r in h2h
           if (head_mode is None or r["head_mode"] == head_mode)
           and (depth is None or r["n_layers"] == depth)
           and (task is None or r["task"] == task)]
    n = len(sub)
    if n == 0:
        return {
            "head_mode": head_mode or "any",
            "depth": depth if depth is not None else "any",
            "task": task or "any",
            "label": label,
            "n_cells": 0,
            "sym_win_frac": float("nan"),
            "median_delta": float("nan"),
            "std_attn_wins": 0,
            "best_sym_wins": 0,
            "best_sym_variant": None,
        }
    deltas = np.array([r["delta"] for r in sub])
    sym_wins = int(np.sum([r["sym_wins"] for r in sub]))
    counter = Counter(r["best_sym_variant"] for r in sub)
    top = counter.most_common(1)[0][0]
    return {
        "head_mode": head_mode or "any",
        "depth": depth if depth is not None else "any",
        "task": task or "any",
        "label": label,
        "n_cells": n,
        "sym_win_frac": float(sym_wins / n),
        "median_delta": float(np.median(deltas)),
        "std_attn_wins": n - sym_wins,
        "best_sym_wins": sym_wins,
        "best_sym_variant": top,
    }


def _per_corner_winner_counts(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reproduce the §6 per-corner winner-count table."""
    grouped: dict[tuple[str, int, str], dict[tuple[str, int], dict[str, float]]] = (
        defaultdict(lambda: defaultdict(dict))
    )
    for r in cells:
        corner = (r["head_mode"], int(r["n_layers"]), r["task"])
        cell_key = (r["dataset"], int(r["H"]))
        grouped[corner][cell_key][r["variant"]] = float(r["mean"])

    out: dict[str, Any] = {}
    for corner, per_cell in grouped.items():
        counts: Counter[str] = Counter()
        for _, per_v in per_cell.items():
            valid = {v: m for v, m in per_v.items() if not np.isnan(m)}
            if not valid:
                continue
            counts[min(valid, key=lambda v: valid[v])] += 1
        key = f"{corner[0]}|{corner[1]}|{corner[2]}"
        out[key] = {"counts": dict(counts), "total": len(per_cell)}
    return out


def _summary_payload(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pre-compute the §6.2-§6.4 numbers stored under ``metrics``."""
    h2h = _headtohead_per_cell(cells)
    overall_n = len(h2h)
    overall_wins = sum(int(r["sym_wins"]) for r in h2h)
    by_head: dict[str, dict[str, float]] = {}
    for hm in {r["head_mode"] for r in h2h}:
        sub = [r for r in h2h if r["head_mode"] == hm]
        n = len(sub)
        wins = sum(int(r["sym_wins"]) for r in sub)
        by_head[hm] = {
            "n": float(n),
            "sym_win_frac": float(wins / n) if n else float("nan"),
            "median_delta": float(np.median([r["delta"] for r in sub])) if n else float("nan"),
        }
    return {
        "headline_overall_sym_win_frac": (
            float(overall_wins / overall_n) if overall_n else float("nan")
        ),
        "headline_overall_n": overall_n,
        "by_head_mode": by_head,
        "per_corner_winner_counts": _per_corner_winner_counts(cells),
    }
