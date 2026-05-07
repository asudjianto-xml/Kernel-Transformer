"""Migrated audit modules for the *Kernels and Transformers for Tabular Data*
monograph (output of AGENT-INFRA-05).

Each :class:`tabkernels.core.base.Audit` subclass exposed here corresponds to
one of the original ``~/jupyterlab/ICL/*.py`` experiment scripts; the cached
JSON results live alongside the book under
``affinity/book/data/cached_audits/`` for figure regeneration without re-running
the audits.

The class -> source script -> chapter mapping is:

============================ ====================================== =========
Audit                        Source                                 Chapter
============================ ====================================== =========
:class:`OracleAudit`         ``oracle_kernel_audit_v2.py``          §10.6
:class:`DirectionalAudit`    ``directional_data_audit.py``          §10.7
:class:`TimeOrderedAuditAR`  ``directional_timeordered.py``         §10.7.3
:class:`PairwiseAudit`       ``asymmetric_*.py``                    §11.4-§11.5
:class:`StructureAnalysis`   ``oracle_kernel_structure_analysis.py``§13.2-§13.4
:class:`ThreeWayAblation`    ``symmetric_kernel_icl_arf.py``        §10.5.1
:class:`StandalonePureAsym`  ``icl_pure_asym_scratch.py``           §10.5.2
:class:`ResidualBoost`       ``icl_pure_asym_boost.py``             §10.5.3
:class:`DualChannelAudit`    ``icl_dual_channel.py``                §10.5.4
:class:`PostHocDecomposition`  ``icl_post_hoc_decomp.py``           §10.5.5
============================ ====================================== =========

In addition, :mod:`tabkernels.audits.benchmark` provides the head-to-head
sweep API (Chapter 12); cached results live under
``cached_audits/benchmark_360cell.json``.
"""
from .base import AuditReport, load_report, save_report, seed_all
from .benchmark import (
    ALL_VARIANTS,
    ASYMMETRIC_VARIANTS,
    Corner,
    SYMMETRIC_VARIANTS,
    aggregate_winrates,
    plot_per_corner_winners,
    run_360cell_benchmark,
)
from .cross_arch import run_cross_arch_audit
from .directional import DirectionalAudit, TimeOrderedAuditAR
from .icl_audit import (
    DualChannelAudit,
    PostHocDecomposition,
    ResidualBoost,
    StandalonePureAsym,
    ThreeWayAblation,
)
from .oracle_audit import OracleAudit
from .pairwise import PairwiseAudit
from .structure_analysis import StructureAnalysis

__all__ = [
    "ALL_VARIANTS",
    "ASYMMETRIC_VARIANTS",
    "AuditReport",
    "Corner",
    "DirectionalAudit",
    "DualChannelAudit",
    "OracleAudit",
    "PairwiseAudit",
    "PostHocDecomposition",
    "ResidualBoost",
    "SYMMETRIC_VARIANTS",
    "StandalonePureAsym",
    "StructureAnalysis",
    "ThreeWayAblation",
    "TimeOrderedAuditAR",
    "aggregate_winrates",
    "load_report",
    "plot_per_corner_winners",
    "run_360cell_benchmark",
    "run_cross_arch_audit",
    "save_report",
    "seed_all",
]
