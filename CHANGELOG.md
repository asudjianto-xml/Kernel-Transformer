# Changelog

All notable changes to `tabkernels` are recorded here. The package version is exposed as
`tabkernels.__version__`. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Added
- `tabkernels.architectures.TabPFNLite` — didactic 2-layer / 4-head TabPFN-shaped regression PFN
  with regression-friendly tokeniser and `attention_blocks()` hook for Chapter 13 inspection
  (AGENT-CH16).
- `tabkernels.directional.{DAGGenerator, ARGenerator, make_treatment_effect_data,
  causal_kernel_inference}` — directional regimes for Chapter 25; classes wrap the existing
  `tabkernels.audits.directional` generators and a new kernel-regression-based ATE estimator
  (AGENT-CH25).
- `tabkernels.priors.design.recommend_prior(architecture, task)` — dispatcher for the
  prior–architecture pairing rule of Chapter 22.
- `tabkernels.priors.{SCMPrior, ARFPrior, MLPSCMPrior, KnowledgeGraphPrior}` — four synthetic-prior
  families for Chapter 20, all subclassing `core.base.Prior`.
- `tabkernels.training.{PFNTrainer, ICLTrainer}` — Chapter 19 / 21 training loops with seed
  reproducibility, LR schedules, gradient-norm tracking, and NaN detection.
- `\argmin` / `\argmax` operators in `paper.sty`.

### Changed
- `paper.sty` now requires `cleveref` (loaded by AGENT-INFRA-02 polish).
- README rewritten with quick-start example, module table, and citation block.

### Internal
- `.gitignore` now ignores per-host pytest coverage files (`.coverage.*`).
- Test suite expanded: 224+ tests passing, 91% line coverage.

## 0.1.0a1 — Initial skeleton (AGENT-INFRA-01)

- Package skeleton: `tabkernels.{core, classical, attention, sparsifiers, symmetrizers, diffusion,
  asymmetric, composite, kernels, audits, transparency, training, priors, architectures, viz, data,
  benchmarks}`.
- Locked API contract in `tabkernels.core.base`: `Kernel`, `Sparsifier`, `Symmetrizer`,
  `Predictor`, `AttentionBlock`, `Architecture`, `Prior`, `Audit`.
- pyproject.toml, LICENSE (MIT), pytest configuration, GitHub Actions workflows.
