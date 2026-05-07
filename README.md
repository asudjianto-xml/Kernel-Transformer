# tabkernels

Companion Python package for the monograph *Kernels and Transformers for Tabular Data*.

The package gives every architectural and theoretical claim in the book a runnable implementation:
classical kernel methods, attention as a parametric kernel smoother, sparsifiers, symmetrisers,
diffusion / asymmetric / composite kernels, the post-hoc decomposition diagnostic, and a small but
faithful TabPFN-shaped foundation model with PFN-style training.

## Status

Pre-alpha. APIs are stabilising chapter by chapter; pin a specific commit when reproducing book results.

## Install

From a clone of the [Kernel-Transformer](https://github.com/asudjianto-xml/Kernel-Transformer) repository:

```bash
pip install .
# or, for development with tests / docs / benchmarks tooling:
pip install -e .[dev,docs,benchmarks]
```

Optional extras: `dev` pulls in pytest, ruff, mypy, papermill, jupytext;
`docs` pulls in Sphinx + RTD theme; `benchmarks` pulls in OpenML and XGBoost
for the head-to-head comparisons of Chapter 12.

PyPI publication is pending; once live, `pip install tabkernels` will work directly.

## Tutorial notebooks

The 26 chapter-companion notebooks ship inside the wheel. After install, copy them
into a working directory of your choice:

```bash
tabkernels-notebooks --copy ~/tabkernels-tutorials
# list bundled notebooks:
tabkernels-notebooks --list
```

The copy command writes each `NN_topic.ipynb` (e.g.\ `10_symmetry_suffices.ipynb`,
`16_tabpfn.ipynb`, `18_kernel_lens.ipynb`) into the destination directory. Notebooks
ship with outputs cleared; re-run them under `jupyter` (or via `papermill`) to
reproduce the book's figures and tables.

## Quick start

A two-line PFN that approximates the Bayesian posterior of a Gaussian linear-regression prior:

```python
import torch
from tabkernels.architectures import TabPFNLite
from tabkernels.priors import SCMPrior, SCMConfig
from tabkernels.training import PFNTrainer

prior = SCMPrior(SCMConfig(structural="mlp", edge_prob=0.5, noise_scale=0.3))
model = TabPFNLite(d_in=4, d_model=64, n_heads=4, n_layers=2)
trainer = PFNTrainer(prior=prior, model=model,
                     n_steps=1500, n_ctx=48, n_query=24, d=4, lr=3e-3)
trainer.train()

X_ctx, y_ctx, X_q, y_q = prior.sample_episode(n_ctx=48, n_query=24, d=4, seed=0)
y_pred = model(X_q, X_ctx, y_ctx)        # single forward pass — no per-task fitting
```

Apply the post-hoc decomposition diagnostic to the trained model (Chapter 13):

```python
from tabkernels.transparency import decompose_attention, kernel_energy_split
for block in model.attention_blocks():
    out = decompose_attention(block)
    for h in range(out["B"].shape[0]):
        a_s, a_a = kernel_energy_split(out["B"][h])
        print(f"head {h}: alpha_S={a_s:.3f}  alpha_A={a_a:.3f}")
```

## Modules

| Module | Chapter(s) | Contents |
|---|---|---|
| `tabkernels.classical` | Ch 2 | kNN, NW, KRR, RBF |
| `tabkernels.attention` | Ch 3 | five attention blocks (Std, SymPSD, SymGen, PureAsym, Dual) + decomposable variant |
| `tabkernels.sparsifiers` | Ch 5 | kNN, sparsemax/entmax, Sinkhorn-top-k, $\epsilon$-ball |
| `tabkernels.symmetrizers` | Ch 6 | additive / max-or / mutual-and / Sinkhorn |
| `tabkernels.diffusion` | Ch 7 | heat, regularised Laplacian, PPR/APPNP, $p$-step, Chebyshev, commute-time, learnable spectral |
| `tabkernels.asymmetric` | Ch 8 | bilinear $QK$, Bregman, directed Laplacian |
| `tabkernels.composite` | Ch 9 | multi-head, multi-scale, multi-kernel |
| `tabkernels.audits` | Ch 10–13 | symmetry-suffices, directional, pairwise, oracle, structure, benchmark, ICL audits |
| `tabkernels.transparency` | Ch 13 | post-hoc decomposition diagnostic |
| `tabkernels.architectures` | Ch 14–17 | didactic FT-Transformer, SAINT, TabPFN, TabICL re-implementations |
| `tabkernels.training` | Ch 19, 21 | `PFNTrainer`, `ICLTrainer` |
| `tabkernels.priors` | Ch 20, 22 | SCM, ARF, MLP-SCM, KG, dispatcher |
| `tabkernels.directional` | Ch 25 | DAG / AR generators + causal-effect demo |

## Notebooks

Every chapter has a companion notebook (`NN_topic.ipynb`) that reproduces the chapter's
figures and tables. They are **bundled inside the installed package** — extract with:

```bash
tabkernels-notebooks --copy ~/tabkernels-tutorials
cd ~/tabkernels-tutorials
jupyter notebook  # or jupyter lab
```

To re-execute them in bulk (e.g. when reproducing the book end-to-end):

```bash
for nb in *.ipynb; do
    jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

Working from a repository clone instead of an installed package? The notebook source
lives in `notebooks/` at the repository root.

## Tests

```bash
pytest -q
```

The `main` branch tracks a passing test suite of more than 200 tests with line coverage above 90%.

## Documentation

- Book chapters: `../affinity/book/chapters/CH*.tex`
- API reference: `../affinity/book/appendices/C_api_reference.tex`
- Reproducibility protocol: `../affinity/book/appendices/B_reproducibility.tex`

## Citation

```bibtex
@unpublished{sudjianto2026kernels-and-transformers,
  title  = {Kernels and Transformers for Tabular Data},
  author = {Sudjianto, Agus},
  year   = {2026},
  note   = {Monograph in preparation. Companion package: tabkernels.}
}
```

## License

MIT — see [LICENSE](LICENSE).
