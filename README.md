# TrasMuon

[![arXiv](https://img.shields.io/badge/arXiv-2602.13498-b31b1b.svg)](https://arxiv.org/abs/2602.13498)

Optimizer implementation for **TrasMuon: Trust-Region Adaptive Scaling for Orthogonalized Momentum Optimizers**. TrasMuon combines Muon with adaptive scaling, trust-region column gating, and schedule-free smoothing.

![TrasMuon algorithm: adaptive scaling, trust-region column gating, and schedule-free smoothing](https://raw.githubusercontent.com/zangjiucheng/TrasMuon/main/assets/trasmuon-algorithm.png)

## Installation

Requires Python ≥ 3.10, PyTorch ≥ 2.5.1, and Triton. Use Linux with CUDA-enabled PyTorch for GPU execution.

Install from GitHub:

```bash
pip install git+https://github.com/zangjiucheng/TrasMuon.git
```

After the first PyPI release, install with:

```bash
pip install trasmuon-pytorch
```

## Usage

```python
from trasmuon import TrasMuon

optimizer = TrasMuon(model.parameters(), lr=1e-3, weight_decay=0.01)

optimizer.zero_grad(set_to_none=True)
loss = loss_fn(model(inputs), targets)
loss.backward()
optimizer.step()
```

Two-dimensional parameters use TrasMuon; other parameter shapes use the optimizer's AdamW fallback. See [`TrasMuon.__init__`](https://github.com/zangjiucheng/TrasMuon/blob/main/trasmuon/optimizer.py) for available options.

## Files

- [`trasmuon/optimizer.py`](https://github.com/zangjiucheng/TrasMuon/blob/main/trasmuon/optimizer.py): optimizer implementation.
- [`trasmuon/newton_schulz_triton.py`](https://github.com/zangjiucheng/TrasMuon/blob/main/trasmuon/newton_schulz_triton.py): required Newton–Schulz kernel dependency.
- [`trasmuon/__init__.py`](https://github.com/zangjiucheng/TrasMuon/blob/main/trasmuon/__init__.py): exports `TrasMuon`.

## Publishing

Build and validate a release locally:

```bash
python -m pip install build twine
python -m build
python -m twine check --strict dist/*
```

For the first release, add a [pending Trusted Publisher](https://pypi.org/manage/account/publishing/) on PyPI with:

- Project: `trasmuon-pytorch`
- Owner / repository: `zangjiucheng` / `TrasMuon`
- Workflow: `publish.yml`
- Environment: `pypi`

Create the matching `pypi` environment in the GitHub repository settings. Update the version in `pyproject.toml`, then publish a GitHub Release with the matching tag (initially `v0.1.0`). The release workflow builds and validates both distributions, then publishes them through Trusted Publishing. No PyPI API token is needed. See the [PyPI setup guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

## Citation

If you use this code, cite the TrasMuon paper:

```bibtex
@misc{cheng2026trasmuon,
  title = {TrasMuon: Trust-Region Adaptive Scaling for Orthogonalized Momentum Optimizers},
  author = {Peng Cheng and Jiucheng Zang and Qingnan Li and Liheng Ma and Yufei Cui and Yingxue Zhang and Boxing Chen and Ming Jian and Wen Tong},
  year = {2026},
  eprint = {2602.13498},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2602.13498}
}
```

## License

This repository is released under the MIT license. See [LICENSE](https://github.com/zangjiucheng/TrasMuon/blob/main/LICENSE) and [NOTICE.md](https://github.com/zangjiucheng/TrasMuon/blob/main/NOTICE.md).
