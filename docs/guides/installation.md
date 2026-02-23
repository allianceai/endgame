# Installation

## Requirements

- Python 3.10, 3.11, or 3.12
- pip >= 23.0 recommended

## Quick Install

Install the core library with no optional extras:

```bash
pip install endgame-ml
```

The core install includes: numpy >= 1.24, polars >= 0.20, scikit-learn >= 1.3, optuna >= 3.4,
scipy >= 1.10, and networkx >= 3.0. This covers validation, preprocessing, basic models,
ensemble, calibration, tuning, anomaly detection, and semi-supervised learning.

## Optional Dependency Groups

Install extras based on the domains you need:

```bash
pip install endgame-ml[tabular]      # GBDTs (XGBoost, LightGBM, CatBoost), PyTorch tabular models, EBM, TabPFN
pip install endgame-ml[calibration]  # Conformal prediction and probability calibration (scipy)
pip install endgame-ml[vision]       # Image backbones, TTA, WBF (timm, torchvision, albumentations, smp)
pip install endgame-ml[nlp]          # Transformers, DAPT, LLM utilities (HuggingFace, bitsandbytes)
pip install endgame-ml[audio]        # Spectrogram extraction, SED models (torchaudio, librosa)
pip install endgame-ml[benchmark]    # OpenML suite loading, meta-learning, synthetic data (openml, pymfe)
pip install endgame-ml[explain]      # SHAP, LIME, DiCE counterfactuals (shap, lime, dice-ml)
pip install endgame-ml[fairness]     # Fairness metrics and bias mitigation (fairlearn)
pip install endgame-ml[deployment]   # ONNX export, model serving (onnx, onnxruntime, skl2onnx, hummingbird-ml)
pip install endgame-ml[tracking]     # Experiment tracking (mlflow)
pip install endgame-ml[mcp]          # MCP server for tool-use integrations (mcp >= 1.2.0)
pip install endgame-ml[all]          # All of the above
```

Multiple extras can be combined:

```bash
pip install endgame-ml[tabular,explain,fairness]
```

## Development Install

Clone the repository and install in editable mode with development tools:

```bash
git clone https://github.com/allianceai/endgame.git
cd endgame
pip install -e ".[dev]"
```

The `dev` extra installs: pytest, pytest-cov, ruff, and mypy.

Run the test suite to confirm everything works:

```bash
pytest tests/ -v
```

## Verify Installation

```python
python -c "import endgame; print(endgame.__version__)"
```

Expected output: `1.0.0`

## Platform Notes

### Linux

Linux is the primary development and test platform. All extras are supported. For GPU
acceleration install the CUDA-enabled PyTorch build before installing endgame extras:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install endgame-ml[tabular,vision,nlp,audio]
```

Replace `cu121` with the CUDA version that matches your driver (check with `nvidia-smi`).

### macOS

All extras are supported on macOS (Apple Silicon and Intel). LightGBM requires OpenMP,
which is not bundled with the Apple toolchain. Install it via Homebrew before running
`pip install endgame-ml[tabular]`:

```bash
brew install libomp
```

If you see `OMP: Error #15` at runtime, set the following environment variable:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

GPU acceleration on Apple Silicon uses the MPS backend (no CUDA required). PyTorch
will use MPS automatically when available.

### Windows

Core and most extras work on Windows. A C++ build toolchain (Visual Studio Build Tools
or the "Desktop development with C++" workload) is required for packages that compile
native extensions, including LightGBM, CatBoost, and several signal processing
dependencies. Install the build tools from:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

## Troubleshooting

### PyTorch CUDA version mismatch

If you see errors like `CUDA error: no kernel image is available` or torch reports
`cuda.is_available() == False` despite a GPU being present, your installed PyTorch
was built against a different CUDA version than your driver provides. Reinstall PyTorch
using the correct wheel:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Check https://pytorch.org/get-started/locally/ for the correct index URL for your
CUDA version.

### PySR / SymbolicRegressor requires Julia

`eg.models.symbolic.SymbolicRegressor` is backed by PySR, which requires a Julia
runtime. Julia is downloaded automatically by PySR on first use, but this requires
internet access and write permissions. To pre-install:

```bash
pip install juliacall
python -c "import juliacall"   # triggers Julia download
pip install pysr
```

If Julia install fails in a restricted environment, set `JULIA_DEPOT_PATH` to a
writable directory and ensure outbound HTTPS is available.

### LightGBM on macOS — libomp not found

See the macOS section above. The short fix is `brew install libomp`. If Homebrew is
not available, build LightGBM from source with the bundled OpenMP:

```bash
pip install lightgbm --install-option=--mpi
```

### ImportError for heavy modules

Heavy modules (vision, nlp, audio, benchmark, kaggle, quick) are lazy-loaded and only
imported on first access. If you see an `ImportError` when accessing `eg.vision` or
similar, install the corresponding extra:

```bash
pip install endgame-ml[vision]
```
