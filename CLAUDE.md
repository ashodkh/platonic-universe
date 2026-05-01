# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Platonic Universe** tests the Platonic Representation Hypothesis (PRH) on astronomical data: the idea that neural networks trained on different modalities and objectives converge toward a shared statistical model of reality. The project embeds galaxy images/spectra with various vision models (ViT, DINOv2, ConvNeXtv2, AstroPT, Specformer, etc.) and measures cross-modal representational similarity.

## Commands

```bash
# Install (uv recommended)
pip install uv && uv sync
uv pip install .          # base
uv pip install ".[sam2]"  # with SAM2 support

# Run experiments (CLI)
platonic_universe run --model vit --mode jwst --batch-size 128 --knn-k 10
platonic_universe compare <parquet_file> --metrics cka mknn
platonic_universe run-physics --model vit --split test

# Tests
pytest tests/
pytest tests/test_alignment.py -v   # batch invariance + determinism

# Lint
ruff check src/
```

## Architecture

### Data flow

```
HuggingFace Hub (Smith42/* datasets)
    → DatasetAdapter.prepare()        # loads + preprocesses to HF Dataset
    → ModelAdapter.embed_for_mode()   # batch embeddings via GPU
    → Parquet file (embeddings)
    → metrics/io.py compare_from_parquet()
    → MKNN / CKA / etc. → results dict + figs/
```

### Key abstractions

**`src/pu/models/base.py` — `ModelAdapter`**  
Abstract base for all models. Subclasses must implement `load()`, `get_preprocessor()`, and `embed_for_mode()`. Supports layer-wise extraction via forward hooks (`_capture_module_outputs`) at four granularities (BLOCKS, RESIDUAL, LEAVES, ALL). All model adapters self-register via `register_adapter(alias, cls)` and are discovered through `src/pu/models/registry.py`.

**`src/pu/pu_datasets/base.py` — `DatasetAdapter`**  
Abstract base for datasets. `prepare(processor, modes, filterfun)` returns a preprocessed HuggingFace `Dataset`. Datasets self-register via the same registry pattern.

**`src/pu/metrics/`**  
Standalone metrics library. Primary metric used in the paper is MKNN (`neighbors.py`, default k=10). CKA (`kernel.py`) is used for validation. `io.py` handles Parquet I/O and batched metric computation via `compare_from_parquet`. All metrics return floats; most are in [0, 1].

**`src/pu/experiments.py`**  
Main orchestration: loads model + two dataset adapters (HSC as reference + comparison mode), generates embeddings, computes metrics, optionally runs physics validation and saves visualizations.

**`src/pu/preprocess.py` + `zoom.py`**  
Image preprocessing: converts astronomical flux arrays → PIL RGB via `flux_to_pil`. `resize_galaxy_to_fit` supports two strategies: `match` (fixed extents aligned to reference survey) and `fill` (Otsu-based cropping so galaxy fills frame). Normalization percentiles live in `data/percentiles.json`.

### Model families

- **HuggingFace** (`models/hf.py`): ViT, DINOv2, DINOv3, ConvNeXtv2, CLIP, LLaVA, PaliGemma, IJEPA
- **Astronomy-specific** (`models/astropt.py`, `models/specformer.py`, `models/aion.py`): AstroPT, Specformer, AION/Polymathic

### Datasets

All data is loaded from HuggingFace Hub. Crossmatched datasets pair HSC (reference optical imaging) with a comparison modality (JWST infrared, Legacy Survey, DESI/SDSS spectra). Column convention: `{mode}_image` for image data.

### Physics validation (`physics_experiment.py`, `metrics/physics.py`)

Tests whether embeddings encode known galaxy properties (stellar mass, redshift, Sérsic index, SFR, morphology fractions) via linear/nonlinear probing with cross-validation.

## Important test properties

- **Batch-size invariance**: embeddings for a given sample must be identical regardless of batch size (tested in `test_alignment.py`).
- **Determinism**: two runs with the same seed must produce bit-identical embeddings.
- **Fingerprint ordering**: HF Dataset fingerprints verify the data pipeline hasn't shuffled samples.
