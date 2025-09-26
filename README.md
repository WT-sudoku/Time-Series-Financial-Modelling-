# GRU-Time-Series-Financial-Modelling

> Auto-generated README based on the project notebook. Edit this file to refine details as needed.

## Overview

Tasks: Computer Vision (CNN), Natural Language Processing (RNN/Transformers). Custom models: KerasSequential. Datasets mentioned: csv, stock. Tech stack: PyTorch, TensorFlow, scikit-learn, pandas, NumPy, Matplotlib.

## Table of Contents (from the notebook)

- [Table of Contents (Cell 1):](#table-of-contents-cell-1)
- [Check GPU availability](#check-gpu-availability)
- [Enable and Read the content from Drive.](#enable-and-read-the-content-from-drive)
- [Load the Dataset](#load-the-dataset)
- [Main Source Code: Business-Commented GRU Baseline — One-Day-Ahead High](#main-source-code-business-commented-gru-baseline--one-day-ahead-high)
- [Title & Objective: GRU Fine-Tuning (Random Search, accelerated) — minimize Validation MASE vs naïve; evaluate on Test](#title--objective-gru-fine-tuning-random-search-accelerated--minimize-validation-mase-vs-nave-evaluate-on-test)
- [Diebold–Mariano (DM)](#dieboldmariano-dm)
- [Areas of Targeted Improvement: New Proposed GRU Model Methodology:](#areas-of-targeted-improvement-new-proposed-gru-model-methodology)
- [Title & Objective: Upper-band forecasting for next-day High using Close→High log-ratio with pinball loss (τ = 0.90); build a risk-aware blend vs naïve.](#title--objective-upper-band-forecasting-for-next-day-high-using-closehigh-log-ratio-with-pinball-loss---090-build-a-risk-aware-blend-vs-nave)
- [Diebold–Mariano (DM)](#dieboldmariano-dm)
- [7. Business insights & storytelling:](#7-business-insights--storytelling)

## Getting Started

### Option 1: Run in Google Colab

1. Open the notebook in Colab.
2. Go to **Runtime → Change runtime type** and select GPU if available.
3. Run cells top-to-bottom.

### Option 2: Run Locally

```bash
# (Optional) create a clean environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install google math matplotlib numpy os pandas random scikit-learn tensorflow torch
# Launch Jupyter
pip install jupyter
jupyter notebook
```

## Data

This project references/mentions the following datasets or sources (detected heuristically from code):

- csv
- stock

> Update this section with exact dataset download links/paths as appropriate.

## Models & Training

Custom architectures detected:

- KerasSequential

Primary tasks:

- Computer Vision (CNN)
- Natural Language Processing (RNN/Transformers)

Training loops, hyperparameters, and evaluation steps are implemented in the notebook cells.

## Dependencies

This notebook imports:

- PyTorch
- TensorFlow
- scikit-learn
- pandas
- NumPy
- Matplotlib

A quick-start install line (edit as needed):

```bash
pip install google math matplotlib numpy os pandas random scikit-learn tensorflow torch
```

## Repository Layout

- `README.md` — this file

> If you add scripts (`/src`), data (`/data`), or configs, document them here.

## Reproducibility Tips

- Set random seeds where applicable (NumPy/PyTorch/TensorFlow).
- Log package versions (`pip freeze > requirements.txt`).
- For large artifacts (models/data), consider Git LFS or external storage.

## License

No license specified yet. Consider adding `LICENSE` (e.g., MIT) to clarify reuse terms.

## Acknowledgements

This work was prepared for a mini-project.
