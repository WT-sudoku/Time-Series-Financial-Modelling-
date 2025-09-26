# GRU-Based Upper-Band Forecasting
**Notebook:** ``  
**Kernel:** Python 3  
**Language:** python 

## Project Overview
This repository contains the notebook used for the mini-project implements a time-series forecasting pipeline to predict an *upper band* for the next-day High price of an equity using a GRU-based neural network. The modeling approach targets the next-day Close→High ratio (often in log space), with training via a quantile/pinball loss at a high quantile (e.g., τ=0.90) to estimate an upper bound. Feature engineering can include exogenous signals (e.g., index levels, volatility proxies), and model blending weights are selected via rolling-origin validation focused on recent slices of history.

## Notebook Structure (Headings)
The following headings were detected in the notebook (first 60 shown):
- # Table of Content:
- # Check GPU availability
- # Enable and Read the content from Drive.
- # Scrap the Data from yahoo finance:
- # Load the Dataset
- # Preprocessing to parse 'date'
- # Main Source Code
- # Stand-alone Fine-tuning code
- # Stand-alone Fine-tuning code (Modified New)
- # Fine-Tuning Metrics Performance Evaluation:
- # Diebold–Mariano (DM)
- # A New Proposed GRU Model Methodology:
- ## Based on the Main Code & Fine-Tuning Outputs Report Findings:
- # * How to use: run this as stand-alone code. It will execute even if the optional exogenous dataset CSVs is not provided.
- # Diebold–Mariano (DM)

## Code Architecture at a Glance
- **Functions defined:** 21 (_init_precomp_validation_constants, _make_ds, _vectorized_windowize, build_gru, dm_test, loss, make_model, make_windows, maybe_merge_csv, naive_high, naive_levels, norm_cdf, one_step_levels, pinball_loss, prepare_windows_for, ratio_to_level, sample_params, scale_block, series_metrics, validate_params ...)
- **Classes defined:** 0 ()
- **Keras/TF layers referenced:** Dense, Dropout, GRU, Input, LSTM
- **Losses referenced in `.compile()` blocks:** huber
- **Metrics referenced in `.compile()` blocks:** Not detected in static scan
- **Random seeds set (detected patterns):** 0

## Dependencies
- matplotlib
- numpy
- pandas
- scikit-learn
- tensorflow
- torch

> Tip: After running successfully, consider freezing exact versions:
> ```bash
> pip freeze > requirements.txt
> ```

## Data Inputs
No explicit `pd.read_csv()` file paths were detected in a static scan of the notebook.

If your data is not listed above, the dataset paths may be provided dynamically or loaded from other sources (APIs/URLs/Google Drive/Colab). Review the corresponding data-loading cells to confirm.

## Artifacts & Outputs
- /content/drive/MyDrive/AAPL_stock_history.csv
- stock_history_downloaded.csv

## How to Run
### Run on Google Colab
1. Open the notebook `AIB552_GBA_JULY2025_Group_01-9.ipynb` in Google Colab.
2. (Optional) Enable GPU: *Runtime → Change runtime type → Hardware accelerator: GPU*.
3. Install dependencies in a cell if missing, e.g.:
   ```python
   !pip install matplotlib numpy pandas scikit-learn tensorflow torch
   ```
4. Upload or mount your data as required (see **Data Inputs**).
5. Run the cells top-to-bottom, adjusting configuration (paths/parameters) where needed.

### Run Locally (Jupyter)
1. Create and activate a virtual environment (recommended).
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install --upgrade pip
   pip install matplotlib numpy pandas scikit-learn tensorflow torch notebook
   ```
2. Start Jupyter and open the notebook:
   ```bash
   jupyter notebook
   ```
3. Update data paths under the **Data Inputs** section and run the notebook sequentially.

## Modeling Details (from code scan)
- This notebook appears to use the following deep learning layers: **Dense, Dropout, GRU, Input, LSTM**.
- Training is likely done with `.fit(...)` calls; 10 `fit` block(s) were detected.
- Loss functions referenced in compile calls: **huber**.
- Common evaluation metrics detected: **Not detected in static scan**.

If your objective is the *upper band* (e.g., 90th percentile) of next-day High, ensure the loss is a **pinball/quantile** loss with τ close to your target bound (e.g., 0.90).

## Reproducibility
- Ensure seeds are set across NumPy/TensorFlow/Python for repeatability.
- Keep a note of pandas/NumPy/TensorFlow versions.
- Consider saving `model.summary()` and configuration (hyperparameters, pinball τ) into the repo for traceability.

## Suggested Repo Layout
```
.
├── notebooks/
│   └── AIB552_GBA_JULY2025_Group_01-9.ipynb
├── data/                # (gitignored) raw / processed data
├── models/              # saved weights / artifacts
├── reports/             # figures, charts
├── README.md
└── requirements.txt
```
Add a `.gitignore` entry for large or private files (see below).

## .gitignore (suggested)
```
data/*
models/*
reports/*
*.ckpt
*.h5
*.hdf5
*.pt
.ipynb_checkpoints/
.DS_Store
.venv/
```

## Results & Evaluation
Review the latter sections of the notebook for plots and metrics (RMSE/MAE/R² if computed). If applicable, summarize:
- Windowing & feature set (lags/exogenous variables)
- Train/validation split and rolling-origin validation
- Best model hyperparameters (GRU units, layers, dropout, learning rate)
- Final performance and error bands

## How to Cite
If this work is part of SUSS AIB552 coursework, cite your sources (texts, code snippets, datasets). Example references:
- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*
- Goodfellow et al., *Deep Learning*
- Relevant financial datasets/APIs used

---

*This README was generated via static analysis of the notebook. Please verify paths, parameters, and sections against your final code & results.*


## Function Reference (Auto-Extracted)
- `scale_block(block)`  *(defined in code cell #13)* — Apply train_core scalers to a block of rows; keeps validation/test clean.
- `make_windows(X2d, y2d, n_steps)`  *(defined in code cell #13)* — Turn a 2D array [T, F] into sequences [N, n_steps, F] with aligned labels [N, 1].
- `make_model(n_steps, n_features)`  *(defined in code cell #13)* — GRU with Huber loss: small, fast, less prone to overfit, and robust to spikes (earnings days).
- `one_step_levels(ret_pred, last_level, true_levels)`  *(defined in code cell #13)* — Map predicted Δlog(High) to a *level* forecast using the ACTUAL last price.
- `naive_levels(last_level, true_levels)`  *(defined in code cell #13)* — Persistence baseline: tomorrow’s High equals today’s High.
- `series_metrics(y_true, y_pred, y_naive)`  *(defined in code cell #13)* — Compute:
- `windowize(X2d, y2d, n_steps)`  *(defined in code cell #15)* — Convert a 2D array of rows (time x features) into overlapping sequences for RNNs.
- `prepare_windows_for(n_steps)`  *(defined in code cell #15)* — Build train/val/test windows for a *given* lookback length, with context bridging.
- `build_gru(n_steps, n_features, units, layers, dropout, rdrop, l2, lr, huber_delta, clipnorm, layernorm)`  *(defined in code cell #15)* — Construct a configurable GRU model:
- `validate_params(params, n_steps)`  *(defined in code cell #15)* — Train a model with a candidate hyperparameter set and score it on validation
- `sample_params()`  *(defined in code cell #15)* — Randomly sample one configuration (except n_steps, which is sampled separately).
- `_vectorized_windowize(X2d, y2d, n_steps)`  *(defined in code cell #17)* — Vectorized rolling windows:
- `_init_precomp_validation_constants()`  *(defined in code cell #17)* — Compute constants used in every trial once.
- `prepare_windows_for(n_steps)`  *(defined in code cell #17)* — Build/cached train/val/test windows for a given lookback length.
- `build_gru(n_steps, n_features, units, layers, dropout, rdrop, l2, lr, huber_delta, clipnorm, layernorm, jit_compile)`  *(defined in code cell #17)*
- `_make_ds(X, y, batch, shuffle)`  *(defined in code cell #17)*
- `validate_params(params, n_steps)`  *(defined in code cell #17)* — Train a model for the sampled hyperparameters; score VALIDATION in level space.
- `sample_params()`  *(defined in code cell #17)*
- `dm_test(loss_diff, h)`  *(defined in code cell #22)* — Simple DM test with Newey-West variance (lag = h-1 for 1-step ahead -> 0).
- `norm_cdf(x)`  *(defined in code cell #22)*
- `maybe_merge_csv(left, path, date_col, how, rename, select, prefix)`  *(defined in code cell #25)* — Try to merge an external CSV by date. If the file doesn't exist, return 'left' unchanged.
- `scale_block(block)`  *(defined in code cell #25)*
- `make_windows(X2d, y2d, n_steps)`  *(defined in code cell #25)*
- `pinball_loss(tau)`  *(defined in code cell #25)*
- `make_model(n_steps, n_features)`  *(defined in code cell #25)*
- `ratio_to_level(y_ratio, prev_close)`  *(defined in code cell #25)* — Map predicted log ratio to a *level*:
- `naive_high(prev_high)`  *(defined in code cell #25)* — Persistence baseline: High_hat_t^naive = High_{t-1}.
- `series_metrics(y_true, y_pred, y_naive)`  *(defined in code cell #25)*
- `loss(y_true, y_pred)`  *(defined in code cell #25)*
- `dm_test(loss_diff, h)`  *(defined in code cell #30)* — Simple DM test with Newey-West variance (lag = h-1 for 1-step ahead -> 0).
- `norm_cdf(x)`  *(defined in code cell #30)*

## Training Configuration (from code)
**Compile #1**:
```python
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3)
```
**Compile #2**:
```python
model.compile(optimizer=opt,
                  loss=tf.keras.losses.Huber(delta=huber_delta)
```
**Compile #3**:
```python
model.compile(optimizer=opt,
        loss=tf.keras.losses.Huber(delta=huber_delta)
```
**Compile #4**:
```python
model.compile(optimizer=tf.keras.optimizers.Adam(3e-4)
```

**Fit #1**:
```python
model.fit(train_core[X_cols].values)
```
**Fit #2**:
```python
model.fit(train_core[["y"]].values)
```
**Fit #3**:
```python
model.fit(Xtr, ytr,
    validation_data=(Xv, yv)
```
**Fit #4**:
```python
model.fit(Xtr, ytr, validation_data=(Xv, yv)
```
**Fit #5**:
```python
model.fit(Xtr, ytr, validation_data=(Xv, yv)
```
**Fit #6**:
```python
model.fit(ds_tr, validation_data=ds_val, epochs=80, callbacks=[es, rlr], verbose=0)
```
**Fit #7**:
```python
model.fit(ds_tr, validation_data=ds_val,
               epochs=120, callbacks=[es, rlr], verbose=1)
```
**Fit #8**:
```python
model.fit(train_core[X_cols].values)
```
**Fit #9**:
```python
model.fit(train_core[["y"]].values)
```
**Fit #10**:
```python
model.fit(Xtr, ytr,
    validation_data=(Xv, yv)
```
