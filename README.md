pair-matcher
===========

Simple example to train a text pair matching classifier (TF-IDF + LogisticRegression).

Usage
-----

Train from a CSV (requires `text_a`/`text_b` or detected text columns and a `label` column):

```powershell
python train.py --csv data/pairs.csv --version v1
```

Key CLI options
---------------
- `--val_size` : validation split size (float 0-1 or absolute int). Default: `0.2`.
- `--calibration` : `none|sigmoid|isotonic`. Note: `isotonic` needs more data; for very small datasets prefer `--calibration sigmoid` or `--calibration none`.
- `--balanced` : upsample classes to at least 2 samples (useful for tiny datasets).
- `--force-nonstratify` : force a non-stratified train/validation split even when stratification would be possible.
 - `--min_val_frac` : minimum validation fraction to enforce for small datasets (0-1). Default: `0.0` (disabled).
 - `--tiny_threshold` : dataset size below which `--min_val_frac` is applied. Default: `100`.

Notes for tiny datasets
-----------------------
- The script automatically ensures at least 2 samples per class when `--balanced` is used.
- When the dataset is extremely small, the script will adjust the validation set size so stratification is possible, or fall back to a non-stratified split. Use `--force-nonstratify` to explicitly opt out of stratification.
- Isotonic calibration is unreliable on very small training sets; prefer `--calibration sigmoid` or `--calibration none` when using tiny datasets.

Minimum validation fraction behavior
-----------------------------------
- The `--min_val_frac` and `--tiny_threshold` flags let you enforce a minimum validation set size for small datasets.
	- Example: `--min_val_frac 0.2 --tiny_threshold 50` ensures at least 20% of the data is reserved for validation when the dataset has fewer than 50 samples.
	- Defaults leave behavior unchanged (`--min_val_frac 0.0` disables enforcement).
	- The script prints a warning when it increases the test set size to satisfy `--min_val_frac`.

Testing
-------
Run tests (recommended inside the project's venv):

```powershell
& ".\.venv\Scripts\python.exe" -m pytest -q
```

Artifacts
---------
Models and metadata are written to `artifacts/<version>/`.

CSV format examples
-------------------
The training script is flexible about column names. Below are a few common CSV layouts that work.

1) Two text columns and explicit `label` column (preferred):

```csv
text_a,text_b,label
"How do I reset my password?","Reset password instructions",match
"Order status","Where is my order?",no match
```

2) Alternate column names (the script will try common names):

```csv
left,right,label
"Title A","Title B",match
"Query","Candidate",no match
```

3) Single text column and label (the script will use the lone text column):

```csv
text,label
"Short sentence about product",match
"Unrelated sentence",no match
```

Label values and mapping
------------------------
- The script normalizes common label strings. Examples it maps automatically:
	- "match", "1", "true" -> Match
	- "no match", "0", "false" -> No match
- If your labels use other strings, the script will fall back to `LabelEncoder` and treat each distinct value as a class.

Tips
----
- For very small datasets prefer `--balanced` to upsample classes for training.
- If you see warnings about isotonic calibration on small data, use `--calibration sigmoid` or `--calibration none`.

Deployment / Procfile
---------------------
- A `Procfile` is included for platforms that use it (Heroku/EB). To run the app with the provided entrypoint use:

```powershell
web: uvicorn main:app --host 0.0.0.0 --port 8000
```

