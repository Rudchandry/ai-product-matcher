import argparse
import json
import os
import tempfile
from types import SimpleNamespace

import pandas as pd

from train import main


def make_tiny_csv(path):
    # Create a tiny dataset with 2 classes and 1 sample each before upsampling
    df = pd.DataFrame(
        {
            "text_a": ["Hello world", "Goodbye world"],
            "text_b": ["Hello", "Farewell"],
            "label": ["match", "no match"],
        }
    )
    df.to_csv(path, index=False)


def test_tiny_dataset_split_no_error(tmp_path):
    csv_path = tmp_path / "tiny.csv"
    make_tiny_csv(csv_path)

    args = SimpleNamespace(
        csv=str(csv_path),
        val_size=0.2,
        calibration="isotonic",
        balanced=True,
        version="testv",
        random_state=42,
    force_nonstratify=False,
    min_val_frac=0.0,
    tiny_threshold=100,
    )

    # Should not raise (no ValueError due to stratified split on tiny data)
    main(args)
