# -*- coding: utf-8 -*-
"""
conftest.py — shared pytest fixtures for the neurotransmitter pipeline tests.
"""

import io
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def receptor_csv_bytes():
    """Minimal in-memory receptor CSV matching the real file schema."""
    df = pd.DataFrame({
        "label":  [1001, 1002, 17, 18],
        "region": [
            "ctx-lh-entorhinal",
            "ctx-rh-parahippocampal",
            "Left-Hippocampus",
            "Right-Hippocampus",
        ],
        "D1":       [ 0.80,  1.20, -0.30,  0.10],
        "D2":       [ 0.50,  0.90,  0.40, -0.20],
        "DAT":      [ 0.10,  0.30,  0.70,  0.60],
        "5HT1a":    [ 1.10, -0.50,  0.20,  0.30],
        "GABAa":    [-0.20,  0.40,  0.80,  0.50],
        "mGluR5":   [ 0.60,  0.70, -0.10,  0.20],
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf.read()


@pytest.fixture(scope="session")
def receptor_df(receptor_csv_bytes):
    return pd.read_csv(io.BytesIO(receptor_csv_bytes))


@pytest.fixture(scope="session")
def longitudinal_df():
    """Simulated longitudinal ADNI-style dataframe."""
    return pd.DataFrame({
        "patient_id":    [1, 1, 1, 2, 2, 3],
        "timepoint":     ["bl", "m12", "m24", "bl", "m24", "bl"],
        "dementia_dx_bl":["CON", "CON", "CON", "MCI", "MCI", "AD"],
        "dementia_dx":   ["MCI", "MCI", "AD",  "AD",  "AD",  "AD"],
        "lh_entorhinal": [1.0,   1.5,   2.0,   0.8,   1.2,  -0.5],
        "rh_entorhinal": [0.5,   0.9,   1.3,   0.4,   0.9,  -0.2],
    })
