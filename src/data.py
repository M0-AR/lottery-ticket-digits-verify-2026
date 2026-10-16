"""Data: 1797 pictures of handwritten digits, 8x8, values 0..16.

Public source: sklearn.datasets.load_digits (UCI ML digits, no auth).
Matches the video: 1797 pictures, 8x8, 0=white paper, 16=full ink -> scaled to [0,1].

Split (fixed, stratified, seed=0):
  train 1000 / val 297 / test 500  (1000+297+500=1797)
The video trains on 1000, tests on 500 never-seen. The leftover 297 is the
"pictures that are set aside" used for learning-speed curves.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load_all():
    d = load_digits()
    X = d.data.astype(np.float64) / 16.0  # 0=white .. 1=full ink
    y = d.target.astype(np.int64)
    assert X.shape == (1797, 64), X.shape
    assert X.min() >= 0.0 and X.max() <= 1.0
    return X, y


def fixed_split(seed=0):
    X, y = load_all()
    # first hold out test 500, stratified
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=500, random_state=seed, stratify=y)
    # then train 1000 from remainder (1297 left), rest -> val 297
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=297, train_size=1000, random_state=seed, stratify=y_tmp)
    assert len(X_train) == 1000 and len(X_val) == 297 and len(X_test) == 500
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
