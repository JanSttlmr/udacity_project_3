import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, fbeta_score
import sys
import os
#sys.path.append(os.path.abspath(os.path.dirname("ml")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname("model.py"), 'ml')))
from model import train_model, compute_model_metrics, inference

# Mock dataset
X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y_train = np.array([0, 1, 0, 1])
X_test = np.array([[2, 3], [6, 7]])
y_test = np.array([0, 1])

def test_train_model():
    """Test if the train_model function returns a fitted model."""
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "predict")

def test_compute_model_metrics():
    """Test compute_model_metrics returns valid precision, recall, and f1-score."""
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])
    
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)

    # Validate against sklearn metrics
    assert precision == precision_score(y_true, y_pred, zero_division=1)
    assert recall == recall_score(y_true, y_pred, zero_division=1)
    assert fbeta == fbeta_score(y_true, y_pred, beta=1, zero_division=1)

def test_inference():
    """Test if inference function returns expected predictions."""
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (X_test.shape[0],)