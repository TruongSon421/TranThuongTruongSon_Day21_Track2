import importlib
import sys
from unittest.mock import Mock

import pytest
from fastapi import HTTPException


def _load_serve_module(monkeypatch):
    mock_model = Mock()
    mock_model.predict.return_value = [1]

    mock_s3_client = Mock()
    mock_boto3 = Mock()
    mock_boto3.client.return_value = mock_s3_client

    monkeypatch.setenv("AWS_REGION", "ap-southeast-1")
    monkeypatch.setenv("S3_BUCKET", "unit-test-bucket")
    monkeypatch.setenv("S3_MODEL_KEY", "models/latest/model.pkl")
    monkeypatch.setitem(sys.modules, "boto3", mock_boto3)

    import joblib

    monkeypatch.setattr(joblib, "load", Mock(return_value=mock_model))
    sys.modules.pop("src.serve", None)
    module = importlib.import_module("src.serve")
    return module, mock_boto3, mock_s3_client


def test_download_model_uses_s3_env(monkeypatch):
    module, mock_boto3, mock_s3_client = _load_serve_module(monkeypatch)

    mock_boto3.client.assert_called_once_with("s3", region_name="ap-southeast-1")
    mock_s3_client.download_file.assert_called_once_with(
        "unit-test-bucket",
        "models/latest/model.pkl",
        module.MODEL_PATH,
    )


def test_predict_requires_12_features(monkeypatch):
    module, _, _ = _load_serve_module(monkeypatch)

    with pytest.raises(HTTPException):
        module.predict(module.PredictRequest(features=[0.1] * 11))
