import os
import json
import numpy as np
import pandas as pd
import pytest
from src.train import train


FEATURE_NAMES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type",
]


def _make_temp_data(tmp_path):
    """
    Tao dataset nho voi cung schema Wine Quality de su dung trong test.

    pytest cung cap `tmp_path` la mot thu muc tam thoi, tu dong xoa sau khi test ket thuc.
    Ham nay dung du lieu ngau nhien nen khong can ket noi GCS hay tai file CSV thuc.
    """
    rng = np.random.default_rng(0)
    n = 200

    X = rng.random((n, len(FEATURE_NAMES)))
    y = rng.integers(0, 3, size=n)

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y

    train_path = str(tmp_path / "train.csv")
    eval_path = str(tmp_path / "eval.csv")
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path, index=False)

    return train_path, eval_path


def test_train_returns_float(tmp_path):
    """Kiem tra ham train() tra ve mot so thuc nam trong [0.0, 1.0]."""
    train_path, eval_path = _make_temp_data(tmp_path)

    acc = train(
        {"n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0


def test_metrics_file_created(tmp_path):
    """Kiem tra file outputs/metrics.json duoc tao sau khi huan luyen."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("outputs/metrics.json")
    with open("outputs/metrics.json") as f:
        metrics = json.load(f)
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert "label_distribution" in metrics
    assert set(metrics["label_distribution"].keys()) == {"0", "1", "2"}


def test_model_file_created(tmp_path):
    """Kiem tra file models/model.pkl duoc tao sau khi huan luyen."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("models/model.pkl")


def test_report_file_created(tmp_path):
    """Kiem tra file outputs/report.txt duoc tao sau khi huan luyen."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"model_type": "gradient_boosting", "n_estimators": 20, "max_depth": 2},
        data_path=train_path,
        eval_path=eval_path,
    )

    assert os.path.exists("outputs/report.txt")


def test_support_multiple_model_types(tmp_path):
    """Kiem tra train() chay duoc voi nhieu model_type bao gom extra_trees."""
    train_path, eval_path = _make_temp_data(tmp_path)

    for params in [
        {"model_type": "random_forest", "n_estimators": 10, "max_depth": 3},
        {"model_type": "gradient_boosting", "n_estimators": 20, "max_depth": 2},
        {"model_type": "logistic_regression", "C": 1.0, "max_iter": 200},
        {"model_type": "extra_trees", "n_estimators": 10, "max_depth": 3},
    ]:
        acc = train(params, data_path=train_path, eval_path=eval_path)
        assert 0.0 <= acc <= 1.0


def test_extra_trees_optional_params(tmp_path):
    """Kiem tra extra_trees chay duoc khi khong truyen max_depth va max_leaf_nodes (None)."""
    train_path, eval_path = _make_temp_data(tmp_path)

    # max_depth=None va max_leaf_nodes=None la gia tri mac dinh cua ExtraTreesClassifier
    # train() phai loc cac gia tri None truoc khi truyen vao sklearn
    acc = train(
        {"model_type": "extra_trees", "n_estimators": 10},
        data_path=train_path,
        eval_path=eval_path,
    )
    assert 0.0 <= acc <= 1.0


def test_metrics_json_values_in_range(tmp_path):
    """Kiem tra accuracy va f1_score trong metrics.json nam trong [0.0, 1.0]."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )

    with open("outputs/metrics.json") as f:
        metrics = json.load(f)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    # label_distribution phai cong lai bang 1.0 (cho phep sai so float)
    total = sum(metrics["label_distribution"].values())
    assert abs(total - 1.0) < 1e-6


def test_report_contains_expected_sections(tmp_path):
    """Kiem tra outputs/report.txt chua ca phan Confusion Matrix va Precision/Recall."""
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )

    with open("outputs/report.txt") as f:
        content = f.read()
    assert "Confusion Matrix" in content
    assert "Precision/Recall by class" in content


def test_unsupported_model_type_raises_value_error(tmp_path):
    """Kiem tra train() bao loi ro rang khi model_type khong duoc ho tro."""
    train_path, eval_path = _make_temp_data(tmp_path)

    with pytest.raises(ValueError, match="Unsupported model_type"):
        train(
            {"model_type": "svm"},
            data_path=train_path,
            eval_path=eval_path,
        )
