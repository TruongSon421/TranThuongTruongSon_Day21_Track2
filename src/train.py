import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.base import ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

EVAL_THRESHOLD = 0.70


def _configure_mlflow_from_env() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def _build_model(params: dict) -> tuple[ClassifierMixin, dict]:
    model_type = params.get("model_type", "random_forest")

    if model_type == "random_forest":
        model_params = {
            "n_estimators": params.get("n_estimators", 100),
            "max_depth": params.get("max_depth", 5),
            "min_samples_split": params.get("min_samples_split", 2),
            "random_state": 42,
        }
        return RandomForestClassifier(**model_params), {
            "model_type": model_type,
            **model_params,
        }

    if model_type == "gradient_boosting":
        model_params = {
            "n_estimators": params.get("n_estimators", 100),
            "learning_rate": params.get("learning_rate", 0.1),
            "max_depth": params.get("max_depth", 3),
            "random_state": 42,
        }
        return GradientBoostingClassifier(**model_params), {
            "model_type": model_type,
            **model_params,
        }

    if model_type == "extra_trees":
        model_params = {
            "n_estimators": params.get("n_estimators", 500),
            "max_features": params.get("max_features", 1.0),
            "criterion": params.get("criterion", "gini"),
            "max_depth": params.get("max_depth"),
            "max_leaf_nodes": params.get("max_leaf_nodes"),
            "min_samples_split": params.get("min_samples_split", 2),
            "min_samples_leaf": params.get("min_samples_leaf", 1),
            "min_impurity_decrease": params.get("min_impurity_decrease", 0.0),
            "ccp_alpha": params.get("ccp_alpha", 0.0),
            "bootstrap": params.get("bootstrap", False),
            "max_samples": params.get("max_samples"),
            "oob_score": params.get("oob_score", False),
            "class_weight": params.get("class_weight", "balanced"),
            "n_jobs": params.get("n_jobs", -1),
            "random_state": params.get("random_state", 42),
        }
        model_params_clean = {k: v for k, v in model_params.items() if v is not None}
        return ExtraTreesClassifier(**model_params_clean), {
            "model_type": model_type,
            **model_params_clean,
        }

    if model_type == "logistic_regression":
        model_params = {
            "C": params.get("C", 1.0),
            "max_iter": params.get("max_iter", 1000),
            "multi_class": "auto",
            "random_state": 42,
        }
        return LogisticRegression(**model_params), {
            "model_type": model_type,
            **model_params,
        }

    raise ValueError(
        f"Unsupported model_type='{model_type}'. "
        "Use one of: random_forest, gradient_boosting, logistic_regression, extra_trees."
    )


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua model_type va cac sieu tham so tuong ung.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """

    _configure_mlflow_from_env()
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    with mlflow.start_run():

        model, run_params = _build_model(params)
        mlflow.log_params(run_params)
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")
        precision, recall, _, _ = precision_recall_fscore_support(
            y_eval, preds, labels=[0, 1, 2], zero_division=0
        )
        cm = confusion_matrix(y_eval, preds, labels=[0, 1, 2])
        label_dist = y_train.value_counts(normalize=True).reindex([0, 1, 2], fill_value=0.0)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")
        for label, ratio in label_dist.items():
            if ratio < 0.10:
                print(
                    f"WARNING: label {label} chiem {ratio:.2%} (< 10%) trong tap huan luyen."
                )

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.txt", "w") as f:
            f.write("Confusion Matrix (labels: 0, 1, 2)\n")
            for row in cm.tolist():
                f.write(" ".join(str(item) for item in row) + "\n")
            f.write("\nPrecision/Recall by class\n")
            for idx, label in enumerate([0, 1, 2]):
                f.write(
                    f"class_{label}: precision={precision[idx]:.4f}, "
                    f"recall={recall[idx]:.4f}\n"
                )

        with open("outputs/metrics.json", "w") as f:
            json.dump(
                {
                    "accuracy": acc,
                    "f1_score": f1,
                    "label_distribution": {
                        "0": float(label_dist.loc[0]),
                        "1": float(label_dist.loc[1]),
                        "2": float(label_dist.loc[2]),
                    },
                },
                f,
            )

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
