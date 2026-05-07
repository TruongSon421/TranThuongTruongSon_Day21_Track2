import mlflow
import pytest


@pytest.fixture(autouse=True)
def isolate_mlflow(tmp_path):
    """Redirect MLflow tracking to a temp dir for each test.

    Prevents tests from hitting DagsHub or reading a corrupt local mlruns/
    directory. Each test gets a clean, isolated MLflow file store.
    """
    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
    yield
    mlflow.set_tracking_uri(None)
