"""
train.py — Trains a Random Forest classifier on the Iris dataset,
saves the model as model.joblib, and logs everything to MLflow.
"""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib
import mlflow
import mlflow.sklearn

# ── Hyperparameters ──────────────────────────────────────────────
N_ESTIMATORS = 100
MAX_DEPTH     = None   # unlimited
RANDOM_STATE  = 42
TEST_SIZE     = 0.2
MODEL_NAME    = "IrisRandomForest"


def train():
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average="weighted")
    print(f"Accuracy: {acc:.4f}  |  F1: {f1:.4f}")

    # Save model locally
    joblib.dump(model, "model.joblib")
    print("Model saved to model.joblib")

    # ── MLflow tracking ──────────────────────────────────────────
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Iris-Classification")

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("max_depth",    MAX_DEPTH)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("test_size",    TEST_SIZE)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Log model artifact + register in Model Registry
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        # Also log the joblib file as extra artifact
        mlflow.log_artifact("model.joblib")

        print(f"MLflow run logged. Model registered as '{MODEL_NAME}'.")


if __name__ == "__main__":
    train()
