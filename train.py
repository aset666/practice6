from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
def train():
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model trained. Accuracy on test set: {acc:.4f}")
    # Save model
    joblib.dump(model, "model.joblib")
    print("Model saved to model.joblib")
if __name__ == "__main__":
    train()
