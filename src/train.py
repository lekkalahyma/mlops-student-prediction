import os
import pickle
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from utils import load_data, get_features_target


mlflow.set_experiment("student-mlops-project")
def train():
    df = load_data()
    X, y = get_features_target(df)

    with mlflow.start_run():
        model = LogisticRegression()
        model.fit(X, y)

        acc = model.score(X, y)

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

    os.makedirs("models", exist_ok=True)

    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"✅ Model trained | Accuracy: {acc}")

if __name__ == "__main__":
    train()