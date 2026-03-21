from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI(title="Student MLOps API")

def load_model():
    with open("models/model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

@app.get("/")
def home():
    return {"message": "MLOps API running 🚀"}

@app.get("/predict")
def predict(hours: float, marks: float):
    data = np.array([[hours, marks]])
    pred = model.predict(data)[0]

    return {
        "input": {"hours": hours, "marks": marks},
        "prediction": int(pred)
    }