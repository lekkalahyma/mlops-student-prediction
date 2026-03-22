# Clone repo
git clone https://github.com/lekkalahyma/mlops-student-prediction.git
cd mlops-student-prediction

# Install dependencies
pip install -r requirements.txt

# Train model
python src/train.py

# Start MLflow UI
mlflow ui

# Run API
uvicorn app.main:app --reload



# Open in browser
http://127.0.0.1:8000/docs


# Direct API test
http://127.0.0.1:8000/predict?hours=5&marks=60


# Usage Endpoint:
/predict
Example:
http://127.0.0.1:8000/predict?hours=5&marks=60
Response:
{
  "prediction": 1
}


# Experiment Tracking

Using MLflow:

mlflow ui

Open:

http://127.0.0.1:5000
