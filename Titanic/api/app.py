from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Prédiction survie Titanic")
modele = joblib.load("modele_titanic.joblib")


class Passager(BaseModel):
    Pclass: int
    Sex: int  # 0 = male, 1 = female
    Age: float
    Fare: float
    Embarked: float  # 0 = C, 1 = Q, 2 = S


@app.post("/predict")
def predict(p: Passager):
    X = pd.DataFrame([p.model_dump()])[["Pclass", "Sex", "Age", "Fare", "Embarked"]]
    pred = int(modele.predict(X)[0])
    proba = modele.predict_proba(X)[0]
    return {
        "survit": bool(pred),
        "probabilite_survie": round(float(proba[1]), 4),
        "probabilite_deces": round(float(proba[0]), 4),
    }
