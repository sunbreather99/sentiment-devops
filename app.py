# Logistic Regression model with TF-IDF vectorization for sentiment analysis, served via FastAPI.
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load Logistic Regression model
with open("model.pkl", "rb") as f:
    lr_model = pickle.load(f)
    
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Placeholder for BERT

classifier = None

class InputText(BaseModel):
    text: str

# Load BERT on startup using lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    from transformers import pipeline
    classifier = pipeline("sentiment-analysis")
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
def home():
    return {"message": "Dual Model API running"}

@app.post("/predict")
def predict(data: InputText):
    text = data.text

    # Logistic Regression prediction
    vec = vectorizer.transform([text])
    lr_pred = lr_model.predict(vec)[0]

    # BERT prediction
    bert_result = classifier(text)[0]

    return {
        "input": text,
        "logistic_regression": lr_pred,
        "bert_prediction": bert_result["label"],
        "bert_confidence": float(bert_result["score"])
    }

import uvicorn
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)