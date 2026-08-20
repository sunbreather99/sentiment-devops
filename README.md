# 🚀 DevOps + AI: Dual-Model Sentiment Analysis System

## 📌 Project Overview

This project implements a **Twitter Sentiment Analysis system** using both **Machine Learning** and **Deep Learning (Transformer-based)** approaches. It integrates **DevOps practices** for deployment and scalability.

The system compares:

* **Logistic Regression (Baseline Model)**
* **DistilBERT (Advanced Transformer Model)**

The application is deployed as a **FastAPI service** and hosted on the cloud using **Render**.

---

## 🎯 Objectives

* Perform sentiment analysis on Twitter data
* Compare traditional ML vs modern DL models
* Build a scalable API using FastAPI
* Apply DevOps practices (CI/CD, deployment)

---

## 📊 Dataset

* **Source:** Kaggle (Twitter Sentiment Dataset)
* **File:** `Sentiment.csv`
* **Columns:**

  * `text` → Tweet content
  * `sentiment` → Positive / Negative / Neutral

---

## 🧠 Models Used

### 🔹 Logistic Regression (Baseline)

* TF-IDF Vectorization (`max_features=5000`)
* Fast and efficient
* Limitation: Cannot understand context

### 🔹 DistilBERT (Advanced)

* Pretrained transformer model
* Captures context, sarcasm, and semantics
* Returns prediction + confidence score

---

## ⚙️ Tech Stack

* **Language:** Python
* **ML Libraries:** Scikit-learn, Transformers, Torch
* **API Framework:** FastAPI
* **Deployment:** Render
* **DevOps Tools:** GitHub, Docker (optional)

---

## 🏗️ Project Structure

```
project/
│── app.py              # FastAPI application
│── train.py            # Model training script
│── model.pkl           # Logistic Regression model
│── vectorizer.pkl      # TF-IDF vectorizer
│── Sentiment.csv       # Dataset
│── requirements.txt    # Dependencies
│── Dockerfile          # Containerization
│── .github/            # CI/CD workflows
```

---

## 🚀 How to Run Locally

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run API

```bash
uvicorn app:app --reload
```

### 3️⃣ Open in Browser

```
http://127.0.0.1:8000/docs
```

---

## ☁️ Deployment

The application is deployed using **Render**.

### 🔹 Start Command:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

### 🔹 Notes:

* Uses dynamic port binding
* Free tier may cause cold start delays

---

## 🧪 API Usage

### POST `/predict`

#### Request:

```json
{
  "text": "I love this project"
}
```

#### Response:

```json
{
  "input": "I love this project",
  "logistic_regression": "positive",
  "bert_prediction": "POSITIVE",
  "bert_confidence": 0.98
}
```

---

## ⚠️ Challenges Faced

* Flask → FastAPI migration
* Multiprocessing issues with Transformers
* Model serialization version mismatch
* Docker environment setup issues
* Slow cold start on Render free tier

---

## ✅ Solutions

* Used `@app.on_event("startup")` for model loading
* Switched to FastAPI for better performance
* Used dynamic port binding (`$PORT`)
* Optimized deployment for cloud environment

---

## 📈 Results

| Model               | Strength      | Limitation               |
| ------------------- | ------------- | ------------------------ |
| Logistic Regression | Fast          | No context understanding |
| DistilBERT          | Context-aware | Higher latency           |

---

## 🎤 Viva Key Points

* Difference between ML vs DL models
* Why DistilBERT performs better
* Role of DevOps in deployment
* CI/CD pipeline importance
* FastAPI advantages over Flask

---

## 🏆 Conclusion

This project demonstrates how **AI models can be integrated with DevOps practices** to build scalable and production-ready applications. It highlights the performance gap between traditional ML and transformer-based models.

---

## 📌 Future Improvements

* Add real-time streaming data
* Improve model accuracy with fine-tuning
* Use GPU-based deployment
* Implement monitoring & logging

---

## 👨‍💻 Author

Saksham Chopra

---

## ⭐ Acknowledgment

* Kaggle for dataset
* HuggingFace for transformer models
* Render for deployment platform
