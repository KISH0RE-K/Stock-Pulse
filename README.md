# StockPulse 📈

### End-to-End Stock Market ETL Pipeline & ML Prediction Dashboard

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge\&logo=python)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge\&logo=docker)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge\&logo=streamlit)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge\&logo=sqlite)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge\&logo=scikit-learn)

An end-to-end data engineering and machine learning project that extracts real stock market data, transforms it through an ETL pipeline, stores it in a SQL database, visualizes key business metrics, and predicts next-day stock price movement using machine learning.

---

## 🏗️ Architecture

yfinance API
↓
Extract
↓
Transform (Feature Engineering)
↓
SQLite Data Warehouse
↓
SQL Analytics
↓
Plotly Visualizations
↓
Random Forest Model
↓
Streamlit Dashboard

---

## 🐳 Run with Docker

### Build Image

```bash
docker build -t stockpulse .
```

### Run Container

```bash
docker run -p 8501:8501 stockpulse
```

### Open Dashboard

```text
http://localhost:8501
```

No local Python installation required.

---

## ⚡ Quick Start (Without Docker)

```bash
git clone https://github.com/KISH0RE-K/Stock-Pulse.git

cd Stock-Pulse

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

python -m streamlit run app.py
```

---

## 📊 ETL Pipeline Metrics

| Metric                 | Value         |
| ---------------------- | ------------- |
| Trading Days Processed | 252           |
| Raw Features Extracted | 7             |
| Engineered Features    | 12            |
| Database Engine        | SQLite        |
| ETL Framework          | Pandas        |
| Visualization Library  | Plotly        |
| ML Algorithm           | Random Forest |

---

## 🎯 Key Data Engineering Concepts Demonstrated

* ETL Pipeline Design
* Data Cleaning & Validation
* Feature Engineering
* SQL Database Integration
* Interactive Analytics Dashboards
* Time-Series Machine Learning
* Docker Containerization
* Reproducible Deployment

---

## 📈 Model Performance

| Metric              | Result                       |
| ------------------- | ---------------------------- |
| Accuracy            | 61.70%                       |
| Train/Test Strategy | Sequential Time-Series Split |
| Algorithm           | Random Forest Classifier     |
| Trees               | 150                          |
| Max Depth           | 4                            |

---

## 🚀 Future Improvements

* PostgreSQL Migration
* PySpark ETL Pipeline
* Airflow Orchestration
* Multi-Stock Support
* Real-Time Streaming Data
* Cloud Deployment
* Model Monitoring Dashboard

---

Built to demonstrate practical Data Engineering, ETL, Analytics, and Machine Learning workflows in a production-style environment.
