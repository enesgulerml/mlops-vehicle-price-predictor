# 🏎️ MLOps Vehicle Price Predictor

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![MLOps](https://img.shields.io/badge/Architecture-Modular%20Pipeline-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)

## 📖 Project Overview
This project is a production-grade machine learning application designed to predict used vehicle prices based on various features (brand, year, condition, technical specs). 

Unlike traditional data science notebooks, this project focuses on **MLOps best practices**, featuring a modular architecture, reproducible pipelines, and a clear separation of concerns (Ingestion, Transformation, Training).

## 🏗️ Architecture
The project follows a component-based modular structure:

```text
mlops-vehicle-price-predictor/
├── data/                # Data storage (gitignored)
│   ├── raw/             # Raw CSV files
│   └── processed/       # Train/Test splits & artifacts
├── models/              # Serialized models (.pkl) & Encoders
├── src/                 # Source Code
│   ├── components/      # Core Logic Units
│   │   ├── data_ingestion.py      # Data splitting & schema handling
│   │   ├── data_transformation.py # Feature Eng., Cleaning & Encoding
│   │   └── model_trainer.py       # XGBoost training & Evaluation
│   ├── pipelines/       # Orchestration (Training & Prediction)
│   │   └── prediction_pipeline.py # Inference Logic
│   └── utils/           # Helper modules (Logging, Config)
└── requirements.txt     # Dependencies
```

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Machine Learning:** XGBoost, Scikit-learn, Pandas, NumPy
* **MLOps:** Modular Pipeline Design, Artifact Management, Logging
* **Future Roadmap:** Docker, MLflow, FastAPI, CI/CD, AWS Deployment

## 🚀 Getting Started
### 1. Prerequisites
* Python 3.8+
* Virtual Environment (Recommended)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/enesgulerml/mlops-vehicle-price-predictor.git

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Usage
The pipeline is designed to be triggered component by component (currently):

#### Step 1: Data Ingestion Reads the raw CSV, handles schema, and creates train.csv / test.csv.
```bash
python -m src.components.data_ingestion
```

#### Step 2: Data Transformation Cleans data, handles outliers, performs feature engineering, saves LabelEncoders, and produces .npy arrays.
```bash
python -m src.components.data_transformation
```

#### Step 3: Model Training trains the XGBoost model on processed data.
```bash
python -m src.components.model_trainer
```

### 4. Running the API
Start the FastAPI server:

```bash
uvicorn src.api.app:app --reload
```
Then, open your browser and go to Swagger UI to test predictions interactively: 👉 http://127.0.0.1:8000/docs


## 📈 Model Performance
* **Current Model:** XGBoost Regressor
* **Metrics:**
  * **R2 Score:** ~0.82 (on filtered test set)
  * **RMSE:** ~6100 (on filtered test set)

## 👤 Author
Enes Guler - MLOps Engineer