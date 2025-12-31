# 🚗 End-to-End Used Car Price Prediction (MLOps)

## 📖 Project Overview
This project is a production-grade machine learning application designed to predict used vehicle prices based on various features (brand, year, condition, technical specs). 

Unlike traditional data science notebooks, this project focuses on **MLOps best practices**, featuring a modular architecture, reproducible pipelines, and a clear separation of concerns (Ingestion, Transformation, Training).

## 🏗️ Architecture
The project follows a component-based modular structure:

```text
src/
├── components/          # Core Logic Units
│   ├── data_ingestion.py      # Splits raw data into Train/Test artifacts
│   ├── data_transformation.py # Cleaning, Feature Engineering & Encoding (saves .pkl)
│   └── model_trainer.py       # Model training (XGBoost/RF) & serialization
├── pipelines/           # Orchestrators (Training & Prediction Pipelines)
├── utils/               # Helpers (Logging, Config management)
└── ...
```

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Machine Learning:** XGBoost, Scikit-learn, Pandas, NumPy
* **MLOps:** Modular Pipeline Design, Artifact Management, Logging
* **Future Roadmap:** Docker, FastAPI, CI/CD, AWS Deployment

## 🚀 Getting Started
### 1. Prerequisites
* Python 3.8+
* Virtual Environment (Recommended)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/enesgulerml/regression-project.git

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

#### Step 3: Model Training (Coming Soon) Trains the XGBoost model on processed data.

## 📈 Model Performance
* **Current Model:** XGBoost Regressor
* **Metrics:** Tracking RMSE and R2 Score (Details to be updated after full training).

## 👤 Author
Enes Guler - MLOps Engineer