import os
import sys
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
from src.utils.logger import get_logger
from src.utils.config import MODEL_PATH

logger = get_logger("ModelTrainer")


class ModelTrainer:
    def __init__(self):
        self.model_path = MODEL_PATH

    def initiate_model_trainer(self, train_arr_path, test_arr_path):
        try:
            logger.info("Model training is starting...")

            # 1. Load data
            train_arr = np.load(train_arr_path)
            test_arr = np.load(test_arr_path)

            logger.info("Numpy arrays have been loaded.")

            # 2. X and y Split
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # 3. Model (XGBoost)
            logger.info("XGBoost model is being launched...")
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=7,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=42
            )

            model.fit(X_train, y_train)
            logger.info("Model fitting process completed.")

            # 4. Results (Outlier Filtering for Evaluation)
            y_prediction = model.predict(X_test)


            filter_mask = (y_test > 500) & (y_test < 100000)

            y_test_clean = y_test[filter_mask]
            y_pred_clean = y_prediction[filter_mask]

            r2 = r2_score(y_test_clean, y_pred_clean)
            rmse = np.sqrt(mean_squared_error(y_test_clean, y_pred_clean))

            logger.info(f"Model Performance (Filtered Test Set) -> R2: {r2:.4f} | RootMSE: {rmse:.2f}")

            # 5. Saving the Model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(model, self.model_path)
            logger.info(f"The model was successfully saved: {self.model_path}")

            return self.model_path

        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise e


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    try:
        logger.info("--- MODEL TRAINING TEST HAS STARTED ---")

        # 1. Load the Data
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        # 2. Transform the Data
        transformer = DataTransformation()
        train_arr, test_arr, _ = transformer.initiate_data_transformation(train_path, test_path)

        # 3. Train the Model
        trainer = ModelTrainer()
        trainer.initiate_model_trainer(train_arr, test_arr)

        print("\n--- TRAINER TEST SUCCESSFUL ---")

    except Exception as e:
        print("\n--- TEST FAILED ---")
        print(e)