import sys
import os
import pandas as pd
import joblib
import datetime
import numpy as np
from src.utils.logger import get_logger
from src.utils.config import MODEL_PATH, ENCODER_PATH, MODEL_FEATURES

logger = get_logger("PredictionPipeline")

class PredictionPipeline:
    def __init__(self):
        self.model_path = MODEL_PATH
        self.encoder_path = ENCODER_PATH
        self.model = None
        self.encoders = None

    def load_artifacts(self):
        """It loads the model and encoder files only when needed."""
        try:
            if self.model is None:
                self.model = joblib.load(self.model_path)

            if self.encoders is None:
                self.encoders = joblib.load(self.encoder_path)

        except Exception as e:
            logger.error(f"Artifact loading error: {e}")
            raise e

    def preprocess_input(self, input_data: dict) -> pd.DataFrame:
        """It converts the raw data (dict) received from the user into a format (Dataframe) which the model can understand."""
        try:
            # 1. Dict -> Dataframe
            df = pd.DataFrame([input_data])

            # 2. Feature Engineering
            current_year = datetime.datetime.now().year
            if 'year' in df.columns:
                df['car_age'] = current_year - int(df['year'].iloc[0])
            else:
                df['car_age'] = 0 # Default

            # 3. Model Name Handling
            if 'model' in df.columns:
                df.rename(columns={'model':'model_new'}, inplace=True)

            # 4. Encoding
            self.load_artifacts()

            cat_cols = ['manufacturer', 'model_new', 'condition', 'cylinders', 'fuel',
                        'title_status', 'transmission', 'drive', 'size', 'type',
                        'paint_color', 'state']

            for col in cat_cols:
                val = str(df[col].iloc[0])

                if col in self.encoders:
                    le = self.encoders[col]

                    if val in le.classes_:
                        df[col] = le.transform([val])[0]
                    else:
                        logger.warning(f"Unknown value detected: {col} -> {val}. Assigning default.")
                        df[col] = le.transform([le.classes_[0]])[0]

            # 5. Reordering
            for col in MODEL_FEATURES:
                if col not in df.columns:
                    df[col] = 0 # Default

            df = df[MODEL_FEATURES]

            return df

        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            raise e

    def predict(self, input_data: dict):
        try:
            self.load_artifacts()

            # Prepare the data
            df_processed = self.preprocess_input(input_data)

            # Predict
            prediction = self.model.predict(df_processed)

            return float(prediction[0])

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise e

if __name__ == "__main__":
    sample_data = {
        'year': 2017,
        'manufacturer': 'ford',
        'model': 'f-150',
        'condition': 'excellent',
        'cylinders': '8 cylinders',
        'fuel': 'gas',
        'odometer': 48000,
        'title_status': 'clean',
        'transmission': 'automatic',
        'drive': '4wd',
        'size': 'full-size',
        'type': 'pickup',
        'paint_color': 'black',
        'state': 'tx'
    }

    try:
        pipeline = PredictionPipeline()
        price = pipeline.predict(sample_data)

        print("\n" + "=" * 30)
        print(f"Estimated Vehicle Price: ${price:,.2f}")
        print("=" * 30 + "\n")

    except Exception as e:
        print(e)
