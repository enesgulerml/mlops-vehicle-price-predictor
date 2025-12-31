import sys
import os
import pandas as pd
import numpy as np
import joblib
import datetime
from sklearn.preprocessing import LabelEncoder
from src.utils.logger import get_logger
from src.utils.config import ENCODER_PATH, DATA_DIR

logger = get_logger("DataTransformation")

class DataTransformation:
    def __init__(self):
        self.encoder_path = ENCODER_PATH
        self.train_arr_path = os.path.join(DATA_DIR, "processed", "train_arr.npy")
        self.test_arr_path = os.path.join(DATA_DIR, "processed", "test_arr.npy")

        self.cat_cols = ['manufacturer', 'model_new', 'condition', 'cylinders', 'fuel',
                         'title_status', 'transmission', 'drive', 'size', 'type',
                         'paint_color', 'state']

    def _clean_and_engineer(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """
        It performs data cleaning and feature engineering operations.
        """
        df = df.copy()


        # Standardization:
        df.columns = df.columns.str.lower().str.strip()

        # Dropping unnecessary columns:
        drop_cols = ['url', 'region_url', 'image_url', 'lat', 'long',
                     'vin', 'id', 'county', 'description', 'region', 'posting_date']
        df = df.drop(columns=drop_cols, errors='ignore')

        # Outlier Filter:
        if is_train:
            if 'price' not in df.columns:
                raise KeyError(f"The 'price' column was not found! Available columns: {df.columns.tolist()}")

            df = df[(df['price'] > 1000) & (df['price'] < 60000)]

            # Check the 'year' column.
            if 'year' in df.columns:
                df = df[df['year'] > 2005]

        # Feature Engineering:
        if 'year' in df.columns:
            current_year = datetime.datetime.now().year
            df['car_age'] = current_year - df['year']
            df = df.drop(columns=['year'])

        if 'model' in df.columns:
            df.rename(columns={'model': 'model_new'}, inplace=True)

        # Missing Data Filling:
        # Categorical
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna('unknown')
        # Numerical
        for col in df.select_dtypes(include=['number']).columns:
            df[col] = df[col].fillna(df[col].mean())

        return df

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logger.info("Data transformation has begun...")

            # Read the data:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info("Train and Test data has been read...")

            # Feature Engineering and Cleaning:
            train_df = self._clean_and_engineer(train_df, is_train=True)
            test_df = self._clean_and_engineer(test_df, is_train=False)

            logger.info(f"Train Size After Cleaning: {train_df.shape}")

            # Encoding:
            encoders = {}

            # Label Encoding:
            for col in self.cat_cols:
                if col in train_df.columns:
                    le = LabelEncoder()
                    train_df[col] = train_df[col].astype(str)
                    test_df[col] = test_df[col].astype(str)

                    # Fit and Transform (Train)
                    le.fit(train_df[col])
                    train_df[col] = le.transform(train_df[col])

                    le_dict = dict(zip(le.classes_, le.transform(le.classes_)))

                    test_df[col] = test_df[col].apply(lambda x: le_dict.get(x, -1))

                    encoders[col] = le

            logger.info("Encoding process has completed.")

            # Save the Encoders (with joblib)
            os.makedirs(os.path.dirname(self.encoder_path), exist_ok=True)
            joblib.dump(encoders, self.encoder_path)
            logger.info(f"Encoder has been saved to: {self.encoder_path}")

            # Separating the Objective Variable and Converting it to a Numpy Array
            target_col = "price"

            # Train
            X_train = train_df.drop(columns=[target_col], axis=1)
            y_train = train_df[target_col]

            # Test
            X_test = test_df.drop(columns=[target_col], axis=1)
            y_test = test_df[target_col]

            # Combine as Numpy Array (X and y side-by-side)
            train_arr = np.c_[X_train.values, y_train.values]
            test_arr = np.c_[X_test.values, y_test.values]

            # Save Arrays:
            np.save(self.train_arr_path, train_arr)
            np.save(self.test_arr_path, test_arr)

            logger.info("Train and Test data has been saved as .npy.")
            return (
                self.train_arr_path,
                self.test_arr_path,
                self.encoder_path
            )

        except Exception as e:
            logger.error(f"Transformation Error: {e}")
            raise e

if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion

    try:
        # First Ingestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        # Then Transformation
        transformer = DataTransformation()
        train_arr, test_arr, _ = transformer.initiate_data_transformation(train_path, test_path)

        print("\n--- TRANSFORMATION TEST SUCCESSFUL ---")
        print(f"Train Array Shape: {np.load(train_arr).shape}")
        print(f"Test Array Shape: {np.load(test_arr).shape}")

    except Exception as e:
        print("\n--- TEST FAILED ---")
        print(e)