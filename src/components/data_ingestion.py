import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger
from src.utils.config import RAW_DATA_DIR, DATA_DIR

# Calling Logger
logger = get_logger("DataIngestion")

class DataIngestion:
    def __init__(self):
        self.raw_data_path = RAW_DATA_DIR
        # Where the processed (split) data will reside.
        self.train_data_path = os.path.join(DATA_DIR, "processed" ,"train.csv")
        self.test_data_path = os.path.join(DATA_DIR, "processed" ,"test.csv")

    def initiate_data_ingestion(self):
        logger.info("The data ingestion process has begun.")
        try:
            # 1. Read the data.
            df = pd.read_csv(self.raw_data_path)
            logger.info(f"Raw data was read. Size: {df.shape}")

            # 2. Create the Processed folder if it doesn't exist.
            os.makedirs(os.path.dirname(self.train_data_path), exist_ok=True)

            # 3. Train-Test Split
            logger.info("Train-Test split procedure is being performed...")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # 4. Save the files
            train_set.to_csv(self.train_data_path, index=False, header=True)
            test_set.to_csv(self.test_data_path, index=False, header=True)

            logger.info(f"Data saved:\nTrain: {self.train_data_path}\nTest: {self.test_data_path}")
            logger.info("Data ingestion process has been completed successfully.")

            return (
                self.train_data_path,
                self.test_data_path,
            )

        except Exception as e:
            logger.error(f"Error during Data Ingestion: {e}")
            raise e

if __name__ == "__main__":
    try:
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()
        print(f"\n--- TEST SUCCESSFUL ---")
        print(f"Train path: {train_data}")
        print(f"Test path: {test_data}")
    except Exception as e:
        print(f"\n--- TEST FAILED ---")
        print(e)