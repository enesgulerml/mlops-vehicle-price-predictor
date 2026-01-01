import os

PROJECT_ROOT= os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data Path
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw', 'vehicles.csv')

# Model and Artifact Path
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'car_price_model.pkl')
ENCODER_PATH = os.path.join(MODELS_DIR, 'encoder.pkl')

# Model Parameters
TARGET_COLUMN = "price"
DROP_COLS = ['url', 'region_url', 'image_url', 'lat', 'long',
             'vin', 'id', 'county', 'description', 'region', 'posting_date']

MODEL_FEATURES = [
    'manufacturer', 'model_new', 'condition', 'cylinders', 'fuel',
    'odometer', 'title_status', 'transmission', 'drive', 'size',
    'type', 'paint_color', 'state', 'car_age'
]