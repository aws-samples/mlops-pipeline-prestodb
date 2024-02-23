## import the necessary variables
import os
import json
import joblib
import logging
import datetime
import pandas as pd
from io import StringIO

## define the logger
logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def model_fn(model_dir):
    """
    Load the model and its config from the model directory.
    """
    logger.info("Loading model.")
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    
    logger.info("Loading model configuration.")
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # # Attach config to the model object
    model.features = config.get('FEATURES')
    
    return model

def input_fn(input_data, content_type):
    """
    Parse input data.
    """
    if content_type == 'text/csv':
        df = pd.read_csv(StringIO(input_data))
        return df
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """
    Predict function to apply model to the input data.
    """
    
    # at the beginning of the batch data, start the record
    start_time_str = os.environ.get('START_TIME_UTC', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    start_time = datetime.datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
    logger.info(f"Prediction start time: {start_time}")
        
    # Ensure we use the features as specified in the model's config
    features_df = input_data[model.features]
    predictions = model.predict(features_df)
    features_df['prediction'] = predictions.tolist()
    
    end_time_str = os.environ.get('END_TIME_UTC', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    end_time = datetime.datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S')
    duration = end_time - start_time
    logger.info(f"Prediction end time: {end_time}")
    logger.info(f"Total prediction time: {duration}")
    
    return features_df

def output_fn(prediction, accept):
    """
    Format and return the prediction output.
    """
    if accept == "application/json" or accept == "text/csv":
        return json.dumps(prediction.to_dict(orient="records"))
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
