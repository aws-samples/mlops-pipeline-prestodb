"""Evaluation script for measuring model accuracy."""

import json
import joblib
import logging
import pathlib
import tarfile
import argparse
import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    # Parse arguments from the command line specifically for the target and the features
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--features', type=str, required=True) 

    # Parse arguments
    args = parser.parse_args()
    logger.info(f"args={args}")

    # Extract and use the arguments
    target = args.target

    # Convert features from string to list to process for this evaluation script
    try:
        features = literal_eval(args.features)
        if not isinstance(features, list):
            raise ValueError("--features argument must be a list of feature names.")
    except (ValueError, SyntaxError) as e:
        logger.error(f"Error parsing --features argument: {e}")
        raise

    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    #must have to change the name? 
    logger.debug("Loading random forest model.")
    #model = pickle.load(open("sklearn-model", "rb"))
    model = joblib.load('model.joblib')
    logger.debug("Loading test input data.")
    
    # Paths are set based on the input channels names provided in the TrainingInput configuration
    # test_data_path = os.path.join(os.getenv('SM_CHANNEL_TEST'), 'test.csv')
    test_data_path = "/opt/ml/processing/test/test.csv"
    logger.info(f"the test data path --> {test_data_path}")
    df = pd.read_csv(test_data_path, header=None)
    logger.debug("Reading test data.")

    # Load dataset
    test_df = pd.read_csv(test_data_path)
    logger.info(f"test df -> {test_df.head(5)}")

    # Prepare data
    X_test = test_df[features].values
    y_test = test_df[target].values
    

    logger.info("Performing predictions against test data.")
    prediction_probabilities = model.predict(X_test)
    predictions = np.round(prediction_probabilities)

    precision = precision_score(y_test, predictions, zero_division=1)
    recall = recall_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    fpr, tpr, _ = roc_curve(y_test, prediction_probabilities)

    logger.debug("Accuracy: {}".format(accuracy))
    logger.debug("Precision: {}".format(precision))
    logger.debug("Recall: {}".format(recall))
    logger.debug("Confusion matrix: {}".format(conf_matrix))

    # Available metrics to add to model: https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html
    report_dict = {
        "binary_classification_metrics": {
            "accuracy": {"value": accuracy, "standard_deviation": "NaN"},
            "precision": {"value": precision, "standard_deviation": "NaN"},
            "recall": {"value": recall, "standard_deviation": "NaN"},
            "confusion_matrix": {
                "0": {"0": int(conf_matrix[0][0]), "1": int(conf_matrix[0][1])},
                "1": {"0": int(conf_matrix[1][0]), "1": int(conf_matrix[1][1])},
            },
            "receiver_operating_characteristic_curve": {
                "false_positive_rates": list(fpr),
                "true_positive_rates": list(tpr),
            },
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    evaluation_path = f"{output_dir}/evaluation.json"
    try:
        with open(evaluation_path, "w") as f:
            f.write(json.dumps(report_dict))
            logger.info("Successfully wrote evaluation report.")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
