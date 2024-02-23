## represents the training script to train the model - used in the training step of the first pipeline
import os
import sys
import json
import joblib
import logging
import argparse
import subprocess
import numpy as np
import pandas as pd
from typing import List
from io import StringIO 
from ast import literal_eval
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


## define the logger
logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Parse arguments from the command line specifically for the target and the features
parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, required=True)
parser.add_argument('--features', type=str, required=True) 

# Adding arguments for hyperparameters with default values as per your configuration file
parser.add_argument('--n_estimators', type=int,
                    help='Number of trees in the forest. Defaults to 75 as per the configuration file.')
parser.add_argument('--max_depth', type=int,
                    help='Maximum depth of the tree. Defaults to 10 as per the configuration file.')
parser.add_argument('--min_samples_split', type=int,
                    help='Minimum number of samples required to split an internal node. Defaults to 2 as per the configuration file.')
parser.add_argument('--max_features', type=str,
                    help='The number of features to consider when looking for the best split. Defaults to "sqrt" as per the configuration file.')

# Parse arguments
args = parser.parse_args()

# Attempt to use environment variables if set; otherwise, use command-line arguments
TARGET = os.environ.get('TARGET', args.target)
FEATURES = os.environ.get('FEATURES', args.features)

# Safe parsing of features argument into a Python list
try:
    FEATURES = literal_eval(args.features)
    if not isinstance(FEATURES, list):
        raise ValueError("FEATURES argument is not a valid list.")
except (ValueError, SyntaxError):
    raise ValueError("Invalid format for --features. Please provide a list of feature names.")


# Hyperparameters
n_estimators = args.n_estimators
max_depth = args.max_depth
min_samples_split = args.min_samples_split
max_features = args.max_features

# Ensure that TARGET and FEATURES are defined
if not TARGET or not FEATURES:
    raise ValueError("TARGET and FEATURES must be specified either through environment variables or command-line arguments.")

# Main training function
def train_and_evaluate(features, target, model_dir):
    # Paths are set based on the input channels names provided in the TrainingInput configuration
    train_data_path = os.path.join(os.getenv('SM_CHANNEL_TRAIN'), 'train.csv')
    test_data_path = os.path.join(os.getenv('SM_CHANNEL_TEST'), 'test.csv')

    # Load dataset
    train_df = pd.read_csv(train_data_path)
    logger.info(f"train df -> {train_df.head(5)}")
    test_df = pd.read_csv(test_data_path)
    logger.info(f"test df -> {test_df.head(5)}")

    # Prepare data
    X_train = train_df[features].values
    y_train = train_df[target].values

    X_test = test_df[features].values
    y_test = test_df[target].values

    # Define and train the model
    # Define and train the model with parsed hyperparameters
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_features=max_features,
        random_state=42,
        class_weight='balanced')

    clf.fit(X_train, y_train)

    y_pred_proba = clf.predict_proba(X_test)[:, 1]  # Get probability scores for the positive class
    y_pred = clf.predict(X_test)  # Predict the class labels for the test set
    auc_score = roc_auc_score(y_test, y_pred_proba)

    # Log the AUC score and evaluation metrics in the expected format
    logger.info(f"auc {auc_score}")
    logger.info(classification_report(y_test, y_pred))
    logger.info("Confusion Matrix:")
    logger.info(confusion_matrix(y_test, y_pred))
    
    # list of feature names used in training
    config = {'TARGET': TARGET,
        'FEATURES': FEATURES}

    # Save the config file to the same directory as the model
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f)

    # Save the model to the specified directory
    joblib.dump(clf, os.path.join(model_dir, "model.joblib"))

if __name__ == "__main__":
    
    # Use SageMaker's default environment variables to define paths
    model_dir = os.getenv('SM_MODEL_DIR', '.')
    
    # Specify features and target
    ## have all variables be abstract and pass features as cli args, and create features array
    features = FEATURES
    logger.info(f"features of training this random forest classifier -> {features}")
    target = TARGET 
    logger.info(f"target parameter of training this random forest classifier -> {target}")
    n_estimators = n_estimators
    max_depth = max_depth
    min_samples_split = min_samples_split
    max_features = max_features
    
    # Train and evaluate the model
    train_and_evaluate(features, target, model_dir)
