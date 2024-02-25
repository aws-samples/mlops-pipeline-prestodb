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


# Main training function
def train_and_evaluate(features, target, model_dir):
    
    # Paths are set based on the input channels names provided in the TrainingInput configuration
    train_data_path = os.path.join(os.getenv('SM_CHANNEL_TRAIN'), 'train.csv')
    logger.info(f"training path stored by the pipeline step .... {train_data_path}...")
    
    test_data_path = os.path.join(os.getenv('SM_CHANNEL_TEST'), 'test.csv')
    logger.info(f"test data path stored by the pipeline step .... {test_data_path}...")

    # Load train and test datasets
    train_df = pd.read_csv(train_data_path)
    logger.info(f"train df -> {train_df.head(5)}")
    
    test_df = pd.read_csv(test_data_path)
    logger.info(f"test df -> {test_df.head(5)}")

    # Preparing the data
    X_train = train_df[features].values
    y_train = train_df[target].values

    ## Preparing the test data
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
    config = {'TARGET': target,
        'FEATURES': features}

    # Save the config file to the same directory as the model so the data can be used during the batch inference step of the pipeline
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f)
        logger.info(f"target and features stored in model config.json to be accessed during batch inference.... ")

    # Save the model to the specified directory
    joblib.dump(clf, os.path.join(model_dir, "model.joblib"))

if __name__ == "__main__":
    
    # Parse arguments from the command line specifically for the target and the features
    parser = argparse.ArgumentParser()
    
    ## Represents the targets and the features required for training
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
    
    # Extract and use the arguments
    target = args.target

    # Safe parsing of features argument into a Python list
    try:
        features = literal_eval(args.features)
        if not isinstance(features, list):
            raise ValueError("FEATURES argument is not a valid list.")
    except (ValueError, SyntaxError):
        raise ValueError("Invalid format for --features. Please provide a list of feature names.")


    # Hyperparameters
    n_estimators = args.n_estimators
    logger.info(f"n_estimators of training this random forest classifier -> {n_estimators}")
    max_depth = args.max_depth
    logger.info(f"max_depth of training this random forest classifier -> {max_depth}")
    min_samples_split = args.min_samples_split
    logger.info(f"min_samples_split of training this random forest classifier -> {min_samples_split}")
    max_features = args.max_features
    logger.info(f"max_features of training this random forest classifier -> {max_features}")

    # Ensure that TARGET and FEATURES are defined
    if not target or not features:
        raise ValueError("Target and Features must be specified either through environment variables or command-line arguments.....")
    
    # Use SageMaker's default environment variables to define paths
    model_dir = os.getenv('SM_MODEL_DIR', '.')
    
    logger.info(f"features of training this random forest classifier -> {features}")
    
    logger.info(f"target parameter of training this random forest classifier -> {target}")

    # Train and evaluate the model
    train_and_evaluate(features, target, model_dir)