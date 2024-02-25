#!/usr/bin/env python
# coding: utf-8

## Install all the requirements: includes prestodb as well as the python client it runs on

import os
import sys
import json
import boto3
import logging
import argparse
import subprocess
import numpy as np
import pandas as pd
from query import TRAINING_DATA_QUERY
from presto_utils import fetch_data_from_presto

## define the logger
logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def split_data(df):
    """
    Preprocess the fetched data and split it into train, validation, and test sets.
    """
    # Splitting the data into train, validation, and test
    train_data, validation_data, test_data = np.split(
        df.sample(frac=1, random_state=1729),
        [int(args.train_split * len(df)), int(args.test_split * len(df))]
    )

    return train_data, validation_data, test_data

def save_dataframes(train_data, validation_data, test_data, base_dir="DSG_order"):
    """
    Save the train, validation, and test DataFrames to CSV files.
    """
    os.makedirs(base_dir, exist_ok=True)
    
    train_dir = "/opt/ml/processing/train"
    validation_dir = "/opt/ml/processing/validation"
    test_dir = "/opt/ml/processing/test"
    
    paths = {
        "train": os.path.join(train_dir, "train.csv"),
        "validation": os.path.join(validation_dir, "validation.csv"),
        "test": os.path.join(test_dir, "test.csv")
    }
    
    train_data.to_csv(paths["train"], index=False)
    validation_data.to_csv(paths["validation"], index=False)
    test_data.to_csv(paths["test"], index=False)

    logger.info("Train, validation, and test sets saved.")

    return paths

if __name__ == "__main__":
    # Parse arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, required=True)
    parser.add_argument('--port', type=int, required=True) 
    parser.add_argument('--region', type=str, required=True)
    parser.add_argument('--presto_credentials_key', type=str, required=True)
    parser.add_argument('--presto_catalog', type=str, required=True)
    parser.add_argument('--presto_schema', type=str, required=True)
    parser.add_argument("--train_split", type=float, help="The train split metric as a float.")
    parser.add_argument("--test_split", type=float, help="The test split metric as a float.")
    
    ## add start time and end time as str -- todo
    # parser.add_argument('--dataframe-path', type=str, required=True, help='The local path to the CSV file to upload')

    args = parser.parse_args()
    logger.info(f"args={args}")

    client = boto3.client('secretsmanager', region_name=args.region)
    response = client.get_secret_value(SecretId=args.presto_credentials_key)
    secrets_credentials = json.loads(response['SecretString'])
    password = secrets_credentials.get('password')
    username = secrets_credentials.get('username', 'ec2-user')

    # Fetch data from Presto and store it in a DataFrame
    df = fetch_data_from_presto(args, username, password, args.presto_catalog, args.presto_schema, TRAINING_DATA_QUERY)

    logger.info("read data of shape={df.shape} for training")
    logger.info(f"boto3 version={boto3.__version__}, pandas version={pd.__version__}")
    
    # Preprocess the data and split it
    train_data, validation_data, test_data = split_data(df)
    
    # Save the split datasets
    save_dataframes(train_data, validation_data, test_data)
