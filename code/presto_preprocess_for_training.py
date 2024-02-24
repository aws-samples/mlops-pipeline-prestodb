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

## Install dependencies
subprocess.check_call([sys.executable, "-m", "pip", "install", "presto-python-client==0.8.4", "boto3==1.24.17", "pandas==1.1.3"])

## define the logger
logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Connect to the presto server running on your EC2 instance
#from prestodb import dbapi, auth
import prestodb
from prestodb import dbapi
## function to connect to the presto server
def connect_presto_server(args, username, password, catalog, schema):
    """
    Connect to the Presto server.
    """
    
    if password: 
        ## connect to presto using password authentication
        conn = prestodb.dbapi.connect(
            host=args.host,
            port=args.port,
            user=username,
            catalog=catalog,
            schema=schema,
            http_scheme='https',
            auth=prestodb.auth.BasicAuthentication(username, password)
        )
        logger.info(f"user name used to connect to the presto server: {username}...")
    else:
        conn = prestodb.dbapi.connect(
            host=args.host,
            port=args.port,
            user=username,
            catalog=catalog,
            schema=schema,
    )
    logger.info("Connected successfully to Presto server.")
    return conn

def fetch_data_from_presto(args, username, password):
    """
    Fetch data from Presto and return it as a pandas DataFrame.
    """
    conn = connect_presto_server(args, username, password, 'tpch', 'tiny')  # Example catalog and schema
    cur = conn.cursor()

    query = """
    SELECT
        o.orderkey,
        COUNT(l.linenumber) AS lineitem_count,
        SUM(l.quantity) AS total_quantity,
        AVG(l.discount) AS avg_discount,
        SUM(l.extendedprice) AS total_extended_price,
        o.orderdate,
        o.orderpriority,
        CASE
            WHEN SUM(l.extendedprice) > 20000 THEN 1
            ELSE 0
        END AS high_value_order
    FROM
        orders o
    JOIN
        lineitem l ON o.orderkey = l.orderkey
    GROUP BY
        o.orderkey,
        o.orderdate,
        o.orderpriority
    ORDER BY 
        RANDOM() 
    LIMIT 5000
    """
    cur.execute(query)
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]

    df = pd.DataFrame(rows, columns=columns)
    cur.close()
    conn.close()
    logger.info("Data fetched successfully.")
    return df

def preprocess_and_split_data(df):
    """
    Preprocess the fetched data and split it into train, validation, and test sets.
    """
    # Simple classification based on high_value_order
    df['Order Value Classification'] = df['high_value_order'].apply(lambda x: "1" if x == 1 else "0")

    # Splitting the data into train, validation, and test
    train_data, validation_data, test_data = np.split(
        df.sample(frac=1, random_state=1729),
        [int(0.7 * len(df)), int(0.9 * len(df))]
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

    ## add start time and end time as str -- todo
    # parser.add_argument('--dataframe-path', type=str, required=True, help='The local path to the CSV file to upload')

    args = parser.parse_args()
    client = boto3.client('secretsmanager', region_name=args.region)
    response = client.get_secret_value(SecretId=args.presto_credentials_key)
    secrets_credentials = json.loads(response['SecretString'])
    password = secrets_credentials.get('password')
    username = secrets_credentials.get('username', 'ec2-user')
    logger.info(f"the secrets password recorded.... {username}")

    logger.info(f"boto3 version={boto3.__version__}, pandas version={pd.__version__}")
    
    
    # Fetch data from Presto and store it in a DataFrame
    df = fetch_data_from_presto(args, username, password)

    logger.info("Starting data extraction and preprocessing pipeline.")
    logger.info(f"boto3 version={boto3.__version__}, pandas version={pd.__version__}")
    
    # Preprocess the data and split it
    train_data, validation_data, test_data = preprocess_and_split_data(df)
    
    # Save the split datasets
    save_dataframes(train_data, validation_data, test_data)