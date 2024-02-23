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

# Parse arguments from the command line
parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, required=True)
parser.add_argument('--port', type=int, required=True) 
parser.add_argument('--user', type=str, required=True)

args = parser.parse_args()

# collect snowflake credentials from Secrets Manager
PRESTO_CREDENTIALS = False
client = boto3.client('secretsmanager', region_name='us-east-1')
response = client.get_secret_value(SecretId="presto-credentials")
secrets_credentials = json.loads(response['SecretString'])
# presto_password = secrets_credentials['password']
presto_password = 'incorrect_password'
presto_username = secrets_credentials['username']

## Install dependencies
subprocess.check_call([sys.executable, "-m", "pip", "install", "presto-python-client==0.8.4", "boto3", "pandas"])

## define the logger
logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Connect to the presto server running on your EC2 instance
#from prestodb import dbapi, auth
import prestodb
## function to connect to the presto server
def connect_presto_server(catalog, schema):
    """
    Connect to the Presto server.
    """
    
    if PRESTO_CREDENTIALS: 
        ## connect to presto using password authentication
        conn = prestodb.dbapi.connect(
            host=args.host,
            port=args.port,
            user=presto_username,
            catalog=catalog,
            schema=schema,
            http_scheme='https',
            auth=prestodb.auth.BasicAuthentication(presto_username, presto_password)
        )
        logger.info(f"user name used to connect to the presto server: {presto_username}...")
    else:
        conn = prestodb.dbapi.connect(
            host=args.host,
            port=args.port,
            user=args.user,
            catalog=catalog,
            schema=schema,
    )
    logger.info("Connected successfully to Presto server.")
    return conn

def fetch_data_from_presto():
    """
    Fetch data from Presto and return it as a pandas DataFrame.
    """
    conn = connect_presto_server('tpch', 'tiny')  # Example catalog and schema
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
        o.orderkey
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
    logger.info("Starting data extraction and preprocessing pipeline.")
    
    # Fetch data from Presto
    df = fetch_data_from_presto()
    
    # Preprocess the data and split it
    train_data, validation_data, test_data = preprocess_and_split_data(df)
    
    # Save the split datasets
    save_dataframes(train_data, validation_data, test_data)
