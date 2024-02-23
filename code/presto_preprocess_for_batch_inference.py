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
from io import StringIO


# Parse arguments from the command line
parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, required=True)
parser.add_argument('--port', type=int, required=True) 
parser.add_argument('--user', type=str, required=True)
## add start time and end time as str -- todo
# parser.add_argument('--dataframe-path', type=str, required=True, help='The local path to the CSV file to upload')

args = parser.parse_args()

## collect snowflake credentials from Secrets Manager
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

## Connect to the presto server running on your EC2 instance
import prestodb
from prestodb import dbapi

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
        logger.info("Connected successfully to Presto server.")
        return conn
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


def upload_dataframe_to_s3(df):
    """
    Uploads the given DataFrame to an S3 bucket.
    """
    
    # os.makedirs(base_dir, exist_ok=True)
    
    batch_dir = "/opt/ml/processing/batch"
    
    paths = {
        "batch_data": os.path.join(batch_dir, "batch_data.csv"),
    }
    
    csv_buffer = StringIO()
    df.to_csv(paths["batch_data"], index=False)


    csv_buffer = StringIO()
    
    preview_buffer = StringIO()
    df.head(10).to_csv(preview_buffer, index=False)
    logger.info(f"Preview of the first 10 rows of the batch data for visibility:\n{preview_buffer.getvalue()}")

    logger.info(f"batch data is saved to --> {paths}")
    return paths

if __name__ == "__main__":
    # Fetch data from Presto and store it in a DataFrame
    df = fetch_data_from_presto()

    # Upload the DataFrame to S3
    upload_dataframe_to_s3(df)


