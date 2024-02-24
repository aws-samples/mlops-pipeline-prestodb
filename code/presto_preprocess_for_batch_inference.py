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

## Install dependencies
subprocess.check_call([sys.executable, "-m", "pip", "install", "presto-python-client==0.8.4", "boto3==1.24.17", "pandas==1.1.3"])

## define the logger
logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

## Connect to the presto server running on your EC2 instance
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

    # save dataframe locally so that the processing job can upload it to S3
    batch_dir = "/opt/ml/processing/batch"
    fpath = os.path.join(batch_dir, "batch_data.csv")
    df.to_csv(fpath, index=False)
    logger.info(f"batch data is saved to --> {fpath}")

    logger.info(f"Preview of the first 10 rows of the batch data for visibility:\n{df.head(10)}")