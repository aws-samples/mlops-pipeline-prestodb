import os
import json
import boto3
import logging
import argparse
import pandas as pd
from query import BATCH_INFERENCE_QUERY
from presto_utils import fetch_data_from_presto
 
## define the logger
logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Parse arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, required=True)
    parser.add_argument('--port', type=int, required=True) 
    parser.add_argument('--region', type=str, required=True)
    parser.add_argument('--presto_credentials_key', type=str, required=True)
    parser.add_argument('--presto_catalog', type=str, required=True)
    parser.add_argument('--presto_schema', type=str, required=True)
    parser.add_argument('--query', type=str, help='The PrestoDB query to run')
    

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
    df = fetch_data_from_presto(args, username, password, args.presto_catalog, args.presto_schema, BATCH_INFERENCE_QUERY)
    logger.info(f"read data of shape={df.shape} for batch inference")

    # save dataframe locally so that the processing job can upload it to S3
    batch_dir = "/opt/ml/processing/batch"
    fpath = os.path.join(batch_dir, "batch_data.csv")
    df.to_csv(fpath, index=False)
    logger.info(f"batch data is saved to --> {fpath}")

    logger.info(f"Preview of the first 10 rows of the batch data for visibility:\n{df.head(10)}")
