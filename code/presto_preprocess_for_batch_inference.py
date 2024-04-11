import os
import json
import boto3
import logging
import argparse
import pandas as pd
from datetime import timedelta
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
    parser.add_argument('--start_time', type=str, required=True)
    parser.add_argument('--end_time', type=str, required=True)
    
    args = parser.parse_args()
    logger.info(f"args={args}")
    
    start, end = args.start_time, args.end_time

    if end is None:
        end = ""
    else:
        end = f"AND created_at <= date_parse('{end}', '%Y-%m-%d %H:%i:%s')"

    s_date = (pd.Timestamp(start) - timedelta(days=1)).date().isoformat()
    e_date = (pd.Timestamp(start) + timedelta(days=1)).date().isoformat()
    
    query = BATCH_INFERENCE_QUERY.format(s_date=s_date, e_date=e_date, start=start, end=end)

    client = boto3.client('secretsmanager', region_name=args.region)
    response = client.get_secret_value(SecretId=args.presto_credentials_key)
    secrets_credentials = json.loads(response['SecretString'])
    password = secrets_credentials.get('password')
    username = secrets_credentials.get('username', 'ec2-user')
    
    BATCH_INFERENCE_QUERY = f"{BATCH_INFERENCE_QUERY}".format

    # Fetch data from Presto and store it in a DataFrame
    df = fetch_data_from_presto(args, username, password, args.presto_catalog, args.presto_schema, query)
    logger.info(f"read data of shape={df.shape} for batch inference")

    # save dataframe locally so that the processing job can upload it to S3
    batch_dir = "/opt/ml/processing/batch"
    fpath = os.path.join(batch_dir, "batch_data.csv")
    df.to_csv(fpath, index=False)
    logger.info(f"batch data is saved to --> {fpath}")

    logger.info(f"Preview of the first 10 rows of the batch data for visibility:\n{df.head(10)}")
