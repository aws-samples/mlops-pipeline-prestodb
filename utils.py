import yaml
import json
import boto3
import logging
import requests
from typing import Dict
from botocore.exceptions import NoCredentialsError

## define the logger
logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# utility functions

make_s3_prefix = lambda x, dttm: f"{x}/yyyy={dttm.year}/mm={dttm.month}/dd={dttm.day}/hh={dttm.hour}/mm={dttm.minute}"

def print_pipeline_execution_summary(steps, name):
    failed_steps = 0
    steps_that_had_to_retried = 0
    logger.info(f"pipeline steps={json.dumps(steps, indent=2, default=str)}")
    for s in steps:
        if s['StepStatus'] != 'Succeeded':
            logger.error(f"FAILED STEP: {s}")
            failed_steps += 1
        if s['AttemptCount'] > 1:
            logger.error(f"Retried STEP: {s}")
            steps_that_had_to_retried += 1
    logger.info(f"for pipeline={name}, failed_steps={failed_steps}, steps_that_had_to_retried={steps_that_had_to_retried}")

def load_config(config_file) -> Dict:
    """
    Load configuration from a local file or an S3 URI.

    :param config_file: Path to the local file or S3 URI (s3://bucket/key)
    :return: Dictionary with the loaded configuration
    """

    # Check if config_file is an S3 URI
    if config_file.startswith("s3://"):
        try:
            # Parse S3 URI
            s3_client = boto3.client('s3')
            bucket, key = config_file.replace("s3://", "").split("/", 1)

            # Get object from S3 and load YAML
            response = s3_client.get_object(Bucket=bucket, Key=key)
            return yaml.safe_load(response["Body"])
        except NoCredentialsError:
            print("AWS credentials not found.")
            raise
        except Exception as e:
            print(f"Error loading config from S3: {e}")
            raise
    # Check if config_file is an HTTPS URL
    elif config_file.startswith("https://"):
        try:
            response = requests.get(config_file)
            response.raise_for_status()  # Raises a HTTPError if the response was an error
            return yaml.safe_load(response.text)
        except requests.exceptions.RequestException as e:
            print(f"Error loading config from HTTPS URL: {e}")
            raise
    else:
        # Assume local file system if not S3 or HTTPS
        try:
            with open(config_file, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading config from local file system: {e}")
            raise
    