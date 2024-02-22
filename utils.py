import re
import os
import sys
import yaml
import math
import boto3
import logging
import requests
import posixpath
import subprocess
import unicodedata
from typing import Dict
from pathlib import Path
from botocore.exceptions import NoCredentialsError

## define the logger
logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


# utility functions
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
    