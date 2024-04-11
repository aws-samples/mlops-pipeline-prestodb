## Install all the requirements: includes prestodb as well as the python client it runs on
import sys
import subprocess
## Install dependencies
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

import logging
import prestodb
import pandas as pd

## define the logger
logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

## function to connect to the presto server
def _connect_presto_server(args, username, password, catalog, schema):
    """
    Connect to the Presto server.
    """
    logger.info(f"args={args}, username={username}, password={'is set' if password else 'not set'}, catalog={catalog}, schema={schema}")
    
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


def _run_query(args, username, password, catalog, schema, query):
    
    conn = _connect_presto_server(args, username, password, catalog, schema)
    cur = conn.cursor()

    cur.execute(query)
    rows = cur.fetchall()
    columns = [desc[0] for desc in cur.description]

    df = pd.DataFrame(rows, columns=columns)
    logger.info(f"returning dataframe of shape={df.shape} read from PrestoDB")
    cur.close()
    conn.close()
    logger.info("Data fetched successfully.")
    
    return df


def fetch_data_from_presto(args, username, password, catalog, schema, query):
    """
    Fetch data from Presto and return it as a pandas DataFrame.
    """
    try:
        df = _run_query(args, username, password, catalog, schema, query)
    except:
        try:
            df = _run_query(args, username, password, catalog, schema, query)
        except Exception as e:
            print(e)
            raise ValueError("PRESTO FETCH ERROR")
    return df
