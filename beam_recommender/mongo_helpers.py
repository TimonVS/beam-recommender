import os
from pymongo import MongoClient
from pymongo.database import Database
import pandas as pd

def create_mongo_uri(username, password, database, host='localhost', port=27017):
    return 'mongodb://{}:{}@{}:{}/{}'.format(username, password, host, port, database)

def read_mongo(db, collection, query={}, uri='mongodb://localhost:27017', no_id=True):
    """Read from Mongo and Store into DataFrame"""
    db = connect_mongo(uri, db)
    cursor = db[collection].find(query)
    df = pd.DataFrame.from_records(cursor)

    # Delete _id
    if no_id:
        del df['_id']

    return df

def connect_mongo(**kwargs) -> Database:
    host = kwargs.get('host', 'localhost')
    port = kwargs.get('port', '27017')
    username = kwargs.get('username', os.environ.get('MONGODB_USERNAME'))
    password = kwargs.get('password', os.environ.get('MONGODB_PASSWORD'))
    database = kwargs.get('database', os.environ.get('MONGODB_DATABASE'))
    uri = create_mongo_uri(host=host, port=port, username=username, password=password, database=database)
    client = MongoClient(uri)
    return client[database]
