import os, datetime
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, DESCENDING


load_dotenv()


uri = os.getenv("MONGODB_URI")
client = MongoClient(uri)

db_name = os.getenv("MONGODB_DB")
db = client[db_name]


# 1) users
db.create_collection("users")
db.users.create_index("email", unique=True)

# 2) sources
db.create_collection("sources")
db.sources.create_index("url", unique=True)
db.sources.create_index([("publisher", ASCENDING), ("publishedAt", DESCENDING)])

# 3) datapoints (optional catalog of Snowflake keys)
db.create_collection("datapoints")
db.datapoints.create_index(
    [("dataset", ASCENDING), ("iso2", ASCENDING), ("date", ASCENDING), ("metric", ASCENDING)],
    unique=True
)

# 4) annotations (main) + JSON Schema validator
db.create_collection("annotations")
db.command("collMod", "annotations", validator={
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["datapointId", "type", "authorId", "createdAt"],
        "properties": {
            "datapointId": {"bsonType": "string"},
            "type": {"enum": ["comment", "correction", "quality_flag", "tag"]},
            "text": {"bsonType": ["string", "null"]},
            "labels": {"bsonType": "array", "items": {"bsonType": "string"}},
            "authorId": {"bsonType": "objectId"},
            "sourceIds": {"bsonType": "array", "items": {"bsonType": "objectId"}},
            "status": {"enum": ["active", "resolved", "hidden"]},
            "attachments": {"bsonType": "array"},
            "createdAt": {"bsonType": "date"},
            "updatedAt": {"bsonType": ["date", "null"]},
            "version": {"bsonType": ["int", "long"], "minimum": 1}
        }
    }
})
db.annotations.create_index([("datapointId", ASCENDING), ("createdAt", DESCENDING)])
db.annotations.create_index("labels")
db.annotations.create_index([("authorId", ASCENDING), ("createdAt", DESCENDING)])





print(db.list_collection_names())
