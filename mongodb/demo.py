import os, datetime
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

uri = os.getenv("MONGODB_URI")
client = MongoClient(uri)
db_name = os.getenv("MONGODB_DB")
db = client[db_name]
user = db.users.find_one({"email": "email123@gmail.com"})
user_id = user["_id"]


#adding user Boris to DB
new_user = {
    "email": "email123@gmail.com",
    "name": "Boris",
    "role": "editor",
    "createdAt": datetime.datetime.utcnow(),
    "updatedAt": datetime.datetime.utcnow()
}

result = db.users.insert_one(new_user)


#adding test annotation
db.annotations.insert_one({
    "datapointId": "JHU_COVID_19|IT|2021-03-12|Confirmed",
    "type": "comment",
    "text": "Spike likely due to backlog after regional holiday.",
    "labels": ["reporting-lag", "holiday"],
    "authorId": user_id,
    "status": "active",
    "sourceIds": [],
    "attachments": [{"kind": "url", "value": "https://www.salute.gov.it/", "title": "MoH bulletin"}],
    "createdAt": datetime.datetime.utcnow(),
    "updatedAt": None,
    "version": 1
})

#adding another test annotation
db.annotations.insert_one({
    "datapointId": "OWID_VACCINATIONS|FJ|2021-07-15|total_vaccinations_per_hundred",
    "type": "comment",
    "text": "Vaccinations spiked after Fiji introduced a mandatory vaccination policy for all workers in July 2021.",
    "labels": ["policy-change", "mandatory-vaccination"],
    "authorId": user_id,
    "status": "active",
    "sourceIds": [],
    "attachments": [{
        "kind": "url",
        "value": "https://www.reuters.com/world/asia-pacific/fiji-mandates-covid-19-vaccinations-all-workers-2021-07-08/",
        "title": "Reuters: Fiji mandates COVID-19 vaccinations for all workers"
    }],
    "createdAt": datetime.datetime.utcnow(),
    "updatedAt": None,
    "version": 1
})

