from snowflake.snowpark import Session
from dotenv import load_dotenv
import os
import pandas as pd
load_dotenv()


connection_parameters = {
    "account":   os.getenv("SNOWFLAKE_ACCOUNT"),
    "user":      os.getenv("SNOWFLAKE_USER"),
    "password":  os.getenv("SNOWFLAKE_PASSWORD"),
    "role":      os.getenv("SNOWFLAKE_ROLE"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database":  os.getenv("SNOWFLAKE_DATABASE"),
    "schema":    os.getenv("SNOWFLAKE_SCHEMA")
}


session = Session.builder.configs(connection_parameters).create()


gdp = pd.read_csv("../KaggleDatasets/GDP per capita.csv")
pop = pd.read_csv("../KaggleDatasets/Population.csv")
hdi = pd.read_csv("../KaggleDatasets/HDI.csv", encoding="latin-1")

session.write_pandas(gdp, "RAW_GDP_PER_CAPITA_STG", auto_create_table=True, overwrite=True)
session.write_pandas(pop, "RAW_POPULATION_STG",     auto_create_table=True, overwrite=True)
session.write_pandas(hdi, "RAW_HDI_STG",            auto_create_table=True, overwrite=True)

print("Uploaded: RAW_GDP_PER_CAPITA_STG, RAW_POPULATION_STG, RAW_HDI_STG")
