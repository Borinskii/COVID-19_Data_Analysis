# test_connections.py
"""
Test script to verify Snowflake and MongoDB connections
Run this before starting the API to ensure everything is configured correctly
"""

import os
import sys
from dotenv import load_dotenv
import snowflake.connector
from pymongo import MongoClient
from datetime import datetime, timezone

# Load environment variables
load_dotenv()


def test_snowflake_connection():
    """Test Snowflake connection and verify required tables"""
    print("\n" + "=" * 50)
    print("Testing Snowflake Connection")
    print("=" * 50)

    # Required tables for the API
    required_tables = [
        "JHU_COVID_19",
        "OWID_VACCINATIONS",
        "APPLE_MOBILITY",
        "CDC_TESTING",
        "CDC_INPATIENT_BEDS_ICU_ALL",
        "HUM_RESTRICTIONS_AIRLINE"
    ]

    try:
        # Get credentials from environment
        account = os.getenv("SNOWFLAKE_ACCOUNT")
        user = os.getenv("SNOWFLAKE_USER")
        password = os.getenv("SNOWFLAKE_PASSWORD")
        warehouse = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")

        print(f"Account: {account}")
        print(f"User: {user}")
        print(f"Warehouse: {warehouse}")

        # Connect to Snowflake
        conn = snowflake.connector.connect(
            account=account,
            user=user,
            password=password,
            warehouse=warehouse
        )

        cursor = conn.cursor()

        # List databases
        cursor.execute("SHOW DATABASES")
        databases = cursor.fetchall()

        print("\n✓ Snowflake connection successful!")
        print("\nAvailable databases:")
        for db in databases:
            print(f"  - {db[1]}")

        # Test COVID19_EPIDEMIOLOGICAL_DATA database
        print("\nTesting COVID19_EPIDEMIOLOGICAL_DATA database...")
        cursor.execute("USE DATABASE COVID19_EPIDEMIOLOGICAL_DATA")
        cursor.execute("USE SCHEMA PUBLIC")

        # Get all tables
        cursor.execute("SHOW TABLES")
        all_tables = cursor.fetchall()
        available_tables = [table[1] for table in all_tables]

        print("Checking required tables:")
        missing_tables = []

        for table in required_tables:
            if table in available_tables:
                print(f"  ✓ {table}")

                # Test each table with a simple query
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table} LIMIT 1")
                    count_result = cursor.fetchone()
                    row_count = count_result[0] if count_result else 0
                    print(f"    → {row_count:,} total rows")

                    # Test sample data
                    cursor.execute(f"SELECT * FROM {table} LIMIT 3")
                    sample_data = cursor.fetchall()
                    if sample_data:
                        print(f"    → Sample data available")

                except Exception as table_error:
                    print(f"    ✗ Error accessing {table}: {table_error}")
                    missing_tables.append(table)
            else:
                print(f"  ✗ {table} - NOT FOUND")
                missing_tables.append(table)

        # Test COVID_ENRICHED database
        print("\nTesting COVID_ENRICHED database...")
        cursor.execute("USE DATABASE COVID_ENRICHED")
        cursor.execute("USE SCHEMA RAW")

        cursor.execute("SHOW TABLES")
        enriched_tables = cursor.fetchall()

        print("Tables in COVID_ENRICHED.RAW:")
        enriched_required = ["RAW_GDP_PER_CAPITA_STG", "RAW_HDI_STG", "RAW_POPULATION_STG"]
        enriched_available = [table[1] for table in enriched_tables]

        for table in enriched_required:
            if table in enriched_available:
                print(f"  ✓ {table}")
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count_result = cursor.fetchone()
                    row_count = count_result[0] if count_result else 0
                    print(f"    → {row_count:,} rows")
                except Exception as e:
                    print(f"    ✗ Error: {e}")
            else:
                print(f"  ✗ {table} - NOT FOUND")
                missing_tables.append(table)

        conn.close()

        if missing_tables:
            print(f"\n⚠️  Missing tables: {missing_tables}")
            print("API functionality may be limited without these tables.")
            return True  # Still return True if connection works, just warn about missing tables

        return True

    except Exception as e:
        print(f"\n✗ Snowflake connection failed: {e}")
        return False


def test_mongodb_connection():
    """Test MongoDB connection and create collections if needed"""
    print("\n" + "=" * 50)
    print("Testing MongoDB Connection")
    print("=" * 50)

    try:
        # Get MongoDB URI from environment
        mongodb_uri = os.getenv("MONGODB_URI")
        mongodb_database = os.getenv("MONGODB_DATABASE", "COVID_COMMENTS")

        print(f"MongoDB URI: {mongodb_uri[:30]}...")
        print(f"Database: {mongodb_database}")

        # Connect to MongoDB
        client = MongoClient(mongodb_uri)

        # Test connection
        client.admin.command('ping')
        print("\n✓ MongoDB connection successful!")

        # Access database
        db = client[mongodb_database]

        # List existing collections
        collections = db.list_collection_names()
        print(f"\nCollections in {mongodb_database}:")
        if collections:
            for coll in collections:
                count = db[coll].count_documents({})
                print(f"  - {coll} ({count:,} documents)")
        else:
            print("  No collections yet (will be created automatically)")

        # Create collections with indexes if they don't exist
        required_collections = {
            "users": [
                {"field": "email", "unique": True}
            ],
            "annotations": [
                {"field": [("datapointId", 1), ("createdAt", -1)]},
                {"field": "labels"},
                {"field": [("authorId", 1), ("createdAt", -1)]}
            ],
            "datapoints": [
                {"field": [("dataset", 1), ("iso2", 1), ("date", 1), ("metric", 1)], "unique": True}
            ],
            "sources": [
                {"field": "url", "unique": True},
                {"field": [("publisher", 1), ("publishedAt", -1)]}
            ]
        }

        for collection_name, indexes in required_collections.items():
            if collection_name not in collections:
                print(f"\nCreating '{collection_name}' collection with indexes...")
                db.create_collection(collection_name)

                for index_config in indexes:
                    unique = index_config.get("unique", False)
                    field = index_config["field"]
                    db[collection_name].create_index(field, unique=unique)

                print(f"✓ '{collection_name}' collection created with {len(indexes)} indexes")
            else:
                print(f"✓ '{collection_name}' collection already exists")

        # Test write/read operation
        test_doc = {
            "test": True,
            "timestamp": datetime.now(timezone.utc),
            "message": "Connection test successful"
        }

        # Insert and then delete test document
        result = db.test_collection.insert_one(test_doc)
        db.test_collection.delete_one({"_id": result.inserted_id})
        print("\n✓ MongoDB read/write test successful!")

        client.close()
        return True

    except Exception as e:
        print(f"\n✗ MongoDB connection failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if your MongoDB Atlas cluster is running")
        print("2. Verify your username and password are correct")
        print("3. Make sure your IP address is whitelisted in MongoDB Atlas")
        print("4. Check if the connection string format is correct")
        return False


def test_table_schemas():
    """Test the schema of required tables to ensure API compatibility"""
    print("\n" + "=" * 50)
    print("Testing Table Schemas")
    print("=" * 50)

    try:
        # Get credentials from environment
        account = os.getenv("SNOWFLAKE_ACCOUNT")
        user = os.getenv("SNOWFLAKE_USER")
        password = os.getenv("SNOWFLAKE_PASSWORD")
        warehouse = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")

        conn = snowflake.connector.connect(
            account=account,
            user=user,
            password=password,
            warehouse=warehouse,
            database="COVID19_EPIDEMIOLOGICAL_DATA",
            schema="PUBLIC"
        )

        cursor = conn.cursor()

        # Test key table schemas
        test_tables = {
            "JHU_COVID_19": "SELECT COUNTRY_REGION, DATE, CASE_TYPE, CASES FROM JHU_COVID_19 LIMIT 1",
            "OWID_VACCINATIONS": "SELECT COUNTRY_REGION, DATE, TOTAL_VACCINATIONS FROM OWID_VACCINATIONS LIMIT 1",
            "APPLE_MOBILITY": "SELECT COUNTRY_REGION, DATE, TRANSPORTATION_TYPE FROM APPLE_MOBILITY LIMIT 1"
        }

        for table_name, test_query in test_tables.items():
            try:
                cursor.execute(test_query)
                columns = [col[0] for col in cursor.description]
                print(f"✓ {table_name} schema compatible")
                print(f"  Columns: {', '.join(columns[:5])}{'...' if len(columns) > 5 else ''}")
            except Exception as e:
                print(f"✗ {table_name} schema issue: {e}")

        conn.close()
        return True

    except Exception as e:
        print(f"✗ Schema testing failed: {e}")
        return False


def main():
    """Run all connection tests"""
    print("\n" + "=" * 50)
    print("COVID-19 Data Analysis API - Connection Test")
    print("=" * 50)

    # Check if .env file exists
    if not os.path.exists(".env"):
        print("\n✗ .env file not found!")
        print("Please create a .env file with your credentials")
        sys.exit(1)

    print("\n✓ .env file found")

    # Test connections
    snowflake_ok = test_snowflake_connection()
    mongodb_ok = test_mongodb_connection()
    schema_ok = test_table_schemas() if snowflake_ok else False

    # Summary
    print("\n" + "=" * 50)
    print("Connection Test Summary")
    print("=" * 50)

    if snowflake_ok and mongodb_ok:
        print("\n✓ All connections successful!")
        if schema_ok:
            print("✓ Table schemas are compatible!")
        else:
            print("⚠️  Some table schema issues detected")

        print("\nYou can now run the API with:")
        print("  python main.py")
        print("\nOr with auto-reload for development:")
        print("  uvicorn main:app --reload")
        print("\nAPI documentation will be available at:")
        print("  http://localhost:8000/docs")
        print("  http://localhost:8000/redoc")
    else:
        print("\n✗ Some connections failed")
        if not snowflake_ok:
            print("  - Fix Snowflake connection issues")
        if not mongodb_ok:
            print("  - Fix MongoDB connection issues")
        print("\nPlease check your credentials and try again")
        sys.exit(1)


if __name__ == "__main__":
    main()