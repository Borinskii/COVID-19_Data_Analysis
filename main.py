from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime, date, timezone
from enum import Enum
import snowflake.connector
from pymongo import MongoClient, DESCENDING
from bson import ObjectId
import os
from dotenv import load_dotenv
import logging
from contextlib import contextmanager
import hashlib
import json
import time

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
    SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
    SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
    SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
    SNOWFLAKE_DATABASE_COVID = "COVID19_EPIDEMIOLOGICAL_DATA"
    SNOWFLAKE_DATABASE_ENRICHED = "COVID_ENRICHED"
    SNOWFLAKE_SCHEMA = "PUBLIC"

    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    MONGODB_DATABASE = "COVID_COMMENTS"

    API_VERSION = "1.0.0"
    API_TITLE = "COVID-19 Data Analytics API"

    ALLOWED_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:8050",
        "http://localhost:8080"
    ]

    CACHE_TTL = 300  # 5 minutes
    ENABLE_CACHE = True

    @classmethod
    def validate(cls):
        required_vars = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"]
        missing = [var for var in required_vars if not getattr(cls, var)]
        if missing:
            logger.error(f"Missing required environment variables: {missing}")
            raise ValueError(f"Missing required environment variables: {missing}")

config = Config()
config.validate()

class CommentCreate(BaseModel):
    datapointId: str = Field(..., description='DATASET|COUNTRY_CODE|YYYY-MM-DD|FEATURE')
    type: Literal["comment", "correction", "quality_flag", "tag"] = "comment"
    text: Optional[str] = None
    labels: List[str] = Field(default_factory=list)
    sourceIds: List[str] = Field(default_factory=list)
    status: Literal["active", "resolved", "hidden"] = "active"
    attachments: List[Dict[str, Any]] = Field(default_factory=list)
    authorEmail: EmailStr

class CommentOut(BaseModel):
    id: str
    datapointId: str
    type: str
    text: Optional[str]
    labels: List[str]
    sourceIds: List[str]
    status: str
    attachments: List[Dict[str, Any]]
    authorId: str
    createdAt: datetime
    updatedAt: Optional[datetime] = None
    version: int = 1

app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["*"],
)

cache_store = {}

def get_cache_key(endpoint: str, params: dict) -> str:
    params_str = json.dumps(params, sort_keys=True, default=str)
    key = f"{endpoint}:{params_str}"
    return hashlib.md5(key.encode()).hexdigest()

def get_cached_result(cache_key: str) -> Optional[Dict[str, Any]]:
    if not config.ENABLE_CACHE:
        return None

    if cache_key in cache_store:
        cached = cache_store[cache_key]
        if time.time() - cached['timestamp'] < config.CACHE_TTL:
            logger.info(f"Cache hit for key: {cache_key}")
            return cached['data']
        else:
            del cache_store[cache_key]
    return None

def set_cached_result(cache_key: str, data: Dict[str, Any]):
    if config.ENABLE_CACHE:
        cache_store[cache_key] = {
            'data': data,
            'timestamp': time.time()
        }
        logger.info(f"Cached result for key: {cache_key}")

@contextmanager
def get_snowflake_connection(database: str):
    conn = None
    try:
        conn = snowflake.connector.connect(
            account=config.SNOWFLAKE_ACCOUNT,
            user=config.SNOWFLAKE_USER,
            password=config.SNOWFLAKE_PASSWORD,
            warehouse=config.SNOWFLAKE_WAREHOUSE,
            database=database,
            schema=config.SNOWFLAKE_SCHEMA,
            client_session_keep_alive=True,
            connection_timeout=30,
            network_timeout=60,
        )
        yield conn
    except Exception as e:
        logger.error(f"Snowflake connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")
    finally:
        if conn:
            conn.close()

def get_mongodb_client():
    try:
        client = MongoClient(config.MONGODB_URI, serverSelectionTimeoutMS=5000)
        db = client[config.MONGODB_DATABASE]
        db.command("ping")
        return db
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise HTTPException(status_code=500, detail="MongoDB connection failed")

def get_db():
    db = get_mongodb_client()
    try:
        yield db
    finally:
        pass

def _oid_str(v) -> str:
    try:
        return str(v)
    except Exception:
        return str(v)

def _ensure_user_by_email(db, email: str):
    return db.users.find_one({"email": email})

@app.get("/", tags=["General"])
async def root():
    return {
        "message": "COVID-19 Data Analytics API",
        "version": config.API_VERSION,
        "status": "healthy",
        "endpoints": {
            "documentation": "/docs",
            "health_check": "/health",
            "covid_cases": "/api/v1/covid-cases",
            "vaccinations": "/api/v1/vaccinations",
            "mobility": "/api/v1/mobility",
            "testing": "/api/v1/testing",
            "icu_beds": "/api/v1/icu-beds",
            "airline_restrictions": "/api/v1/airline-restrictions",
            "enriched_data": "/api/v1/enriched-data",
            "countries_list": "/api/v1/countries",
            "us_states_list": "/api/v1/us-states",
            "cache_status": "/api/v1/cache/status",
            "cache_clear": "/api/v1/cache/clear (POST)",
            "comments_create": "POST /api/v1/comments",
            "comments_list": "GET /api/v1/comments",
            "comments_get": "GET /api/v1/comments/{comment_id}",
        },
        "tables_used": [
            "JHU_COVID_19",
            "OWID_VACCINATIONS",
            "APPLE_MOBILITY",
            "CDC_TESTING",
            "CDC_INPATIENT_BEDS_ICU_ALL",
            "HUM_RESTRICTIONS_AIRLINE"
        ]
    }

@app.get("/health", tags=["General"])
async def health_check():
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": config.API_VERSION,
        "services": {},
        "cache": {
            "enabled": config.ENABLE_CACHE,
            "items": len(cache_store)
        }
    }

    try:
        with get_snowflake_connection(config.SNOWFLAKE_DATABASE_COVID) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            health_status["services"]["snowflake"] = "connected"
    except:
        health_status["services"]["snowflake"] = "disconnected"
        health_status["status"] = "degraded"

    try:
        db = get_mongodb_client()
        db.command("ping")
        health_status["services"]["mongodb"] = "connected"
    except:
        health_status["services"]["mongodb"] = "disconnected"

    return health_status

@app.get("/api/v1/covid-cases", tags=["COVID Data"])
async def get_covid_cases(
    country: Optional[str] = Query(None, description="Country name"),
    start_date: Optional[date] = Query(None, description="Start date"),
    end_date: Optional[date] = Query(None, description="End date"),
    case_type: Optional[str] = Query(None, description="Case type: Confirmed, Deaths, Recovered"),
    limit: int = Query(default=1000, le=10000, ge=1)
):
    cache_params = {
        'country': country,
        'start_date': start_date,
        'end_date': end_date,
        'case_type': case_type,
        'limit': limit
    }
    cache_key = get_cache_key('covid-cases', cache_params)

    cached_result = get_cached_result(cache_key)
    if cached_result:
        return cached_result

    try:
        with get_snowflake_connection(config.SNOWFLAKE_DATABASE_COVID) as conn:
            cursor = conn.cursor()

            sql = """
                SELECT COUNTRY_REGION, PROVINCE_STATE, DATE, CASE_TYPE, 
                       CASES, DIFFERENCE, LAT, LONG, ISO3166_1
                FROM MY_DB.PUBLIC.JHU_COVID_19_OPTIMIZED
                WHERE 1=1
            """
            params = []

            if country:
                sql += " AND UPPER(COUNTRY_REGION) = UPPER(%s)"
                params.append(country)

            if start_date:
                sql += " AND DATE >= %s"
                params.append(start_date.isoformat())

            if end_date:
                sql += " AND DATE <= %s"
                params.append(end_date.isoformat())

            if case_type:
                sql += " AND UPPER(CASE_TYPE) = UPPER(%s)"
                params.append(case_type)

            sql += " ORDER BY DATE DESC LIMIT %s"
            params.append(limit)

            cursor.execute(sql, params)
            columns = [col[0] for col in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]

            response = {
                "success": True,
                "count": len(results),
                "data": results
            }

            set_cached_result(cache_key, response)
            return response

    except Exception as e:
        logger.error(f"Error querying COVID cases: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/vaccinations", tags=["COVID Data"])
async def get_vaccinations(
    country: Optional[str] = Query(None),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    min_vaccination_rate: Optional[float] = Query(None, ge=0, le=100),
    limit: int = Query(default=1000, le=10000, ge=1)
):
    cache_params = {
        'country': country,
        'start_date': start_date,
        'end_date': end_date,
        'min_vaccination_rate': min_vaccination_rate,
        'limit': limit
    }
    cache_key = get_cache_key('vaccinations', cache_params)

    cached_result = get_cached_result(cache_key)
    if cached_result:
        return cached_result

    try:
        with get_snowflake_connection(config.SNOWFLAKE_DATABASE_COVID) as conn:
            cursor = conn.cursor()

            sql = """
                SELECT DATE, COUNTRY_REGION, ISO3166_1, TOTAL_VACCINATIONS, 
                       PEOPLE_VACCINATED, PEOPLE_FULLY_VACCINATED, DAILY_VACCINATIONS, 
                       PEOPLE_VACCINATED_PER_HUNDRED, PEOPLE_FULLY_VACCINATED_PER_HUNDRED, VACCINES
                FROM MY_DB.PUBLIC.OWID_VACCINATIONS_OPTIMIZED
                WHERE 1=1
            """
            params = []

            if country:
                sql += " AND UPPER(COUNTRY_REGION) = UPPER(%s)"
                params.append(country)

            if start_date:
                sql += " AND DATE >= %s"
                params.append(start_date.isoformat())

            if end_date:
                sql += " AND DATE <= %s"
                params.append(end_date.isoformat())

            if min_vaccination_rate:
                sql += " AND PEOPLE_VACCINATED_PER_HUNDRED >= %s"
                params.append(min_vaccination_rate)

            sql += " ORDER BY DATE DESC LIMIT %s"
            params.append(limit)

            cursor.execute(sql, params)
            columns = [col[0] for col in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]

            response = {
                "success": True,
                "count": len(results),
                "data": results
            }

            set_cached_result(cache_key, response)
            return response

    except Exception as e:
        logger.error(f"Error querying vaccinations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/mobility", tags=["COVID Data"])
async def get_mobility(
    country: Optional[str] = Query(None),
    transportation_type: Optional[str] = Query(None, description="driving, walking, or transit"),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    limit: int = Query(default=1000, le=10000, ge=1)
):
    cache_params = {
        'country': country,
        'transportation_type': transportation_type,
        'start_date': start_date,
        'end_date': end_date,
        'limit': limit
    }
    cache_key = get_cache_key('mobility', cache_params)

    cached_result = get_cached_result(cache_key)
    if cached_result:
        return cached_result

    try:
        with get_snowflake_connection(config.SNOWFLAKE_DATABASE_COVID) as conn:
            cursor = conn.cursor()

            sql = """
                SELECT COUNTRY_REGION, PROVINCE_STATE, DATE, 
                       TRANSPORTATION_TYPE, DIFFERENCE, ISO3166_1
                FROM MY_DB.PUBLIC.APPLE_MOBILITY_OPTIMIZED
                WHERE 1=1
            """
            params = []

            if country:
                sql += " AND UPPER(COUNTRY_REGION) = UPPER(%s)"
                params.append(country)

            if transportation_type:
                sql += " AND UPPER(TRANSPORTATION_TYPE) = UPPER(%s)"
                params.append(transportation_type)

            if start_date:
                sql += " AND DATE >= %s"
                params.append(start_date.isoformat())

            if end_date:
                sql += " AND DATE <= %s"
                params.append(end_date.isoformat())

            sql += " ORDER BY DATE DESC LIMIT %s"
            params.append(limit)

            cursor.execute(sql, params)
            columns = [col[0] for col in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]

            response = {
                "success": True,
                "count": len(results),
                "data": results
            }

            set_cached_result(cache_key, response)
            return response

    except Exception as e:
        logger.error(f"Error querying mobility: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/enriched-data", tags=["Enriched Data"])
async def get_enriched_data(
    country: Optional[str] = Query(None),
    year: Optional[int] = Query(None, ge=2015, le=2024),
    include_gdp: bool = Query(True),
    include_hdi: bool = Query(True),
    include_population: bool = Query(True)
):
    cache_params = {
        'country': country,
        'year': year,
        'include_gdp': include_gdp,
        'include_hdi': include_hdi,
        'include_population': include_population
    }
    cache_key = get_cache_key('enriched', cache_params)

    cached_result = get_cached_result(cache_key)
    if cached_result:
        return cached_result

    try:
        with get_snowflake_connection(config.SNOWFLAKE_DATABASE_ENRICHED) as conn:
            cursor = conn.cursor()
            results = {}

            if include_population and year:
                try:
                    year_str = str(year)
                    sql = f'SELECT "Country", "{year_str}" as POPULATION FROM RAW.RAW_POPULATION_STG WHERE "{year_str}" IS NOT NULL'
                    params = []

                    if country:
                        sql += ' AND UPPER("Country") = UPPER(%s)'
                        params.append(country)

                    cursor.execute(sql, params)
                    pop_data = cursor.fetchall()

                    results['population'] = [
                        {"country": row[0], "population": int(row[1])}
                        for row in pop_data
                        if row[0] and row[1] is not None
                    ]
                except Exception as e:
                    logger.error(f"Population query failed: {e}")
                    results['population'] = []

            if include_gdp and year:
                try:
                    year_str = str(year)
                    gdp_col = 'GDP per capita, current prices\n (U.S. dollars per capita)'
                    sql = f'SELECT "{gdp_col}", "{year_str}" FROM RAW.RAW_GDP_PER_CAPITA_STG WHERE "{year_str}" IS NOT NULL'
                    params = []

                    if country:
                        sql += f' AND UPPER("{gdp_col}") = UPPER(%s)'
                        params.append(country)

                    cursor.execute(sql, params)
                    gdp_data = cursor.fetchall()

                    results['gdp'] = []
                    for row in gdp_data:
                        if row[0] and row[1] and row[1] != 'no data':
                            try:
                                results['gdp'].append({
                                    "country": row[0],
                                    "gdp_per_capita": float(row[1])
                                })
                            except:
                                continue
                except Exception as e:
                    logger.error(f"GDP query failed: {e}")
                    results['gdp'] = []

            if include_hdi and year:
                try:
                    sql_hdi = 'SELECT country, hdi FROM RAW.RAW_HDI_STG WHERE year = %s AND hdi IS NOT NULL'
                    params = [year]

                    if country:
                        sql_hdi += ' AND UPPER(country) = UPPER(%s)'
                        params.append(country)

                    cursor.execute(sql_hdi, params)
                    hdi_data = cursor.fetchall()

                    results['hdi'] = [
                        {"country": row[0], "hdi": float(row[1])}
                        for row in hdi_data
                        if row[0] and row[1] is not None
                    ]
                except Exception as e:
                    logger.error(f"HDI query failed: {e}")
                    results['hdi'] = []

            response = {
                "success": True,
                "data": results
            }

            set_cached_result(cache_key, response)
            return response

    except Exception as e:
        logger.error(f"Error querying enriched data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/cache/clear", tags=["Utility"])
async def clear_cache():
    cache_store.clear()
    return {"success": True, "message": "Cache cleared", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/api/v1/cache/status", tags=["Utility"])
async def cache_status():
    return {
        "enabled": config.ENABLE_CACHE,
        "ttl_seconds": config.CACHE_TTL,
        "cached_items": len(cache_store),
        "cache_keys": list(cache_store.keys())[:10]
    }

@app.get("/api/v1/testing", tags=["COVID Data"])
async def get_testing(
    country: Optional[str] = Query(None, description="ISO3166_1 country code"),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    limit: int = Query(default=1000, le=10000, ge=1)
):
    cache_params = {
        'country': country,
        'start_date': start_date,
        'end_date': end_date,
        'limit': limit
    }
    cache_key = get_cache_key('testing', cache_params)

    cached_result = get_cached_result(cache_key)
    if cached_result:
        return cached_result

    try:
        with get_snowflake_connection(config.SNOWFLAKE_DATABASE_COVID) as conn:
            cursor = conn.cursor()

            sql = """
                SELECT ISO3166_1, ISO3166_2, DATE, 
                       POSITIVE, NEGATIVE, INCONCLUSIVE
                FROM CDC_TESTING
                WHERE 1=1
            """
            params = []

            if country:
                sql += " AND UPPER(ISO3166_1) = UPPER(%s)"
                params.append(country)

            if start_date:
                sql += " AND DATE >= %s"
                params.append(start_date.isoformat())

            if end_date:
                sql += " AND DATE <= %s"
                params.append(end_date.isoformat())

            sql += " ORDER BY DATE DESC LIMIT %s"
            params.append(limit)

            cursor.execute(sql, params)
            columns = [col[0] for col in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]

            response = {
                "success": True,
                "count": len(results),
                "data": results
            }

            set_cached_result(cache_key, response)
            return response

    except Exception as e:
        logger.error(f"Error querying testing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/icu-beds", tags=["COVID Data"])
async def get_icu_beds(
    state: Optional[str] = Query(None, description="US State"),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    limit: int = Query(default=1000, le=10000, ge=1)
):
    cache_params = {
        'state': state,
        'start_date': start_date,
        'end_date': end_date,
        'limit': limit
    }
    cache_key = get_cache_key('icu-beds', cache_params)

    cached_result = get_cached_result(cache_key)
    if cached_result:
        return cached_result

    try:
        with get_snowflake_connection(config.SNOWFLAKE_DATABASE_COVID) as conn:
            cursor = conn.cursor()

            sql = """
                SELECT STATE, DATE, ISO3166_1, ISO3166_2,
           STAFFED_ADULT_ICU_BEDS_OCCUPIED, 
           STAFFED_ADULT_ICU_BEDS_OCCUPIED_LOWER_BOUND,
           STAFFED_ADULT_ICU_BEDS_OCCUPIED_UPPER_BOUND,
           STAFFED_ADULT_ICU_BEDS_OCCUPIED_PCT,
           STAFFED_ADULT_ICU_BEDS_OCCUPIED_PCT_LOWER_BOUND,
           STAFFED_ADULT_ICU_BEDS_OCCUPIED_PCT_UPPER_BOUND,
           TOTAL_STAFFED_ICU_BEDS,
           TOTAL_STAFFED_ICU_BEDS_LOWER_BOUND,
           TOTAL_STAFFED_ICU_BEDS_UPPER_BOUND,
           LAST_REPORTED_FLAG
    FROM CDC_INPATIENT_BEDS_ICU_ALL
    WHERE 1=1
            """
            params = []

            if state:
                sql += " AND UPPER(STATE) = UPPER(%s)"
                params.append(state)

            if start_date:
                sql += " AND DATE >= %s"
                params.append(start_date.isoformat())

            if end_date:
                sql += " AND DATE <= %s"
                params.append(end_date.isoformat())

            sql += " ORDER BY DATE DESC LIMIT %s"
            params.append(limit)

            cursor.execute(sql, params)
            columns = [col[0] for col in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]

            response = {
                "success": True,
                "count": len(results),
                "data": results
            }

            set_cached_result(cache_key, response)
            return response

    except Exception as e:
        logger.error(f"Error querying ICU beds data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/airline-restrictions", tags=["COVID Data"])
async def get_airline_restrictions(
    country: Optional[str] = Query(None, description="Country name"),
    airline: Optional[str] = Query(None, description="Airline name"),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    limit: int = Query(default=1000, le=10000, ge=1)
):
    cache_params = {
        'country': country,
        'airline': airline,
        'start_date': start_date,
        'end_date': end_date,
        'limit': limit
    }
    cache_key = get_cache_key('airline-restrictions', cache_params)

    cached_result = get_cached_result(cache_key)
    if cached_result:
        return cached_result

    try:
        with get_snowflake_connection(config.SNOWFLAKE_DATABASE_COVID) as conn:
            cursor = conn.cursor()

            sql = """
                SELECT COUNTRY, ISO3166_1, LAT, LONG, 
                       PUBLISHED, SOURCES, AIRLINE, 
                       RESTRICTION_TEXT, LAST_UPDATE_DATE
                FROM HUM_RESTRICTIONS_AIRLINE
                WHERE 1=1
            """
            params = []

            if country:
                sql += " AND UPPER(COUNTRY) = UPPER(%s)"
                params.append(country)

            if airline:
                sql += " AND UPPER(AIRLINE) = UPPER(%s)"
                params.append(airline)

            if start_date:
                sql += " AND PUBLISHED >= %s"
                params.append(start_date.isoformat())

            if end_date:
                sql += " AND PUBLISHED <= %s"
                params.append(end_date.isoformat())

            sql += " ORDER BY PUBLISHED DESC LIMIT %s"
            params.append(limit)

            cursor.execute(sql, params)
            columns = [col[0] for col in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]

            response = {
                "success": True,
                "count": len(results),
                "data": results
            }

            set_cached_result(cache_key, response)
            return response

    except Exception as e:
        logger.error(f"Error querying airline restrictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/countries", tags=["COVID Data"])
async def get_countries():
    cache_key = get_cache_key('countries', {})
    cached_result = get_cached_result(cache_key)
    if cached_result:
        return cached_result

    try:
        with get_snowflake_connection(config.SNOWFLAKE_DATABASE_COVID) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT COUNTRY_REGION 
                FROM MY_DB.PUBLIC.JHU_COVID_19_OPTIMIZED 
                WHERE COUNTRY_REGION IS NOT NULL
                ORDER BY COUNTRY_REGION
            """)
            countries = [row[0] for row in cursor.fetchall()]

            response = {
                "success": True,
                "count": len(countries),
                "data": countries
            }

            # Cache for longer (1 hour)
            cache_store[cache_key] = {
                'data': response,
                'timestamp': time.time() - config.CACHE_TTL + 3600
            }

            return response

    except Exception as e:
        logger.error(f"Error fetching countries: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/us-states", tags=["COVID Data"])
async def get_us_states():
    cache_key = get_cache_key('us-states', {})
    cached_result = get_cached_result(cache_key)
    if cached_result:
        return cached_result

    try:
        with get_snowflake_connection(config.SNOWFLAKE_DATABASE_COVID) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT STATE 
                FROM CDC_INPATIENT_BEDS_ICU_ALL 
                WHERE STATE IS NOT NULL
                ORDER BY STATE
            """)
            states = [row[0] for row in cursor.fetchall()]

            response = {
                "success": True,
                "count": len(states),
                "data": states
            }

            # Cache for longer (1 hour)
            cache_store[cache_key] = {
                'data': response,
                'timestamp': time.time() - config.CACHE_TTL + 3600
            }

            return response

    except Exception as e:
        logger.error(f"Error fetching US states: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/comments", tags=["Comments"], response_model=CommentOut)
async def create_comment(payload: CommentCreate, db=Depends(get_db)):
    # validate author
    user = _ensure_user_by_email(db, payload.authorEmail)
    if user is None:
        raise HTTPException(status_code=403, detail="User not found or not authorized")

    # normalize labels
    labels = []
    seen = set()
    for l in (payload.labels or []):
        s = (l or "").strip()
        if s and s.lower() not in seen:
            labels.append(s)
            seen.add(s.lower())

    now = datetime.now(timezone.utc)

    doc = {
        "datapointId": payload.datapointId,
        "type": payload.type,
        "text": (payload.text or "").strip() or None,
        "labels": labels,
        "authorId": user["_id"],
        "sourceIds": payload.sourceIds or [],
        "status": payload.status,
        "attachments": payload.attachments or [],
        "createdAt": now,
        "updatedAt": None,
        "version": 1,
    }

    res = db.annotations.insert_one(doc)
    out = {
        "id": str(res.inserted_id),
        "datapointId": doc["datapointId"],
        "type": doc["type"],
        "text": doc["text"],
        "labels": doc["labels"],
        "sourceIds": doc["sourceIds"],
        "status": doc["status"],
        "attachments": doc["attachments"],
        "authorId": _oid_str(doc["authorId"]),
        "createdAt": doc["createdAt"],
        "updatedAt": doc["updatedAt"],
        "version": doc["version"],
    }
    return out

@app.get("/api/v1/comments", tags=["Comments"])
async def list_comments(
    datapointId: Optional[str] = Query(None),
    authorEmail: Optional[EmailStr] = Query(None),
    label: Optional[str] = Query(None, description="Filter by a single label"),
    limit: int = Query(50, ge=1, le=200),
    db=Depends(get_db),
):
    q: Dict[str, Any] = {}
    if datapointId:
        q["datapointId"] = datapointId
    if label:
        q["labels"] = label
    if authorEmail:
        user = _ensure_user_by_email(db, authorEmail)
        if user is None:
            return {"success": True, "count": 0, "data": []}
        q["authorId"] = user["_id"]

    cur = db.annotations.find(q).sort("createdAt", DESCENDING).limit(int(limit))
    items = []
    for a in cur:
        items.append({
            "id": _oid_str(a.get("_id")),
            "datapointId": a.get("datapointId"),
            "type": a.get("type"),
            "text": a.get("text"),
            "labels": a.get("labels", []),
            "sourceIds": a.get("sourceIds", []),
            "status": a.get("status", "active"),
            "attachments": a.get("attachments", []),
            "authorId": _oid_str(a.get("authorId")),
            "createdAt": a.get("createdAt"),
            "updatedAt": a.get("updatedAt"),
            "version": a.get("version", 1),
        })

    return {"success": True, "count": len(items), "data": items}

@app.get("/api/v1/comments/{comment_id}", tags=["Comments"], response_model=CommentOut)
async def get_comment(comment_id: str, db=Depends(get_db)):
    try:
        _id = ObjectId(comment_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid comment id")

    a = db.annotations.find_one({"_id": _id})
    if a is None:
        raise HTTPException(status_code=404, detail="Comment not found")

    return CommentOut(
        id=str(a["_id"]),
        datapointId=a.get("datapointId"),
        type=a.get("type"),
        text=a.get("text"),
        labels=a.get("labels", []),
        sourceIds=a.get("sourceIds", []),
        status=a.get("status", "active"),
        attachments=a.get("attachments", []),
        authorId=_oid_str(a.get("authorId")),
        createdAt=a.get("createdAt"),
        updatedAt=a.get("updatedAt"),
        version=a.get("version", 1),
    )

@app.on_event("startup")
async def startup_event():
    logger.info("Starting API and preloading cache...")

    try:
        # Preload countries list
        await get_countries()
        logger.info("Preloaded countries list")

        # Clear old cache entries if any
        current_time = time.time()
        expired_keys = [
            key for key, value in cache_store.items()
            if current_time - value['timestamp'] > config.CACHE_TTL
        ]
        for key in expired_keys:
            del cache_store[key]

        logger.info(f"API startup complete. Cache has {len(cache_store)} items.")

    except Exception as e:
        logger.error(f"Error during startup: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )