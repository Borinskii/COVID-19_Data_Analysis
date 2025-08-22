

--Cases per million in 2021 (we use table JHU_COVID_19 and Global Population Kaggle dataset (you can find SQL queries in repository):
WITH j AS (
  SELECT
    COUNTRY_REGION,
    SUM(IFF(CASE_TYPE='Confirmed', DIFFERENCE, 0)) AS confirmed_2021
  FROM COVID19_EPIDEMIOLOGICAL_DATA.PUBLIC.JHU_COVID_19
  WHERE DATE BETWEEN '2021-01-01' AND '2021-12-31'
    AND CASE_TYPE = 'Confirmed'
  GROUP BY COUNTRY_REGION
),
p AS (
  SELECT "Country", "2021" AS population_2021
  FROM COVID_ENRICHED.RAW.RAW_POPULATION_STG
)
SELECT
  j.COUNTRY_REGION,
  p.population_2021,
  j.confirmed_2021,
  ROUND(j.confirmed_2021 / NULLIF(p.population_2021,0) * 1e6, 2) AS cases_per_million_2021
FROM j
LEFT JOIN p
  ON UPPER(p."Country") = UPPER(j.COUNTRY_REGION)
ORDER BY cases_per_million_2021 DESC


--Vaccination Progress vs GDP per Capita in 2021
WITH vacc AS (
  SELECT
    v.COUNTRY_REGION,
    MAX(v.TOTAL_VACCINATIONS_PER_HUNDRED) AS max_vacc_per_100_2021
  FROM COVID19_EPIDEMIOLOGICAL_DATA.PUBLIC.OWID_VACCINATIONS v
  WHERE YEAR(v.DATE) = 2021
  GROUP BY v.COUNTRY_REGION
)
SELECT
  vacc.COUNTRY_REGION,
  g."2021" AS gdp_per_capita_2021_usd,
  vacc.max_vacc_per_100_2021
FROM vacc
LEFT JOIN COVID_ENRICHED.RAW.RAW_GDP_PER_CAPITA_STG g
  ON UPPER(TRIM(g.$1)) = UPPER(TRIM(vacc.COUNTRY_REGION))   -- $1 = first column (country)
ORDER BY max_vacc_per_100_2021 DESC


