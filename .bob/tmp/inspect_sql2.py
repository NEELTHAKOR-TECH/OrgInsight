import duckdb

path = 'data/processed/organizations_clean.csv'
con = duckdb.connect()
sql = 'CREATE TABLE orgs AS SELECT * FROM read_csv_auto(' + chr(39) + path + chr(39) + ')'
con.execute(sql)

# Deep-check all existing query outputs for correctness

# Q1: check top industry count matches pandas
q1 = con.execute('''
    SELECT Industry, COUNT(*) AS org_count
    FROM orgs GROUP BY Industry ORDER BY org_count DESC LIMIT 3
''').df()
print('=== Q1 TOP 3 INDUSTRIES ===')
print(q1.to_string())

# Q2: top country
q2 = con.execute('''
    SELECT Country, COUNT(*) AS org_count
    FROM orgs GROUP BY Country ORDER BY org_count DESC LIMIT 3
''').df()
print('\n=== Q2 TOP 3 COUNTRIES ===')
print(q2.to_string())

# Q3: sector pct totals 100?
q3 = con.execute('''
    SELECT broad_sector, COUNT(*) AS n,
           ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER(),2) AS pct
    FROM orgs GROUP BY broad_sector ORDER BY n DESC
''').df()
print('\n=== Q3 SECTOR PCT SUM ===')
print('Sum pct:', round(q3['pct'].sum(), 2))
print(q3.to_string())

# Q4: size band ordering — the CASE sorts correctly?
q4 = con.execute('''
    SELECT size_band, COUNT(*) AS n,
           ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER(),2) AS pct,
           ROUND(AVG(company_age),1) AS avg_age
    FROM orgs GROUP BY size_band
    ORDER BY CASE size_band WHEN 'Micro' THEN 1 WHEN 'Small' THEN 2
             WHEN 'Medium' THEN 3 WHEN 'Large' THEN 4 WHEN 'Enterprise' THEN 5 END
''').df()
print('\n=== Q4 SIZE BAND CHECK ===')
print(q4.to_string())
print('Sum:', q4['n'].sum())

# Q5: decade counts sum to 100k?
q5 = con.execute('''
    SELECT founding_decade, COUNT(*) AS n
    FROM orgs WHERE founding_decade IS NOT NULL
    GROUP BY founding_decade ORDER BY founding_decade
''').df()
print('\n=== Q5 DECADE SUM ===', q5['n'].sum())
print(q5.to_string())

# Q6: ROW_NUMBER bug — does it actually pick the right top industry?
# The current query uses ROW_NUMBER OVER (PARTITION BY Country ORDER BY COUNT(*) DESC)
# inside a subquery that already grouped — this is correct
q6_check = con.execute('''
    WITH ranked AS (
        SELECT Country, Industry, COUNT(*) AS cnt,
               ROW_NUMBER() OVER (PARTITION BY Country ORDER BY COUNT(*) DESC) AS rn
        FROM orgs WHERE Country IN ('Congo','Korea','Lebanon')
        GROUP BY Country, Industry
    )
    SELECT Country, Industry, cnt FROM ranked WHERE rn=1
''').df()
print('\n=== Q6 TOP INDUSTRY CHECK (Congo/Korea/Lebanon) ===')
print(q6_check.to_string())

# Q7: Does HAVING COUNT(*) >= 100 filter correctly?
q7_min = con.execute('''
    SELECT MIN(org_count) AS min_count FROM (
        SELECT Industry, COUNT(*) AS org_count FROM orgs GROUP BY Industry HAVING COUNT(*) >= 100
    )
''').df()
print('\n=== Q7 MIN ORG COUNT AFTER FILTER ===')
print(q7_min.to_string())

# Q8: year range in output
q8 = con.execute('SELECT MIN(Founded) AS yr_min, MAX(Founded) AS yr_max, COUNT(*) AS n FROM orgs WHERE Founded >= 2000').df()
print('\n=== Q8 FOUNDED >= 2000 ===')
print(q8.to_string())

# Q10: all countries have >= 50?  (we know min is 351 so HAVING >= 50 is redundant)
q10_all = con.execute('SELECT COUNT(DISTINCT Country) AS total_countries FROM orgs').df()
q10_50 = con.execute('SELECT COUNT(*) AS countries_with_50plus FROM (SELECT Country FROM orgs GROUP BY Country HAVING COUNT(*) >= 50)').df()
print('\n=== Q10 HAVING >= 50 CHECK ===')
print('All countries:', q10_all.iloc[0,0])
print('Countries with >=50 orgs:', q10_50.iloc[0,0])
print('HAVING clause is redundant (all countries have >=351 orgs)')

# NEW: check what extra meaningful queries are missing
# -- Employee percentile distribution by SQL
print('\n=== EMPLOYEE PERCENTILES VIA SQL ===')
pct_q = con.execute('''
    SELECT
      PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY "Number of employees") AS p10,
      PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY "Number of employees") AS p25,
      PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY "Number of employees") AS p50,
      PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY "Number of employees") AS p75,
      PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY "Number of employees") AS p90
    FROM orgs
''').df()
print(pct_q.to_string())

# -- Year with most orgs founded (complete list)
print('\n=== TOP 5 FOUNDING YEARS ===')
print(con.execute('SELECT Founded, COUNT(*) AS n FROM orgs GROUP BY Founded ORDER BY n DESC LIMIT 5').df().to_string())

# -- Compare avg employees: dup flag=1 vs flag=0
print('\n=== DUP FLAG VS AVG EMPLOYEES ===')
print(con.execute('SELECT is_name_country_dup, COUNT(*) AS n, ROUND(AVG("Number of employees"),1) AS avg_emp FROM orgs GROUP BY is_name_country_dup ORDER BY is_name_country_dup').df().to_string())

con.close()
