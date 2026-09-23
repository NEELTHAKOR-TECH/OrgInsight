import duckdb

path = 'data/processed/organizations_clean.csv'
con = duckdb.connect()
sql = 'CREATE TABLE orgs AS SELECT * FROM read_csv_auto(' + chr(39) + path + chr(39) + ')'
con.execute(sql)

print('=== SCHEMA ===')
print(con.execute('DESCRIBE orgs').df().to_string())

print('\n=== TOTALS ===')
print(con.execute('SELECT COUNT(*) AS n, COUNT(DISTINCT Industry) AS ind, COUNT(DISTINCT Country) AS cty FROM orgs').df().to_string())

print('\n=== FOUNDED RANGE ===')
print(con.execute('SELECT MIN(Founded) AS min_yr, MAX(Founded) AS max_yr, COUNT(DISTINCT Founded) AS yrs FROM orgs').df().to_string())

print('\n=== AVG COMPANY_AGE ===')
print(con.execute('SELECT ROUND(AVG(company_age),2) AS avg_age, MIN(company_age) AS min_age, MAX(company_age) AS max_age FROM orgs').df().to_string())

print('\n=== COUNTRY COUNT RANGE ===')
print(con.execute('SELECT MIN(c) AS min_cnt, MAX(c) AS max_cnt, ROUND(AVG(c),1) AS avg_cnt FROM (SELECT Country, COUNT(*) AS c FROM orgs GROUP BY Country)').df().to_string())

print('\n=== SIZE BAND COUNTS ===')
q = 'SELECT size_band, COUNT(*) AS n FROM orgs GROUP BY size_band ORDER BY size_band'
print(con.execute(q).df().to_string())

print('\n=== SECTOR COUNTS ===')
q2 = 'SELECT broad_sector, COUNT(*) AS n FROM orgs GROUP BY broad_sector ORDER BY n DESC'
print(con.execute(q2).df().to_string())

print('\n=== DUP FLAG ===')
q3 = 'SELECT is_name_country_dup, COUNT(*) AS n FROM orgs GROUP BY is_name_country_dup'
print(con.execute(q3).df().to_string())

print('\n=== NULL CHECK KEY COLS ===')
q4 = '''SELECT
  SUM(CASE WHEN Industry IS NULL THEN 1 ELSE 0 END) AS null_industry,
  SUM(CASE WHEN Country IS NULL THEN 1 ELSE 0 END) AS null_country,
  SUM(CASE WHEN "Number of employees" IS NULL THEN 1 ELSE 0 END) AS null_employees,
  SUM(CASE WHEN Founded IS NULL THEN 1 ELSE 0 END) AS null_founded,
  SUM(CASE WHEN size_band IS NULL THEN 1 ELSE 0 END) AS null_size_band,
  SUM(CASE WHEN broad_sector IS NULL THEN 1 ELSE 0 END) AS null_sector
FROM orgs'''
print(con.execute(q4).df().to_string())

print('\n=== Q8 LABEL CHECK (founding years 2000+) ===')
q5 = 'SELECT MIN(Founded) AS min_founded_2000plus, MAX(Founded) AS max_founded FROM orgs WHERE Founded >= 2000'
print(con.execute(q5).df().to_string())

print('\n=== Q6 POTENTIAL ISSUE: top industry per country uses ROW_NUMBER before GROUP ===')
# Verify: does one country really have a clear top industry?
q6 = "SELECT Country, Industry, COUNT(*) AS n FROM orgs WHERE Country='Congo' GROUP BY Country, Industry ORDER BY n DESC LIMIT 5"
print(con.execute(q6).df().to_string())

con.close()
