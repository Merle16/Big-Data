"""
This script is used to clean the data using DuckDB.
For all, use DuckDB.

Clean both data/raw/csv/* and data/raw/IMDB_external_csv/*

1. Missingness detection: Convert disguised missing tokens to NaN with entropy detection.
-> lisa/vanshita

2. Assert accurate data types (ensure numeric columns are numeric, etc.).
-> merle

3. String & Categorical Standardization -> NFKD Unicode normalization
-> lisa/ilesh

4. Remove duplicate rows after normalization based on UUID. 
Movies with the same name but different start years are always different movies.
-> lisa/vanshita

5. Outlier detection & treatment (IQR capping) (ask ilesh)
-> ???? --> use log transformation in feature engineering instead! no IQR in pipeline.

6. MICE imputation for missing values.
-> lisa

7. safe all cleaned data to data/processed/csv/
--> now it will be ready for feature engineering

"""
