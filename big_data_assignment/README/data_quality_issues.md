## Data quality findings (IMDB subset)

This document summarizes the main data quality checks and issues found in the movie subset used for the project.

### Completeness

- **Missing data (explicit NaN + disguised `\N`)**
  - Before treating `\N` as missing, the **missing ratio per column** was:
    - `originalTitle`: 0.501068
    - `numVotes`: 0.099259
  - **Disguised missing values** (special token `\N`) detected:
    - `startYear`: 786 rows with `\N`
    - `endYear`: 7173 rows with `\N`
    - `runtimeMinutes`: 13 rows with `\N`
  - After imputing `\N` as `NaN`, **missing counts per column**:
    - `originalTitle`: 3988
    - `startYear`: 786
    - `endYear`: 7173
    - `runtimeMinutes`: 13
    - `numVotes`: 790
  - After imputing `\N` as `NaN`, **missing ratios per column**:
    - `originalTitle`: 0.501068
    - `startYear`: 0.098756
    - `endYear`: 0.901244
    - `runtimeMinutes`: 0.001633
    - `numVotes`: 0.099259

### Validity (types, ranges, domains)

- **Datatype checks**
  - Initially, `startYear`, `endYear`, `runtimeMinutes`, and `numVotes` are loaded as object/string.
  - After replacing `\N` with `NaN`, these columns can be safely converted back to numeric types.

- **Range and outlier checks (numeric columns)**
  - **`startYear`**:
    - Range: min = 1918.0, max = 2021.0, range = 103.0.
    - IQR approximately \[1949.00, 2053.00\].
    - MAD-based outlier detection flags several years between 1919 and 1936, but these values are still plausible historically, so we keep them for analysis.
  - **`endYear`**:
    - Range: min = 1921.0, max = 2021.0, range = 100.0.
    - No titles with an `endYear` beyond 2026, so there are no obviously impossible future values.
  - **`runtimeMinutes`**:
    - Range: 45 to 551 minutes.
    - Long runtimes are rare but still within a plausible domain (e.g. mini-series, extended cuts).
  - **`numVotes`**:
    - Range: 1001 to 2,503,641 (needs a final double-check).
    - The distribution is heavy-tailed but consistent with popular titles attracting many votes.

### Relationships between `primaryTitle` and `originalTitle`

- **Row count**: 7,959 rows.
- **Missingness and duplicates**
  - `primaryTitle` has **no missing values**.
  - `originalTitle` is missing in **50.11%** of rows.
  - `originalTitle` has **1.33%** duplicate values.
  - `primaryTitle` has **2.74%** duplicate values.
- **Equivalence between titles**
  - About **32% of rows** share the same value for `primaryTitle` and `originalTitle`.
  - This suggests that for many rows, the “original” and localized/primary title are identical, while for others they differ (translations, alternative titles, etc.).

### Data entry errors and potential typos

- There are suspected **data entry inconsistencies** where the same movie appears with slightly different titles (or vice versa):
  - Example: titles such as “Hello” and “Hunger” have different metadata but might correspond to the same underlying movie in some cases.
  - Without an external identifier (e.g. IMDb ID) or richer metadata, it is difficult to robustly decide whether such entries should be merged or treated as distinct.

### Imputation considerations

- **Year columns (`startYear`, `endYear`)**
  - Idea: impute based on **distributional statistics** (e.g. sampling from observed year distributions), assuming a typical production duration between `startYear` and `endYear`.
  - However, there are **no rows where both `startYear` and `endYear` are simultaneously observed** for the same title in this subset, which makes duration-based imputation unreliable.

- **`runtimeMinutes` and `numVotes`**
  - For `runtimeMinutes`, a method like **multiple imputation (MICE)** could be considered, using other movie features as predictors.
  - For `numVotes`, MICE or a similar model-based imputation could also be reasonable, but care is needed:
    - `numVotes` is very skewed and highly informative (popularity), so imputation can change the label distribution and ranking signals.

### Open challenges and questions

- **Title normalization and deduplication**
  - Should we normalize and standardize names (`primaryTitle`, `originalTitle`) and try to merge rows that likely refer to the same movie?
  - If we decide to merge:
    - How do we handle **conflicting numeric attributes** like `runtimeMinutes`, `startYear`, `endYear`, and `numVotes`?
    - For example, if multiple rows share (approximately) the same title:
      - Do we sum `numVotes`, take a maximum, or keep the row with the most complete metadata?
      - How do we derive a consistent runtime and year span (min, max, or some aggregation)?

- **Impact on downstream modelling**
  - Aggressively imputing or deduplicating might:
    - Improve completeness and reduce noise.
    - But also risk introducing bias or collapsing genuinely distinct movies into one record.
  - These trade-offs should be made explicit before final feature engineering and model training.

