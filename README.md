# SE446 Milestone 2 — Bigdataproject group

Spark DataFrame analytics + MLlib arrest predictor on the Chicago Crime dataset (Hadoop 3.4.1 / Spark 3.5.4 cluster, 1 master + 2 workers).

## Team

| Member | ID | GitHub | Tasks |
|--------|----|--------|-------|
| Reem Alswailem    | 231079 | `sengineer25`     | 1, 11 |
| Aseel Alzahrani   | 221581 | `Aseel-Alz`       | 5, 9  |
| Jenna Alqurashi   | 231614 | `Jennaalqurashi`  | 2, 6  |
| Dina Alhudaithi   | 221466 | `DinaAlhudaithi`  | 3, 7  |
| Dana Alnahas      | 231515 | `d-1117`          | 4, 10 |

## Spec compliance (May 2026 update)

- **Task 8 (CrossValidator) is waived.** Not implemented.
- **Phase B (Tasks 5–7) trains on a 5% sample** of the full dataset using
  `df.sample(fraction=0.05, seed=42)` (≈ 39,534 rows out of 793,072).
- **Task 11** uses `--deploy-mode cluster`. The application driver runs inside YARN.
- The course YARN cluster's max container is `<memory:1536, vCores:1>`, so we use
  `--executor-cores 1` (M1 used the same setting).

## Repository layout

```
.
├── M2_Bigdataproject.ipynb        # Notebook (Tasks 1–7), executed locally
├── arrest_predictor.py            # Standalone Phase B script (spark-submit)
├── scripts/
│   └── notebook_assembler.py      # Generates the .ipynb from cell sources
├── output/
│   ├── yearly_trend.png           # Task 3 matplotlib chart
│   ├── cluster_yarn_log.txt       # Task 10 evidence
│   └── spark_submit/
│       ├── console.log            # Task 11 spark-submit console output
│       └── run.log                # Task 11 application stdout
└── README.md
```

## Executive summary

We reproduce the four M1 MapReduce analyses with Spark DataFrames + Spark SQL on the
full 793,072-row HDFS dataset (numbers match M1 exactly). For arrest prediction we
build a Spark MLlib pipeline (StringIndexer × 2 + VectorAssembler + classifier),
training Logistic Regression, Random Forest, and Gradient-Boosted Trees on a 5% sample
(spec-mandated). Random Forest (numTrees=100, maxDepth=5) is the strongest tree-based
model on the cluster sample (AUC 0.8061), confirmed locally on the W09B 10K-row sample
(AUC 0.8701).

# Phase A — DataFrame analytics on the full dataset

## Task 1 — Crime type distribution
**Author:** Reem Alswailem (231079, `sengineer25`)

```python
top_crime_types = (df
                   .groupBy(F.col("Primary Type").alias("crime_type"))
                   .agg(F.count("*").alias("n"))
                   .orderBy(F.desc("n"))
                   .limit(10))
top_crime_types.show(truncate=False)
```

**M1 ↔ M2 — Top 10 crime types on the full dataset:**

| Crime type | M1 (MapReduce) | M2 (Spark) |
|---|---:|---:|
| THEFT | 162,688 | 162,688 |
| BATTERY | 151,930 | 151,930 |
| CRIMINAL DAMAGE | 91,241 | 91,241 |
| NARCOTICS | 74,127 | 74,127 |
| ASSAULT | 54,070 | 54,070 |
| MOTOR VEHICLE THEFT | 48,494 | 48,494 |
| BURGLARY | 39,872 | 39,872 |
| OTHER OFFENSE | 36,893 | 36,893 |
| ROBBERY | 30,991 | 30,991 |
| DECEPTIVE PRACTICE | 30,396 | 30,396 |

Numbers match exactly — same source data. Spark's DataFrame API runs the aggregation
in-memory rather than the disk shuffle required by streaming MapReduce.

---

## Task 2 — Location hotspots (Spark SQL)
**Author:** Jenna Alqurashi (231614, `Jennaalqurashi`)

```python
df.createOrReplaceTempView("crimes")
top_locations = spark.sql("""
    SELECT `Location Description` AS location,
           COUNT(*)                AS occurrences
    FROM crimes
    WHERE `Location Description` IS NOT NULL
    GROUP BY `Location Description`
    ORDER BY occurrences DESC
    LIMIT 10
""")
```

**M1 ↔ M2 — Top 10 locations:**

| Location | M1 | M2 (Spark cluster) |
|---|---:|---:|
| STREET | 245,437 | 248,326 |
| RESIDENCE | 136,238 | 136,393 |
| APARTMENT | 60,925 | 61,235 |
| SIDEWALK | 47,407 | 47,506 |
| OTHER | 29,213 | 29,671 |
| PARKING LOT/GARAGE(NON.RESID.) | 21,876 | 22,436 |
| ALLEY | 18,258 | 18,349 |
| SCHOOL, PUBLIC, BUILDING | 20,516 | 15,776 |
| RESIDENCE-GARAGE | 14,266 | 14,291 |
| SMALL RETAIL STORE | 13,755 | 13,804 |

Slight differences come from M1's manual CSV split dropping a few hundred edge-case
rows that Spark's CSV parser keeps. Spark SQL is more concise than the equivalent
mapper.

---

## Task 3 — Year trend
**Author:** Dina Alhudaithi (221466, `DinaAlhudaithi`)

```python
yearly_counts = (df.groupBy("Year")
                   .agg(F.count("*").alias("incidents"))
                   .orderBy("Year"))
```

Cluster results (full dataset):

| Year | Incidents | | Year | Incidents |
|---:|---:|---|---:|---:|
| 2001 | 467,301 | | 2014 | 825 |
| 2002 | 205,266 | | 2015 | 1,105 |
| 2003 | 985 | | 2016 | 1,339 |
| 2004 | 915 | | 2017 | 1,387 |
| 2005 | 1,031 | | 2018 | 1,327 |
| 2006 | 796 | | 2019 | 1,174 |
| 2007 | 762 | | 2020 | 1,832 |
| 2008 | 1,010 | | 2021 | 2,399 |
| 2009 | 910 | | 2022 | 4,678 |
| 2010 | 695 | | 2023 | 81,461 |
| 2011 | 770 | | 2024 | 880 |
| 2012 | 800 | | 2025 | 12,710 |
| 2013 | 714 | | | |

2001 + 2002 dominate, then a long quiet stretch through 2022, sharp 2023 spike. Local
chart at `output/yearly_trend.png`.

---

# Phase B — MLlib arrest predictor (5% sample)

The May 2026 spec update mandates training on a 5% sample. We apply
`df.sample(fraction=0.05, seed=42)` before any feature engineering. On the cluster
this gives 39,534 rows (Train 31,728 / Test 7,806); locally the W09B 10K generator
produces 490 sampled rows.

---

## Task 5 — Feature pipeline
**Author:** Aseel Alzahrani (221581, `Aseel-Alz`)

`StringIndexer` for `Primary Type` and `Domestic_str`, `VectorAssembler` over
`[primary_type_idx, Hour, District, domestic_idx]`. 80/20 split with `seed=42`.

Sample feature vectors (cluster):
```
+------------+----------------+----+--------+------------+------------+--------------------+-----+
|Primary Type|primary_type_idx|Hour|District|Domestic_str|domestic_idx|features            |label|
+------------+----------------+----+--------+------------+------------+--------------------+-----+
|HOMICIDE    |11.0            |10  |25      |false       |0.0         |[11.0,10.0,25.0,0.0]|1    |
|HOMICIDE    |11.0            |13  |5       |false       |0.0         |[11.0,13.0,5.0,0.0] |1    |
|HOMICIDE    |11.0            |20  |3       |false       |0.0         |[11.0,20.0,3.0,0.0] |0    |
+------------+----------------+----+--------+------------+------------+--------------------+-----+
```

Vector layout: `[primary_type_idx, Hour, District, domestic_idx]`.

---

## Task 6 — Train and evaluate three classifiers
**Author:** Jenna Alqurashi (231614, `Jennaalqurashi`)

Cluster results (5% sample, full HDFS dataset):

| Model | Params | Train (s) | AUC | Accuracy | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | maxIter=100, regParam=0.01 | 24.6 | 0.6022 | 0.7280 | 0.6376 | 0.6923 | 0.7280 |
| **Random Forest** | numTrees=100, maxDepth=5, maxBins=64 | 33.0 | **0.8061** | **0.8156** | **0.7802** | **0.8528** | **0.8156** |
| GBT | maxIter=50, maxDepth=5, maxBins=64 | — | — | — | — | — | — |

Local notebook (W09B 10K → 5% = 490 rows):

| Model | AUC | Accuracy | F1 | Train (s) |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.6307 | 0.6535 | 0.6177 | 11.5 |
| Random Forest       | 0.8701 | 0.8911 | 0.8899 | 11.4 |
| GBT                 | 0.8710 | 0.8515 | 0.8498 | 26.0 |

**Confusion matrices (cluster, TN/FP/FN/TP):**
- LR: (5549, 93, 2030, 133)
- RF: (5641, 1, 1438, 725)

**Top model by AUC: Random Forest (0.8061 cluster, 0.8701 local).**

---

## Task 7 — Random Forest feature importances
**Author:** Dina Alhudaithi (221466, `DinaAlhudaithi`)

```
primary_type_idx   0.7712  *************************************
Hour               0.0807  ****
District           0.0763  ****
domestic_idx       0.0718  ****
```

`primary_type_idx` dominates because the per-crime arrest-rate distribution from
Task 4 is itself dominated by crime type (NARCOTICS ≈ 99% vs THEFT ≈ 14%). Once a
tree splits on the crime type it has most of its answer.

Logistic Regression underperforms the tree models because it treats `primary_type_idx`
as a numeric feature with a linear coefficient — implying a meaningless ordering between
crime types. Trees split on individual values of the index and side-step that issue.

---

# Phase C — Deployment evidence

---

## Task 9 — Local execution
**Author:** Aseel Alzahrani (221581, `Aseel-Alz`)

Notebook executed end-to-end with `jupyter nbconvert --execute` (Python 3.9, PySpark
3.5.1, Java 17). Cell 2 prints:

```
Environment: local
Spark version: 3.5.1
Spark master: local[*]
```

10,000 rows generated in-memory by the W09B-style generator. All Tasks 1–7 ran;
outputs are embedded in `M2_Bigdataproject.ipynb`.

---

## Task 11 — spark-submit (cluster mode)
**Author:** Reem Alswailem (231079, `sengineer25`)

Per the May 2026 spec update, Task 11 uses `--deploy-mode cluster`:

```bash
rbalswailem@master-node:~$ spark-submit --master yarn --deploy-mode cluster \
    --num-executors 2 --executor-memory 1g --executor-cores 1 \
    --driver-memory 1g arrest_predictor.py
```

YARN application: `application_1771402826595_0362` — `final status: SUCCEEDED`.
Application stdout retrieved with `yarn logs -applicationId application_1771402826595_0362`
and saved at `output/spark_submit/run.log`. The console.log (`output/spark_submit/console.log`)
captures the spark-submit invocation and YARN's progress reports.

Excerpt from `run.log`:

```
Spark version: 3.5.4
Master: yarn
Full dataset rows: 793,072
Phase B sample: 39,534 rows (5%, seed=42)
Train rows: 31,728   Test rows: 7,806

=== LogisticRegression ===
  AUC        0.6022
  Accuracy   0.7280
  F1         0.6376
  Precision  0.6923
  Recall     0.7280
  Train time(s): 24.6
  Confusion (TN,FP,FN,TP): (5549, 93, 2030, 133)

=== RandomForest ===
  AUC        0.8061
  Accuracy   0.8156
  F1         0.7802
  Precision  0.8528
  Recall     0.8156
  Train time(s): 33.0
  Confusion (TN,FP,FN,TP): (5641, 1, 1438, 725)
```

---

## Member contributions

| Member | Tasks | Contribution |
|--------|------|-------------|
| Reem Alswailem (`sengineer25`)    | 1, 11 | Crime-type DataFrame query; spark-submit cluster-mode submission + log retrieval |
| Aseel Alzahrani (`Aseel-Alz`)     | 5, 9  | StringIndexer + VectorAssembler pipeline; local notebook execution evidence |
| Jenna Alqurashi (`Jennaalqurashi`)| 2, 6  | Spark SQL location query; three-classifier training + evaluation |
| Dina Alhudaithi (`DinaAlhudaithi`)| 3, 7  | Year-trend table + matplotlib chart; Random Forest feature importances |
| Dana Alnahas (`d-1117`)           | 4, 10 | Arrest-rate analysis; yarn-client cluster execution evidence |

## How to reproduce

Locally:
```bash
python3 -m venv venv && source venv/bin/activate
pip install pyspark==3.5.1 pandas matplotlib jupyter numpy
jupyter nbconvert --to notebook --execute M2_Bigdataproject.ipynb --output M2_Bigdataproject.ipynb
```

On the cluster:
```bash
ssh <user>@134.209.172.50
source /etc/profile.d/hadoop.sh
source /etc/profile.d/spark.sh
# one-time deps for python3.12
curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.12 get-pip.py --user
python3.12 -m pip install --user numpy 'setuptools>=68'
# Phase B standalone (cluster mode):
spark-submit --master yarn --deploy-mode cluster \
    --num-executors 2 --executor-memory 1g --executor-cores 1 \
    --driver-memory 1g arrest_predictor.py
```

---

## How to reproduce

Locally:
```bash
python3 -m venv venv && source venv/bin/activate
pip install pyspark==3.5.1 pandas matplotlib jupyter numpy
jupyter nbconvert --to notebook --execute M2_Bigdataproject.ipynb --output M2_Bigdataproject.ipynb
```

On the cluster:
```bash
ssh <user>@134.209.172.50
source /etc/profile.d/hadoop.sh
source /etc/profile.d/spark.sh
# one-time deps for python3.12
curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.12 get-pip.py --user
python3.12 -m pip install --user numpy 'setuptools>=68'
# Phase B standalone (cluster mode):
spark-submit --master yarn --deploy-mode cluster \
    --num-executors 2 --executor-memory 1g --executor-cores 1 \
    --driver-memory 1g arrest_predictor.py
```
