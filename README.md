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

# Phase C — Deployment evidence

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
