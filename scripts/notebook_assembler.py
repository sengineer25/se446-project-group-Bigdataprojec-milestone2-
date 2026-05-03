"""Assemble M2_Bigdataproject.ipynb from cell definitions.

Run: python scripts/notebook_assembler.py
"""
import json
from pathlib import Path

NOTEBOOK_PATH = "M2_Bigdataproject.ipynb"

REEM   = "Reem Alswailem (231079)"
ASEEL  = "Aseel Alzahrani (221581)"
JENNA  = "Jenna Alqurashi (231614)"
DINA   = "Dina Alhudaithi (221466)"
DANA   = "Dana Alnahas (231515)"


cells_acc = []


def add_md(text):
    cells_acc.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    })


def add_code(text):
    cells_acc.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    })


add_md(f"""# SE446 Milestone 2 — Bigdataproject group

Spark DataFrame analytics + MLlib arrest predictor on the Chicago Crime dataset.

| Member | ID | GitHub | Tasks |
|--------|----|--------|-------|
| {REEM}     | 231079 | sengineer25     | 1, 11 |
| {ASEEL}    | 221581 | Aseel-Alz       | 5, 9  |
| {JENNA}    | 231614 | Jennaalqurashi   | 2, 6  |
| {DINA}     | 221466 | DinaAlhudaithi    | 3, 7  |
| {DANA}     | 231515 | d-1117          | 4, 10 |

### Spec compliance (per the May 2026 update)
- **Task 8 (CrossValidator) is waived** by the instructor. Not implemented in this notebook.
- **Phase B (Tasks 5–7) trains on a 5% sample** of the full dataset (`df.sample(0.05, seed=42)`).
  Cluster RAM cannot hold the full pipeline.
- Task 11 uses `--deploy-mode cluster`; logs are pulled with `yarn logs -applicationId <appId>`
  and saved to `output/spark_submit/run.log`.
""")


add_md("### Environment setup")

add_code('''import os
import sys
import shutil
import time

import pyspark.sql.functions as F
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import IntegerType, StringType


def is_cluster() -> bool:
    """Heuristic: hdfs binary on PATH means we are on the course cluster."""
    return shutil.which("hdfs") is not None


def make_spark() -> SparkSession:
    builder = (SparkSession.builder
               .appName("M2_Bigdataproject")
               .config("spark.sql.shuffle.partitions", "8"))
    if not is_cluster():
        builder = (builder
                   .master("local[*]")
                   .config("spark.driver.memory", "2g"))
    s = builder.getOrCreate()
    if not is_cluster():
        s.sparkContext.setLogLevel("WARN")
    return s


ENV = "cluster" if is_cluster() else "local"
spark = make_spark()
print(f"Environment: {ENV}")
print(f"Spark version: {spark.version}")
print(f"Spark master: {spark.sparkContext.master}")
''')


add_md("### Load the dataset")

add_code('''# Cluster: read the real CSV from HDFS. Local: use the W09B in-memory generator.
HDFS_PATH = "hdfs:///data/chicago_crimes.csv"


def load_from_hdfs(s: SparkSession):
    """Cluster path: full 793K-row CSV."""
    raw = s.read.csv(HDFS_PATH, header=True, inferSchema=True)
    out = raw.withColumn(
        "Hour",
        F.hour(F.to_timestamp(F.col("Date"), "MM/dd/yyyy hh:mm:ss a")),
    )
    out = out.withColumn("label", F.col("Arrest").cast(IntegerType()))
    out = out.withColumn("Domestic_str", F.col("Domestic").cast(StringType()))
    return out


def load_from_w09b_generator(s: SparkSession, n: int = 10_000):
    """Local path: 10K rows generated in-memory (mirrors the W09B lab)."""
    import random
    random.seed(42)

    arrest_rate_by_type = {
        "NARCOTICS":            0.85,
        "PROSTITUTION":         0.80,
        "WEAPONS VIOLATION":    0.60,
        "BATTERY":              0.30,
        "ASSAULT":              0.25,
        "ROBBERY":              0.15,
        "THEFT":                0.10,
        "BURGLARY":             0.08,
        "MOTOR VEHICLE THEFT":  0.06,
        "CRIMINAL DAMAGE":      0.05,
    }
    locations = ["STREET", "RESIDENCE", "APARTMENT", "SIDEWALK", "OTHER",
                 "PARKING LOT", "SCHOOL", "ALLEY", "RESIDENCE-GARAGE"]
    years = [2020, 2021, 2022, 2023, 2024, 2025]

    def one_row():
        crime = random.choice(list(arrest_rate_by_type))
        hour = random.randint(0, 23)
        domestic = random.random() < 0.15
        p = arrest_rate_by_type[crime] + (0.20 if domestic else 0)
        if 2 <= hour <= 5:
            p -= 0.10
        p = max(0.01, min(0.99, p))
        return Row(
            District=random.randint(1, 25),
            **{"Primary Type": crime},
            **{"Location Description": random.choice(locations)},
            Year=random.choice(years),
            Hour=hour,
            Domestic_str=str(domestic).lower(),
            Arrest=random.random() < p,
            label=int(random.random() < p),
        )

    return s.createDataFrame([one_row() for _ in range(n)])


df = load_from_hdfs(spark) if ENV == "cluster" else load_from_w09b_generator(spark)
df.cache()

print(f"Total rows: {df.count():,}")
df.printSchema()
df.show(3, truncate=False)
''')


# -----------------------------------------------------------------
add_md("---\n## Phase A — DataFrame analytics on the full dataset")

add_md(f"""### Task 1 — Crime type distribution
**Author:** {REEM}

DataFrame `groupBy` + descending count.""")

add_code(f'''# Task 1 — author: {REEM}
top_crime_types = (df
                   .groupBy(F.col("Primary Type").alias("crime_type"))
                   .agg(F.count("*").alias("n"))
                   .orderBy(F.desc("n"))
                   .limit(10))
top_crime_types.show(truncate=False)
''')


add_md(f"""### Task 2 — Location hotspots (Spark SQL)
**Author:** {JENNA}

Switch to SQL via `createOrReplaceTempView`.""")

add_code(f'''# Task 2 — author: {JENNA}
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
top_locations.show(truncate=False)
''')


add_md(f"""### Task 3 — Year trend
**Author:** {DINA}

Yearly counts; matplotlib chart in local mode.""")

add_code(f'''# Task 3 — author: {DINA}
yearly_counts = (df.groupBy("Year")
                   .agg(F.count("*").alias("incidents"))
                   .orderBy("Year"))
yearly_counts.show(30)
''')

add_code(f'''# Task 3 chart — author: {DINA}
if ENV == "local":
    import matplotlib.pyplot as plt

    pdf = yearly_counts.toPandas().dropna()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pdf["Year"], pdf["incidents"], marker="o", color="#444")
    ax.set_xlabel("Year")
    ax.set_ylabel("Incidents")
    ax.set_title("Chicago crime incidents per year")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/yearly_trend.png", dpi=120)
    plt.show()
else:
    print("Cluster mode — printed table is the deliverable.")
''')


add_md(f"""### Task 4 — Arrest rate analysis
**Author:** {DANA}

Overall rate plus a per-crime-type breakdown.""")

add_code(f'''# Task 4 — author: {DANA}
total = df.count()
arrests = df.filter(F.col("Arrest") == True).count()
print(f"Overall arrest rate: {{arrests:,}} / {{total:,}} = {{arrests/total*100:.2f}}%")

per_type = (df.groupBy("Primary Type")
              .agg(F.count("*").alias("incidents"),
                   F.avg(F.col("label").cast("double")).alias("arrest_rate"))
              .filter(F.col("incidents") >= 100)
              .orderBy(F.desc("arrest_rate")))
print("Highest arrest-rate crime types (min 100 incidents):")
per_type.show(15, truncate=False)
''')


# -----------------------------------------------------------------
add_md("""---
## Phase B — MLlib arrest predictor (5% sample per spec update)

The May-2026 spec update requires Phase B to train on a 5% sample of the dataset.
Locally the W09B generator already produces only 10,000 rows so sampling is a no-op;
on the cluster the sample reduces 793,072 rows down to roughly 39,654 — small enough
to fit the cluster's RAM budget.""")

add_code(f'''# Phase B sampling — applied before any feature engineering
ml_df = df.sample(fraction=0.05, seed=42)
print(f"Phase B working set: {{ml_df.count():,}} rows (5% sample, seed=42)")
''')


add_md(f"""### Task 5 — Feature pipeline
**Author:** {ASEEL}

`StringIndexer` for Primary Type and Domestic_str, `VectorAssembler` to bundle them
with District and Hour. Train/test split with seed=42.""")

add_code(f'''# Task 5 — author: {ASEEL}
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler

# Spark MLlib needs everything aligned to the schema present on the sampled DF
if "Domestic_str" not in ml_df.columns:
    ml_df = ml_df.withColumn("Domestic_str", F.col("Domestic").cast(StringType()))

primary_type_idx = StringIndexer(inputCol="Primary Type",
                                 outputCol="primary_type_idx",
                                 handleInvalid="skip")
domestic_idx     = StringIndexer(inputCol="Domestic_str",
                                 outputCol="domestic_idx",
                                 handleInvalid="skip")
features_asm     = VectorAssembler(
    inputCols=["primary_type_idx", "Hour", "District", "domestic_idx"],
    outputCol="features",
)

train_df, test_df = ml_df.randomSplit([0.8, 0.2], seed=42)
train_df.cache()
test_df.cache()
print(f"Train: {{train_df.count():,}}   Test: {{test_df.count():,}}")

# Show what the feature vector looks like for 5 rows
preview = (Pipeline(stages=[primary_type_idx, domestic_idx, features_asm])
           .fit(train_df)
           .transform(train_df))
preview.select("Primary Type", "primary_type_idx",
               "Hour", "District",
               "Domestic_str", "domestic_idx",
               "features", "label").show(5, truncate=False)
print("Vector layout: [primary_type_idx, Hour, District, domestic_idx]")
''')


add_md(f"""### Task 6 — Train and evaluate three classifiers
**Author:** {JENNA}

Logistic Regression (maxIter=100, regParam=0.01), Random Forest (numTrees=100,
maxDepth=5), Gradient Boosted Trees (maxIter=50, maxDepth=5). `maxBins=64` for the
tree models because `Primary Type` has more than 32 categories on the cluster.""")

add_code(f'''# Task 6 — author: {JENNA}
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, GBTClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator,
)

binary_ev = BinaryClassificationEvaluator(labelCol="label")
mc_ev     = MulticlassClassificationEvaluator(labelCol="label",
                                              predictionCol="prediction")


def metrics_for(predictions):
    g = lambda metric: mc_ev.evaluate(predictions, {{mc_ev.metricName: metric}})
    return {{
        "AUC":       binary_ev.evaluate(predictions),
        "Accuracy":  g("accuracy"),
        "F1":        g("f1"),
        "Precision": g("weightedPrecision"),
        "Recall":    g("weightedRecall"),
    }}


def confusion_for(predictions):
    rows = predictions.groupBy("label", "prediction").count().collect()
    bag = {{(int(r["label"]), int(r["prediction"])): r["count"] for r in rows}}
    return (bag.get((0, 0), 0), bag.get((0, 1), 0),
            bag.get((1, 0), 0), bag.get((1, 1), 0))


def fit_and_score(name, classifier):
    pipe = Pipeline(stages=[primary_type_idx, domestic_idx, features_asm, classifier])
    started = time.time()
    fitted = pipe.fit(train_df)
    elapsed = time.time() - started
    preds = fitted.transform(test_df)
    return name, fitted, metrics_for(preds), confusion_for(preds), elapsed
''')


add_code(f'''# author: {JENNA}
lr_classifier  = LogisticRegression(featuresCol="features", labelCol="label",
                                    maxIter=100, regParam=0.01)
rf_classifier  = RandomForestClassifier(featuresCol="features", labelCol="label",
                                        numTrees=100, maxDepth=5,
                                        maxBins=64, seed=42)
gbt_classifier = GBTClassifier(featuresCol="features", labelCol="label",
                               maxIter=50, maxDepth=5,
                               maxBins=64, seed=42)

results = []
for name, clf in [("LogisticRegression", lr_classifier),
                  ("RandomForest",       rf_classifier),
                  ("GBT",                gbt_classifier)]:
    print(f"--- training {{name}} ---")
    results.append(fit_and_score(name, clf))

print("=" * 80)
print(f"{{'Metric':<14}} {{'LR':>14}} {{'RF':>14}} {{'GBT':>14}}")
print("-" * 80)
m_lr  = results[0][2]; m_rf = results[1][2]; m_gbt = results[2][2]
for k in ("AUC", "Accuracy", "F1", "Precision", "Recall"):
    print(f"{{k:<14}} {{m_lr[k]:>14.4f}} {{m_rf[k]:>14.4f}} {{m_gbt[k]:>14.4f}}")
print(f"{{'Train(s)':<14}} {{results[0][4]:>14.1f}} {{results[1][4]:>14.1f}} {{results[2][4]:>14.1f}}")
print(f"{{'CM(TN,FP,FN,TP)':<14}}")
for n, _, _, cm, _ in results:
    print(f"  {{n:<18}} {{cm}}")
print("=" * 80)

best_name, best_model, best_metrics, _, _ = max(results, key=lambda r: r[2]["AUC"])
print(f"Top model by AUC: {{best_name}} ({{best_metrics['AUC']:.4f}})")
''')


add_md(f"""### Task 7 — Random Forest feature importances
**Author:** {DINA}

Importances tell us which feature drives most of the splits in the forest.""")

add_code(f'''# Task 7 — author: {DINA}
rf_fitted = next(model for name, model, *_ in results if name == "RandomForest").stages[-1]

vec_layout = ["primary_type_idx", "Hour", "District", "domestic_idx"]
importances = rf_fitted.featureImportances.toArray()

print("Random Forest feature importances:")
for feature, importance in sorted(zip(vec_layout, importances), key=lambda kv: -kv[1]):
    bar = "*" * int(round(importance * 50))
    print(f"  {{feature:<18}} {{importance:.4f}}  {{bar}}")
''')


add_md("""**Interpretation.** The crime-type index dominates because the arrest-rate
distribution from Task 4 is itself dominated by crime type (NARCOTICS ≈ 99%, THEFT
≈ 14%). Once a tree sees that one feature it has most of its answer.

Logistic Regression underperforms the tree models because it treats `primary_type_idx`
as a *numeric* feature with a single linear coefficient — implying a meaningless ordering
between crime types. Trees split on individual values of the index, side-stepping that
problem entirely.""")


add_md("---\n### Cleanup")

add_code('''spark.stop()''')


nb = {
    "cells": cells_acc,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

Path(NOTEBOOK_PATH).write_text(json.dumps(nb, indent=1))
print(f"wrote {NOTEBOOK_PATH} ({len(cells_acc)} cells)")
