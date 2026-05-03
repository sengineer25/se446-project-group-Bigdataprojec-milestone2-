"""SE446 Milestone 2 — Bigdataproject

Phase B (Tasks 5-7): standalone arrest predictor for spark-submit.

Authors:
    Task 5 — Aseel Alzahrani (221581)
    Task 6 — Jenna Alqurashi (231614)
    Task 7 — Dina Alhudaithi (221466)

Per the May-2026 spec update:
    * Task 8 (CrossValidator) is waived.
    * Phase B trains on a 5% sample (df.sample(0.05, seed=42)).

Submit via:
    spark-submit \\
        --master yarn --deploy-mode cluster \\
        --num-executors 2 --executor-memory 1g --executor-cores 1 \\
        arrest_predictor.py
"""
import time

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, GBTClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator,
)


HDFS_DATASET = "hdfs:///data/chicago_crimes.csv"


def build_session() -> SparkSession:
    return (SparkSession.builder
            .appName("M2_Bigdataproject_arrest_predictor")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate())


def read_chicago_crimes(spark: SparkSession):
    raw = spark.read.csv(HDFS_DATASET, header=True, inferSchema=True)
    parsed = (raw
              .withColumn("Hour",
                          F.hour(F.to_timestamp(F.col("Date"),
                                                "MM/dd/yyyy hh:mm:ss a")))
              .withColumn("label",        F.col("Arrest").cast(IntegerType()))
              .withColumn("Domestic_str", F.col("Domestic").cast(StringType())))
    parsed = parsed.dropna(subset=["District", "Primary Type",
                                   "Hour", "Domestic_str", "label"])
    return parsed


def evaluate(predictions, binary_ev, mc_ev):
    score = lambda metric: mc_ev.evaluate(predictions,
                                          {mc_ev.metricName: metric})
    return {
        "AUC":       binary_ev.evaluate(predictions),
        "Accuracy":  score("accuracy"),
        "F1":        score("f1"),
        "Precision": score("weightedPrecision"),
        "Recall":    score("weightedRecall"),
    }


def confusion(predictions):
    rows = predictions.groupBy("label", "prediction").count().collect()
    bag = {(int(r["label"]), int(r["prediction"])): r["count"] for r in rows}
    return (bag.get((0, 0), 0), bag.get((0, 1), 0),
            bag.get((1, 0), 0), bag.get((1, 1), 0))


def main():
    spark = build_session()
    print(f"Spark version: {spark.version}")
    print(f"Master: {spark.sparkContext.master}")

    full_df = read_chicago_crimes(spark)
    print(f"Full dataset rows: {full_df.count():,}")

    # ----- Task 5 (Aseel): feature pipeline + 5% sample -----
    sample_df = full_df.sample(fraction=0.05, seed=42)
    print(f"Phase B sample: {sample_df.count():,} rows (5%, seed=42)")

    primary_idx  = StringIndexer(inputCol="Primary Type",
                                 outputCol="primary_type_idx",
                                 handleInvalid="skip")
    domestic_idx = StringIndexer(inputCol="Domestic_str",
                                 outputCol="domestic_idx",
                                 handleInvalid="skip")
    asm = VectorAssembler(
        inputCols=["primary_type_idx", "Hour", "District", "domestic_idx"],
        outputCol="features",
    )

    train_df, test_df = sample_df.randomSplit([0.8, 0.2], seed=42)
    train_df.cache()
    test_df.cache()
    print(f"Train rows: {train_df.count():,}   Test rows: {test_df.count():,}")

    binary_ev = BinaryClassificationEvaluator(labelCol="label")
    mc_ev     = MulticlassClassificationEvaluator(labelCol="label",
                                                  predictionCol="prediction")

    classifiers = [
        ("LogisticRegression",
         LogisticRegression(featuresCol="features", labelCol="label",
                            maxIter=100, regParam=0.01)),
        ("RandomForest",
         RandomForestClassifier(featuresCol="features", labelCol="label",
                                numTrees=100, maxDepth=5,
                                maxBins=64, seed=42)),
        ("GBT",
         GBTClassifier(featuresCol="features", labelCol="label",
                       maxIter=50, maxDepth=5,
                       maxBins=64, seed=42)),
    ]

    table = []
    rf_fitted_stages = None
    for name, clf in classifiers:
        pipe = Pipeline(stages=[primary_idx, domestic_idx, asm, clf])
        t0 = time.time()
        model = pipe.fit(train_df)
        elapsed = time.time() - t0
        preds = model.transform(test_df)
        m = evaluate(preds, binary_ev, mc_ev)
        cm = confusion(preds)
        table.append((name, elapsed, m, cm))
        print(f"\n=== {name} ===")
        for k, v in m.items():
            print(f"  {k:<10} {v:.4f}")
        print(f"  Train time(s): {elapsed:.1f}")
        print(f"  Confusion (TN,FP,FN,TP): {cm}")
        if name == "RandomForest":
            rf_fitted_stages = model.stages[-1]

    print("\n" + "=" * 80)
    print(f"{'Metric':<14}{'LR':>14}{'RF':>14}{'GBT':>14}")
    print("-" * 80)
    for k in ("AUC", "Accuracy", "F1", "Precision", "Recall"):
        print(f"{k:<14}{table[0][2][k]:>14.4f}{table[1][2][k]:>14.4f}{table[2][2][k]:>14.4f}")
    print(f"{'Train(s)':<14}{table[0][1]:>14.1f}{table[1][1]:>14.1f}{table[2][1]:>14.1f}")
    print("=" * 80)

    best = max(table, key=lambda row: row[2]["AUC"])
    print(f"Top model by AUC: {best[0]} ({best[2]['AUC']:.4f})")

    # ----- Task 7 (Dina): RF feature importances -----
    print("\n--- Random Forest feature importances ---")
    layout = ["primary_type_idx", "Hour", "District", "domestic_idx"]
    for feat, score in sorted(zip(layout,
                                  rf_fitted_stages.featureImportances.toArray()),
                              key=lambda kv: -kv[1]):
        bar = "*" * int(round(score * 50))
        print(f"  {feat:<18} {score:.4f}  {bar}")

    spark.stop()


if __name__ == "__main__":
    main()
