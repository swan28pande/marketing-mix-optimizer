"""Data loading: synthetic generator, pandas, and Spark aggregation."""
import numpy as np
import pandas as pd


def synthetic(n_weeks: int = 104, channels=("tv", "search", "social", "display"), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.gamma(2.0, 50.0, n_weeks) for c in channels})
    df["seasonality"] = np.sin(2 * np.pi * np.arange(n_weeks) / 52)
    true_beta = rng.uniform(50, 200, len(channels))
    sat = df[list(channels)].values / (df[list(channels)].values + 100.0)
    df["revenue"] = 500 + sat @ true_beta + 80 * df["seasonality"] + rng.normal(0, 20, n_weeks)
    return df


def aggregate_with_spark(path: str, date_col: str = "date", freq: str = "week"):
    """Aggregate raw event-level spend logs to weekly channel totals with Spark."""
    from pyspark.sql import SparkSession, functions as F

    spark = SparkSession.builder.appName("mmo-ingest").getOrCreate()
    raw = spark.read.option("header", True).option("inferSchema", True).csv(path)
    weekly = (
        raw.withColumn("period", F.date_trunc(freq, F.col(date_col)))
        .groupBy("period")
        .pivot("channel")
        .agg(F.sum("spend"))
        .orderBy("period")
    )
    return weekly.toPandas().fillna(0.0)
