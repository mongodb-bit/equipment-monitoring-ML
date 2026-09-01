"""
Train an anomaly-scoring model for equipment telemetry and save it as model.pkl.

Replaces the hardcoded if/else threshold logic with a trained RandomForestRegressor
that predicts a continuous anomaly severity score (0.0 - 1.0) from telemetry +
equipment threshold values. Risk-level buckets are still applied on top of the
score (LOW / MEDIUM / HIGH / CRITICAL), but the score itself now comes from a
model artifact you can retrain later on real historical/labeled data.

Run:
    python train_model.py
Produces:
    model.pkl   -> joblib-dumped dict {"model": ..., "feature_names": [...]}
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

RANDOM_STATE = 42
N_SAMPLES = 20000

rng = np.random.default_rng(RANDOM_STATE)


def generate_synthetic_dataset(n=N_SAMPLES):
    """
    Generates synthetic equipment telemetry + thresholds, and computes a
    continuous 'violation severity' label per row based on how far each
    reading falls outside its allowed [min, max] band (0 = perfectly
    within range on all 4 signals, 1 = maximally violating all of them).

    This label generalizes the original hard if/else rule: instead of just
    counting how many of the 4 checks failed, it also captures *how badly*
    each one failed, giving the model a smoother, more informative target
    than a 0/0.25/0.5/0.75/1.0 step function.
    """

    # --- equipment threshold "profiles" (random but realistic ranges) ---
    minTemperature = rng.uniform(0, 40, n)
    maxTemperature = minTemperature + rng.uniform(20, 80, n)

    minPressure = rng.uniform(10, 50, n)
    maxPressure = minPressure + rng.uniform(20, 100, n)

    minFlowRate = rng.uniform(5, 20, n)
    maxFlowRate = minFlowRate + rng.uniform(10, 50, n)

    maxErrorCount = rng.integers(1, 10, n)

    # --- telemetry readings: mix of "in-range" and "out-of-range" samples ---
    def sample_reading(lo, hi, out_of_range_frac=0.35, spread=0.6):
        span = hi - lo
        in_range = rng.uniform(lo, hi)
        # push some samples outside the band on either side
        below = lo - rng.uniform(0, spread) * span
        above = hi + rng.uniform(0, spread) * span
        outside = np.where(rng.random(n) < 0.5, below, above)
        mask = rng.random(n) < out_of_range_frac
        return np.where(mask, outside, in_range)

    temperature = sample_reading(minTemperature, maxTemperature)
    pressure = sample_reading(minPressure, maxPressure)
    flowRate = sample_reading(minFlowRate, maxFlowRate)

    errorCount = rng.poisson(lam=maxErrorCount * 0.6, size=n).astype(float)
    # occasionally spike error counts well past the max
    spike_mask = rng.random(n) < 0.2
    errorCount = np.where(spike_mask, errorCount + rng.integers(1, 15, n), errorCount)

    runtimeHours = rng.uniform(0, 20000, n)
    equipmentId = np.arange(1, n + 1)

    def severity(value, lo, hi):
        span = np.maximum(hi - lo, 1e-6)
        below = np.clip((lo - value) / span, 0, None)
        above = np.clip((value - hi) / span, 0, None)
        raw = np.maximum(below, above)
        # squash to [0, 1] so extreme outliers don't dominate the label
        return raw / (1.0 + raw)

    temp_sev = severity(temperature, minTemperature, maxTemperature)
    pressure_sev = severity(pressure, minPressure, maxPressure)
    flow_sev = severity(flowRate, minFlowRate, maxFlowRate)
    error_sev = np.clip((errorCount - maxErrorCount) / np.maximum(maxErrorCount, 1), 0, None)
    error_sev = error_sev / (1.0 + error_sev)

    anomaly_score = (temp_sev + pressure_sev + flow_sev + error_sev) / 4.0
    # light label noise so the model doesn't just memorize a deterministic formula
    anomaly_score = np.clip(anomaly_score + rng.normal(0, 0.02, n), 0, 1)

    df = pd.DataFrame({
        "equipmentId": equipmentId,
        "temperature": temperature,
        "pressure": pressure,
        "flowRate": flowRate,
        "runtimeHours": runtimeHours,
        "errorCount": errorCount,
        "minTemperature": minTemperature,
        "maxTemperature": maxTemperature,
        "minPressure": minPressure,
        "maxPressure": maxPressure,
        "minFlowRate": minFlowRate,
        "maxFlowRate": maxFlowRate,
        "maxErrorCount": maxErrorCount,
        "anomaly_score": anomaly_score,
    })
    return df


FEATURE_NAMES = [
    "temperature", "pressure", "flowRate", "runtimeHours", "errorCount",
    "minTemperature", "maxTemperature",
    "minPressure", "maxPressure",
    "minFlowRate", "maxFlowRate",
    "maxErrorCount",
]


def main():
    df = generate_synthetic_dataset()

    X = df[FEATURE_NAMES]
    y = df["anomaly_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Validation MAE: {mae:.4f}")
    print("Feature importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda t: -t[1]):
        print(f"  {name:16s} {imp:.3f}")

    artifact = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "model_version": "v2-rf",
    }
    joblib.dump(artifact, "model.pkl")
    print("\nSaved model.pkl")


if __name__ == "__main__":
    main()
