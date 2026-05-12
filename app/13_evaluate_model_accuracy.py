import json
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split


DATASET_FILE = "06_ML_TRAINING_DATASET.xlsx"
MODEL_FILE = "07_ACE_RANDOM_FOREST_MODEL.pkl"
FEATURE_FILE = "07_MODEL_FEATURE_COLUMNS.json"
FINAL_FILE = "12_DIGIDIGEST_FINAL_RESULTS.xlsx"

OUTPUT_FILE = "13_MODEL_VALIDATION_REPORT.xlsx"

RANDOM_STATE = 42


def load_model_and_features():
    bundle = joblib.load(MODEL_FILE)

    if isinstance(bundle, dict):
        model = bundle["model"]
        features = bundle["features"]
    else:
        model = bundle

        with open(FEATURE_FILE, "r") as f:
            features = json.load(f)

    return model, features


def main():
    df = pd.read_excel(DATASET_FILE)
    model, features = load_model_and_features()

    y = df["is_ace"].astype(int)

    X = df[features].copy()
    X = X.select_dtypes(include=["number"]).fillna(0)

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X,
        y,
        df.index,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    print("\nMODEL TEST METRICS")
    for k, v in metrics.items():
        print(f"{k}: {round(v, 4)}")

    print("\nCONFUSION MATRIX")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred, digits=4))

    test_predictions = df.loc[test_idx, [
        "sequence",
        "length",
        "is_ace",
        "ml_label_type",
        "role_for_model",
        "source_activity_type",
        "source_query",
    ]].copy()

    test_predictions["predicted_is_ace"] = y_pred
    test_predictions["predicted_ace_probability"] = y_prob
    test_predictions["correct_prediction"] = (
        test_predictions["is_ace"] == test_predictions["predicted_is_ace"]
    ).astype(int)

    threshold_rows = []

    for threshold in [0.40, 0.50, 0.55, 0.60, 0.70, 0.75, 0.80]:
        pred_t = (y_prob >= threshold).astype(int)

        threshold_rows.append({
            "threshold": threshold,
            "accuracy": accuracy_score(y_test, pred_t),
            "precision": precision_score(y_test, pred_t, zero_division=0),
            "recall": recall_score(y_test, pred_t, zero_division=0),
            "f1_score": f1_score(y_test, pred_t, zero_division=0),
        })

    threshold_df = pd.DataFrame(threshold_rows)

    metrics_df = pd.DataFrame([
        {"metric": k, "value": round(v, 4)}
        for k, v in metrics.items()
    ])

    cm_df = pd.DataFrame(
        cm,
        index=["actual_non_ACE", "actual_ACE"],
        columns=["predicted_non_ACE", "predicted_ACE"]
    )

    final_summary = pd.DataFrame()

    try:
        final_df = pd.read_excel(FINAL_FILE)

        final_summary = pd.DataFrame({
            "metric": [
                "total_final_candidates",
                "mean_final_score",
                "median_final_score",
                "tier_1_count",
                "tier_2_count",
                "high_confidence_count",
            ],
            "value": [
                len(final_df),
                round(final_df["final_digidigest_score"].mean(), 2),
                round(final_df["final_digidigest_score"].median(), 2),
                (final_df["final_candidate_class"] == "Tier 1 oral ACE candidate").sum(),
                (final_df["final_candidate_class"] == "Tier 2 oral ACE candidate").sum(),
                (final_df["final_confidence"] == "high_confidence").sum(),
            ]
        })

        top_final = final_df.head(50).copy()

    except Exception:
        top_final = pd.DataFrame()

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="Model_Metrics", index=False)
        cm_df.to_excel(writer, sheet_name="Confusion_Matrix")
        threshold_df.to_excel(writer, sheet_name="Threshold_Analysis", index=False)
        test_predictions.to_excel(writer, sheet_name="Test_Predictions", index=False)

        if not final_summary.empty:
            final_summary.to_excel(writer, sheet_name="Final_Pipeline_Summary", index=False)
            top_final.to_excel(writer, sheet_name="Top_Final_Candidates", index=False)

    print("\nBİTTİ")
    print("Validation report:", OUTPUT_FILE)
    print("\nNot:")
    print(
        "Bu accuracy, ACE-like ML modelinin test doğruluğudur. "
        "Final DigiDigest score için gerçek lab sonucu olmadan klasik accuracy hesaplanamaz; "
        "onun için literature/lab validation set gerekir."
    )


if __name__ == "__main__":
    main()