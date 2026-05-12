import json
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance


INPUT_FILE = "06_ML_TRAINING_DATASET.xlsx"

MODEL_OUTPUT = "07_ACE_RANDOM_FOREST_MODEL.pkl"
FEATURE_OUTPUT = "07_MODEL_FEATURE_COLUMNS.json"
IMPORTANCE_OUTPUT = "07_FEATURE_IMPORTANCE.xlsx"

RANDOM_STATE = 42


DROP_COLUMNS = [
    "sequence",
    "is_ace",
    "ml_label_type",
    "role_for_model",
    "source_activity_type",
    "source_query",
    "source_name",
    "source_label",
    "n_terminal",
    "c_terminal",
]


def main():
    df = pd.read_excel(INPUT_FILE)

    print("\nML DATASET YÜKLENDİ")
    print("Toplam satır:", len(df))

    y = df["is_ace"].astype(int)

    feature_cols = [
        c for c in df.columns
        if c not in DROP_COLUMNS
    ]

    X = df[feature_cols].copy()

    # Model sadece sayısal feature kullanır.
    # n_terminal / c_terminal gibi string kolonlar dışarıda bırakılır.
    X = X.select_dtypes(include=["number"]).fillna(0)
    feature_cols = list(X.columns)

    print("Feature sayısı:", len(feature_cols))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTrain:", len(X_train))
    print("Test:", len(X_test))

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nMODEL EĞİTİLİYOR...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    print("\nMODEL SONUÇLARI")
    print(f"Accuracy : {acc:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    print("\nCONFUSION MATRIX")
    print(confusion_matrix(y_test, preds))

    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_test, preds, digits=4))

    print("\nFEATURE IMPORTANCE HESAPLANIYOR...")

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    })

    importance_df = importance_df.sort_values(
        "importance_mean",
        ascending=False
    ).reset_index(drop=True)

    importance_df.to_excel(
        IMPORTANCE_OUTPUT,
        index=False
    )

    print("\nTOP 20 FEATURE")
    print(importance_df.head(20))

    joblib.dump(
        {
            "model": model,
            "features": feature_cols,
        },
        MODEL_OUTPUT
    )

    with open(FEATURE_OUTPUT, "w") as f:
        json.dump(feature_cols, f)

    print("\nMODEL KAYDEDİLDİ")
    print(MODEL_OUTPUT)

    print("\nFEATURE LİSTESİ KAYDEDİLDİ")
    print(FEATURE_OUTPUT)

    print("\nFEATURE IMPORTANCE DOSYASI")
    print(IMPORTANCE_OUTPUT)

    print("\nMODELİN GERÇEK AMACI:")
    print(
        "Bu model gerçek biyolojik bağlanma (binding) tahmini yapmaz.\n"
        "Amaç, ACE inhibitör peptitlerde görülen sequence-level örüntüleri\n"
        "öğrenerek unknown peptide ve fragmentler için ACE-like probability üretmektir."
    )


if __name__ == "__main__":
    main()