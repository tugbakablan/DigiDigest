import joblib
import pandas as pd


MODEL_FILE = "07_ACE_RANDOM_FOREST_MODEL.pkl"
OUTPUT_FILE = "14_REAL_LAB_VALIDATION_RESULTS.xlsx"

AA = "ACDEFGHIKLMNPQRSTVWY"

LAB_CASES = [
    {
        "case_id": "P001",
        "title": "ACE inhibitory activity and gastrointestinal digestion stability of umami peptides IIVFGRQLL from yeast extract",
        "parent_sequence": "IIVFGRQLL",
        "observed_outcome": "Sindirim sonrası ACE aktivitesi %39 düştü ama kaybolmadı",
        "reported_fragments": "IVF,VF,FGR,RQL",
        "evidence_strength": "Very strong: in vitro + LC-MS/MS + docking + SHR rat",
        "expected_system_behavior": "Digestive activation / activity retention signal",
        "validation_target": "digestive_activation_scoring",
    }
]


HYDROPHOBIC = set("AILMFWYV")
AROMATIC = set("FWY")
POSITIVE = set("KRH")
NEGATIVE = set("DE")
POLAR = set("QNST")
ALIPHATIC = set("ILVA")
PROLINE = set("P")


def clean_sequence(seq):
    seq = str(seq).strip().upper()
    return "".join(c for c in seq if c in AA)


def split_fragments(x):
    if pd.isna(x) or str(x).strip() == "":
        return []
    return [clean_sequence(i) for i in str(x).split(",") if clean_sequence(i)]


def aa_count(seq, group):
    return sum(1 for c in seq if c in group) if seq else 0


def ratio(seq, group):
    return aa_count(seq, group) / len(seq) if seq else 0


def terminal_onehot(seq, position, aa):
    if not seq:
        return 0
    if position == "N":
        return int(seq[0] == aa)
    return int(seq[-1] == aa)


def extract_features(seq):
    seq = clean_sequence(seq)
    length = len(seq)

    features = {
        "length": length,
        "hydrophobic_ratio": ratio(seq, HYDROPHOBIC),
        "aromatic_ratio": ratio(seq, AROMATIC),
        "positive_charge_ratio": ratio(seq, POSITIVE),
        "negative_charge_ratio": ratio(seq, NEGATIVE),
        "polar_ratio": ratio(seq, POLAR),
        "aliphatic_ratio": ratio(seq, ALIPHATIC),
        "proline_ratio": ratio(seq, PROLINE),
        "hydrophobic_count": aa_count(seq, HYDROPHOBIC),
        "aromatic_count": aa_count(seq, AROMATIC),
        "positive_charge_count": aa_count(seq, POSITIVE),
        "negative_charge_count": aa_count(seq, NEGATIVE),
        "polar_count": aa_count(seq, POLAR),
        "aliphatic_count": aa_count(seq, ALIPHATIC),
        "proline_count": aa_count(seq, PROLINE),
        "has_proline": int("P" in seq),
        "has_aromatic": int(any(a in seq for a in AROMATIC)),
        "c_terminal_hydrophobic": int(length > 0 and seq[-1] in HYDROPHOBIC),
        "c_terminal_aromatic": int(length > 0 and seq[-1] in AROMATIC),
        "c_terminal_proline": int(length > 0 and seq[-1] == "P"),
        "n_terminal_hydrophobic": int(length > 0 and seq[0] in HYDROPHOBIC),
        "n_terminal_aromatic": int(length > 0 and seq[0] in AROMATIC),
        "n_terminal_proline": int(length > 0 and seq[0] == "P"),
    }

    for aa in AA:
        features[f"aac_{aa}"] = seq.count(aa) / length if length else 0
        features[f"nterm_{aa}"] = terminal_onehot(seq, "N", aa)
        features[f"cterm_{aa}"] = terminal_onehot(seq, "C", aa)

    motifs = [
        "VP", "IP", "LP", "PP", "PY", "IY", "VY", "VF",
        "GF", "GR", "PG", "KP", "RP", "FP", "YP", "PR"
    ]

    for motif in motifs:
        features[f"contains_{motif.lower()}"] = int(motif in seq)

    return features


def probability_label(prob):
    if prob >= 0.75:
        return "high ACE-like"
    if prob >= 0.55:
        return "medium ACE-like"
    if prob >= 0.40:
        return "low-medium ACE-like"
    return "low ACE-like"


def predict(seq, model, feature_names):
    seq = clean_sequence(seq)
    features = extract_features(seq)

    X = pd.DataFrame([features])

    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_names].fillna(0)

    prob = float(model.predict_proba(X)[0, 1])

    return round(prob, 4), probability_label(prob)


def validate_case(case, model, feature_names):
    rows = []

    parent = clean_sequence(case["parent_sequence"])
    parent_prob, parent_label = predict(parent, model, feature_names)

    rows.append({
        "case_id": case["case_id"],
        "sequence_type": "parent",
        "sequence": parent,
        "length": len(parent),
        "predicted_ace_probability": parent_prob,
        "predicted_ace_percent": round(parent_prob * 100, 2),
        "prediction_label": parent_label,
        "reported_outcome": case["observed_outcome"],
        "evidence_strength": case["evidence_strength"],
        "validation_target": case["validation_target"],
    })

    fragments = split_fragments(case["reported_fragments"])

    fragment_probs = []

    for frag in fragments:
        prob, label = predict(frag, model, feature_names)
        fragment_probs.append(prob)

        rows.append({
            "case_id": case["case_id"],
            "sequence_type": "reported_digestive_fragment",
            "sequence": frag,
            "length": len(frag),
            "predicted_ace_probability": prob,
            "predicted_ace_percent": round(prob * 100, 2),
            "prediction_label": label,
            "reported_outcome": case["observed_outcome"],
            "evidence_strength": case["evidence_strength"],
            "validation_target": case["validation_target"],
        })

    max_frag_prob = max(fragment_probs) if fragment_probs else 0
    mean_frag_prob = sum(fragment_probs) / len(fragment_probs) if fragment_probs else 0

    if max_frag_prob >= 0.75:
        system_match = "strong_match"
        interpretation = "Sistem, literatürde bildirilen sindirim fragmentlerinden en az birini güçlü ACE-like aday olarak yakaladı."
    elif max_frag_prob >= 0.55:
        system_match = "partial_match"
        interpretation = "Sistem, bildirilen fragmentlerde orta düzey ACE-like sinyal yakaladı."
    else:
        system_match = "weak_match"
        interpretation = "Sistem, bildirilen fragmentlerde güçlü ACE-like sinyal yakalayamadı."

    summary = {
        "case_id": case["case_id"],
        "parent_sequence": parent,
        "parent_predicted_ace_percent": round(parent_prob * 100, 2),
        "max_fragment_predicted_ace_percent": round(max_frag_prob * 100, 2),
        "mean_fragment_predicted_ace_percent": round(mean_frag_prob * 100, 2),
        "system_match": system_match,
        "expected_system_behavior": case["expected_system_behavior"],
        "interpretation_tr": interpretation,
    }

    return rows, summary


def main():
    bundle = joblib.load(MODEL_FILE)

    if isinstance(bundle, dict):
        model = bundle["model"]
        feature_names = bundle["features"]
    else:
        raise RuntimeError("Model dosyası dict formatında değil. 07_train_ace_model.py çıktısını kontrol et.")

    all_rows = []
    summaries = []

    for case in LAB_CASES:
        rows, summary = validate_case(case, model, feature_names)
        all_rows.extend(rows)
        summaries.append(summary)

    detail_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summaries)

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Validation_Summary", index=False)
        detail_df.to_excel(writer, sheet_name="Sequence_Level_Results", index=False)

    print("\nBİTTİ")
    print("Dosya:", OUTPUT_FILE)

    print("\nVALIDATION SUMMARY")
    print(summary_df)

    print("\nSEQUENCE LEVEL")
    print(detail_df[[
        "sequence_type",
        "sequence",
        "predicted_ace_percent",
        "prediction_label"
    ]])


if __name__ == "__main__":
    main()