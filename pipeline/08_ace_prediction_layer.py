import json
import joblib
import pandas as pd


INPUT_FILE = "03_ACE_WITH_DIGESTION_FRAGMENTS.xlsx"
MODEL_FILE = "07_ACE_RANDOM_FOREST_MODEL.pkl"
OUTPUT_FILE = "08_ACE_PREDICTION_LAYER.xlsx"

AA = "ACDEFGHIKLMNPQRSTVWY"

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


def split_list(x):
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
        return "high_predicted_ACE_like"
    if prob >= 0.55:
        return "medium_predicted_ACE_like"
    if prob >= 0.40:
        return "low_medium_predicted_ACE_like"
    return "low_predicted_ACE_like"


def predict_sequence(seq, model, feature_names):
    features = extract_features(seq)
    X = pd.DataFrame([features])

    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_names].fillna(0)

    prob = float(model.predict_proba(X)[0, 1])

    return round(prob, 4), probability_label(prob)


def main():
    df = pd.read_excel(INPUT_FILE)

    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]
    feature_names = bundle["features"]

    rows = []

    for _, row in df.iterrows():
        parent_seq = clean_sequence(row.get("sequence"))
        parent_length = len(parent_seq)

        # Parent prediction sadece 5+ için anlamlı.
        if parent_length >= 5:
            parent_prob, parent_label = predict_sequence(
                parent_seq,
                model,
                feature_names
            )
        else:
            parent_prob, parent_label = None, "not_parent_candidate_length_lt_5"

        unknown_fragments = split_list(row.get("unknown_fragments"))

        predicted_fragment_rows = []

        for frag in unknown_fragments:
            if len(frag) < 2:
                continue

            frag_prob, frag_label = predict_sequence(
                frag,
                model,
                feature_names
            )

            predicted_fragment_rows.append({
                "fragment": frag,
                "prob": frag_prob,
                "label": frag_label,
                "length": len(frag),
            })

        if predicted_fragment_rows:
            best = sorted(
                predicted_fragment_rows,
                key=lambda x: x["prob"],
                reverse=True
            )[0]

            best_fragment = best["fragment"]
            best_fragment_prob = best["prob"]
            best_fragment_label = best["label"]
            predicted_fragment_count = len(predicted_fragment_rows)
            all_predicted_fragments = ",".join(
                f"{x['fragment']}:{x['prob']}" for x in predicted_fragment_rows
            )
        else:
            best_fragment = None
            best_fragment_prob = None
            best_fragment_label = None
            predicted_fragment_count = 0
            all_predicted_fragments = ""

        rows.append({
            "sequence": parent_seq,
            "parent_length": parent_length,

            "parent_predicted_ace_probability": parent_prob,
            "parent_predicted_ace_label": parent_label,

            "best_predicted_unknown_fragment": best_fragment,
            "best_unknown_fragment_predicted_ace_probability": best_fragment_prob,
            "best_unknown_fragment_predicted_ace_label": best_fragment_label,

            "predicted_unknown_fragment_count": predicted_fragment_count,
            "all_unknown_fragment_predictions": all_predicted_fragments,
        })

    out = pd.DataFrame(rows)

    out = out.sort_values(
        by=[
            "best_unknown_fragment_predicted_ace_probability",
            "parent_predicted_ace_probability",
        ],
        ascending=False,
        na_position="last"
    )

    out.to_excel(OUTPUT_FILE, index=False)

    print("\nBİTTİ")
    print("Input parent peptide:", len(df))
    print("Prediction row:", len(out))
    print("Dosya:", OUTPUT_FILE)

    print("\nTop 20:")
    print(out[[
        "sequence",
        "parent_predicted_ace_probability",
        "best_predicted_unknown_fragment",
        "best_unknown_fragment_predicted_ace_probability",
    ]].head(20))


if __name__ == "__main__":
    main()