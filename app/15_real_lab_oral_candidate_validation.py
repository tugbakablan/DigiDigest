import joblib
import pandas as pd


MODEL_FILE = "07_ACE_RANDOM_FOREST_MODEL.pkl"
OUTPUT_FILE = "15_REAL_LAB_ORAL_CANDIDATE_VALIDATION.xlsx"

AA = "ACDEFGHIKLMNPQRSTVWY"

LAB_CASES = [
    {
        "case_id": "P001",
        "title": "ACE inhibitory activity and gastrointestinal digestion stability of umami peptides IIVFGRQLL from yeast extract",
        "parent_sequence": "IIVFGRQLL",
        "reported_fragments": "IVF,VF,FGR,RQL",
        "reported_activity_outcome": "Sindirim sonrası ACE aktivitesi %39 düştü ama kaybolmadı",
        "digestion_stability_evidence": "activity_retained_after_digestion",
        "evidence_strength": "in vitro + LC-MS/MS + docking + SHR rat",
        "known_toxicity": "not_evaluated",
        "known_half_life": "not_evaluated",
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


def predict(seq, model, feature_names):
    seq = clean_sequence(seq)
    features = extract_features(seq)

    X = pd.DataFrame([features])

    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_names].fillna(0)

    prob = float(model.predict_proba(X)[0, 1])
    return round(prob, 4)


def absorption_score(fragments):
    if not fragments:
        return 40

    scores = []

    for frag in fragments:
        length = len(frag)

        if 2 <= length <= 3:
            scores.append(100)
        elif 4 <= length <= 5:
            scores.append(85)
        elif 6 <= length <= 8:
            scores.append(60)
        elif length == 1:
            scores.append(25)
        else:
            scores.append(35)

    return round(sum(scores) / len(scores), 2)


def digestion_evidence_score(evidence):
    evidence = str(evidence).lower()

    if "activity_retained" in evidence:
        return 90
    if "activity_increased" in evidence:
        return 100
    if "activity_lost" in evidence:
        return 30

    return 50


def toxicity_safety_score(value):
    value = str(value).lower()

    if value in ["non_toxic", "safe", "negative"]:
        return 90
    if value in ["toxic", "unsafe", "positive"]:
        return 10

    return 70


def stability_score(value):
    value = str(value).lower()

    if "activity_retained" in value:
        return 85
    if "stable" in value:
        return 85
    if "not_evaluated" in value:
        return 65

    return 60


def classify(score):
    if score >= 85:
        return "Tier 1 validated oral ACE candidate"
    if score >= 75:
        return "Tier 2 validated oral ACE candidate"
    if score >= 65:
        return "Tier 3 validated oral ACE candidate"
    if score >= 50:
        return "Tier 4 oral ACE candidate"
    return "Low priority oral ACE candidate"


def validate_case(case, model, feature_names):
    parent = clean_sequence(case["parent_sequence"])
    fragments = split_fragments(case["reported_fragments"])

    parent_prob = predict(parent, model, feature_names)

    fragment_rows = []
    fragment_probs = []

    for frag in fragments:
        prob = predict(frag, model, feature_names)
        fragment_probs.append(prob)

        fragment_rows.append({
            "case_id": case["case_id"],
            "sequence_type": "reported_digestive_fragment",
            "sequence": frag,
            "length": len(frag),
            "predicted_ace_probability": prob,
            "predicted_ace_percent": round(prob * 100, 2),
        })

    best_fragment_prob = max(fragment_probs) if fragment_probs else 0
    mean_fragment_prob = sum(fragment_probs) / len(fragment_probs) if fragment_probs else 0

    active_like_fragments = [
        fragments[i]
        for i, p in enumerate(fragment_probs)
        if p >= 0.55
    ]

    absorption = absorption_score(active_like_fragments if active_like_fragments else fragments)

    digestion_score = digestion_evidence_score(
        case["digestion_stability_evidence"]
    )

    tox_score = toxicity_safety_score(case["known_toxicity"])

    stab_score = stability_score(
        case["digestion_stability_evidence"]
    )

    parent_score = parent_prob * 100
    best_fragment_score = best_fragment_prob * 100
    mean_fragment_score = mean_fragment_prob * 100

    oral_candidate_score = round(
        0.15 * parent_score
        + 0.30 * best_fragment_score
        + 0.15 * mean_fragment_score
        + 0.15 * absorption
        + 0.10 * digestion_score
        + 0.10 * tox_score
        + 0.05 * stab_score,
        2
    )

    summary = {
        "case_id": case["case_id"],
        "title": case["title"],
        "parent_sequence": parent,
        "reported_fragments": ",".join(fragments),

        "parent_predicted_ace_percent": round(parent_score, 2),
        "best_fragment_predicted_ace_percent": round(best_fragment_score, 2),
        "mean_fragment_predicted_ace_percent": round(mean_fragment_score, 2),

        "active_like_fragments": ",".join(active_like_fragments),
        "absorption_score": absorption,
        "digestion_evidence_score": digestion_score,
        "toxicity_safety_score": tox_score,
        "stability_score": stab_score,

        "oral_candidate_score": oral_candidate_score,
        "candidate_class": classify(oral_candidate_score),

        "reported_activity_outcome": case["reported_activity_outcome"],
        "evidence_strength": case["evidence_strength"],

        "interpretation_tr": (
            f"{parent} parent peptidi modelde %{round(parent_score, 2)} ACE-like sinyal göstermiştir. "
            f"Sindirim sonrası bildirilen fragmentlerde en yüksek ACE-like olasılık %{round(best_fragment_score, 2)} olarak hesaplanmıştır. "
            f"Aktif benzeri fragmentler ({','.join(active_like_fragments)}) kısa/orta uzunlukta olduğu için oral emilim açısından destekleyicidir. "
            f"Literatürde sindirim sonrası aktivitenin tamamen kaybolmaması, bu peptidin digestion-aware oral candidate mantığıyla uyumlu olduğunu gösterir."
        )
    }

    parent_row = {
        "case_id": case["case_id"],
        "sequence_type": "parent",
        "sequence": parent,
        "length": len(parent),
        "predicted_ace_probability": parent_prob,
        "predicted_ace_percent": round(parent_prob * 100, 2),
    }

    detail_rows = [parent_row] + fragment_rows

    return summary, detail_rows


def main():
    bundle = joblib.load(MODEL_FILE)

    model = bundle["model"]
    feature_names = bundle["features"]

    summaries = []
    details = []

    for case in LAB_CASES:
        summary, detail_rows = validate_case(case, model, feature_names)
        summaries.append(summary)
        details.extend(detail_rows)

    summary_df = pd.DataFrame(summaries)
    detail_df = pd.DataFrame(details)

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Oral_Candidate_Summary", index=False)
        detail_df.to_excel(writer, sheet_name="Sequence_Level_Detail", index=False)

    print("\nBİTTİ")
    print("Dosya:", OUTPUT_FILE)

    print("\nORAL CANDIDATE SUMMARY")
    print(summary_df[[
        "case_id",
        "parent_sequence",
        "parent_predicted_ace_percent",
        "best_fragment_predicted_ace_percent",
        "mean_fragment_predicted_ace_percent",
        "absorption_score",
        "digestion_evidence_score",
        "oral_candidate_score",
        "candidate_class"
    ]])

    print("\nSEQUENCE DETAIL")
    print(detail_df)


if __name__ == "__main__":
    main()