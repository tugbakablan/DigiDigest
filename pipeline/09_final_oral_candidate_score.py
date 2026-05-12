import pandas as pd


DIGESTION_FILE = "03_ACE_WITH_DIGESTION_FRAGMENTS.xlsx"
PREDICTION_FILE = "08_ACE_PREDICTION_LAYER.xlsx"
OUTPUT_FILE = "09_FINAL_ORAL_CANDIDATE_SCORE.xlsx"


def safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(str(x).replace(",", "."))
    except Exception:
        return None


def safe_int(x):
    try:
        if pd.isna(x):
            return 0
        return int(float(x))
    except Exception:
        return 0


def split_list(x):
    if pd.isna(x) or str(x).strip() == "":
        return []
    return [i.strip() for i in str(x).split(",") if i.strip()]


def ic50_score(ic50):
    ic50 = safe_float(ic50)

    if ic50 is None:
        return 35
    if ic50 <= 10:
        return 100
    if ic50 <= 50:
        return 85
    if ic50 <= 100:
        return 70
    if ic50 <= 250:
        return 55
    if ic50 <= 500:
        return 40
    return 25


def probability_score(prob):
    prob = safe_float(prob)
    if prob is None:
        return 0
    return round(max(0, min(100, prob * 100)), 2)


def toxicity_safety_score(is_toxic, toxicity_score):
    is_toxic = safe_int(is_toxic)
    score = safe_float(toxicity_score)

    if is_toxic == 1:
        return 10

    if score is None:
        return 70

    if score <= -1.0:
        return 100
    if score <= -0.5:
        return 90
    if score <= 0:
        return 80

    return 55


def known_fragment_score(count):
    count = safe_int(count)

    if count == 0:
        return 0
    if count == 1:
        return 70
    if count == 2:
        return 90
    return 100


def absorption_score(fragments):
    fragments = [f for f in fragments if f]

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


def digestion_behavior_score(row):
    known_count = safe_int(row.get("known_ace_active_fragment_count"))
    best_unknown_prob = safe_float(
        row.get("best_unknown_fragment_predicted_ace_probability")
    )
    total_fragments = safe_int(row.get("digestion_fragment_count"))

    if known_count > 0:
        return 100

    if best_unknown_prob is not None and best_unknown_prob >= 0.75:
        return 85

    if best_unknown_prob is not None and best_unknown_prob >= 0.55:
        return 65

    if total_fragments > 0:
        return 45

    return 30


def uncertainty_penalty(row):
    unknown_count = safe_int(row.get("unknown_fragment_count"))
    total_count = safe_int(row.get("digestion_fragment_count"))
    best_unknown_prob = safe_float(
        row.get("best_unknown_fragment_predicted_ace_probability")
    )

    if total_count == 0:
        return 5

    ratio = unknown_count / total_count

    if best_unknown_prob is not None and best_unknown_prob >= 0.75:
        return 0

    if ratio >= 0.8:
        return 8
    if ratio >= 0.5:
        return 5
    if ratio >= 0.25:
        return 3

    return 0


def other_bioactivity_bonus(count):
    count = safe_int(count)

    if count == 0:
        return 0
    if count == 1:
        return 3
    if count == 2:
        return 5
    return 7


def classify(score):
    if score >= 80:
        return "High oral ACE candidate"
    if score >= 65:
        return "Medium-high oral ACE candidate"
    if score >= 50:
        return "Medium oral ACE candidate"
    if score >= 35:
        return "Low-medium oral ACE candidate"
    return "Low oral ACE candidate"


def evidence_level(row):
    known_count = safe_int(row.get("known_ace_active_fragment_count"))
    parent_prob = safe_float(row.get("parent_predicted_ace_probability"))
    frag_prob = safe_float(row.get("best_unknown_fragment_predicted_ace_probability"))

    if known_count > 0 and frag_prob is not None and frag_prob >= 0.75:
        return "known fragment + predicted fragment evidence"
    if known_count > 0:
        return "known BIOPEP fragment evidence"
    if frag_prob is not None and frag_prob >= 0.75:
        return "strong predicted fragment evidence"
    if parent_prob is not None and parent_prob >= 0.75:
        return "strong predicted parent evidence"
    if frag_prob is not None and frag_prob >= 0.55:
        return "medium predicted fragment evidence"
    if parent_prob is not None and parent_prob >= 0.55:
        return "medium predicted parent evidence"
    return "low evidence"


def make_explanation(row):
    parts = []

    if safe_int(row.get("known_ace_active_fragment_count")) > 0:
        parts.append("Sindirim sonrası BIOPEP’te bilinen ACE inhibitör fragment oluşuyor.")

    frag_prob = safe_float(row.get("best_unknown_fragment_predicted_ace_probability"))
    best_frag = row.get("best_predicted_unknown_fragment")

    if frag_prob is not None and frag_prob >= 0.75:
        parts.append(
            f"Bilinmeyen fragmentlerden {best_frag}, model tarafından yüksek ACE-like olasılıkla değerlendirilmiş."
        )
    elif frag_prob is not None and frag_prob >= 0.55:
        parts.append(
            f"Bilinmeyen fragmentlerden {best_frag}, modelde orta düzey ACE-like sinyal göstermiş."
        )

    parent_prob = safe_float(row.get("parent_predicted_ace_probability"))

    if parent_prob is not None and parent_prob >= 0.75:
        parts.append("Parent peptit de model tarafından yüksek ACE-like aday olarak görülüyor.")
    elif parent_prob is not None and parent_prob >= 0.55:
        parts.append("Parent peptitte orta düzey ACE-like sinyal var.")

    tox_safety = safe_float(row.get("parent_toxicity_safety_score"))

    if tox_safety is not None and tox_safety >= 80:
        parts.append("Parent toksisite riski düşük görünüyor.")
    elif tox_safety is not None and tox_safety < 50:
        parts.append("Parent toksisite riski skoru düşürüyor.")

    absorption = safe_float(row.get("absorption_feasibility_score"))

    if absorption is not None and absorption >= 80:
        parts.append("Aktif/olası aktif fragment uzunluğu oral emilim açısından avantajlı.")

    if not parts:
        parts.append("Bu peptit için güçlü oral ACE adaylık kanıtı sınırlı.")

    return " ".join(parts)


def main():
    digestion = pd.read_excel(DIGESTION_FILE)
    prediction = pd.read_excel(PREDICTION_FILE)

    digestion["sequence"] = digestion["sequence"].astype(str).str.strip().str.upper()
    prediction["sequence"] = prediction["sequence"].astype(str).str.strip().str.upper()

    df = digestion.merge(
        prediction,
        on="sequence",
        how="left"
    )

    df["parent_ic50_score"] = df["ic50_um"].apply(ic50_score)

    df["parent_predicted_ace_score"] = df[
        "parent_predicted_ace_probability"
    ].apply(probability_score)

    df["parent_toxicity_safety_score"] = df.apply(
        lambda r: toxicity_safety_score(
            r.get("is_toxic"),
            r.get("toxicity_score")
        ),
        axis=1
    )

    df["known_fragment_ace_score"] = df[
        "known_ace_active_fragment_count"
    ].apply(known_fragment_score)

    df["predicted_unknown_fragment_ace_score"] = df[
        "best_unknown_fragment_predicted_ace_probability"
    ].apply(probability_score)

    absorption_values = []

    for _, row in df.iterrows():
        known_ace = split_list(row.get("known_ace_active_fragments"))
        best_unknown = row.get("best_predicted_unknown_fragment")

        target_fragments = known_ace.copy()

        if pd.notna(best_unknown) and str(best_unknown).strip():
            target_fragments.append(str(best_unknown).strip())

        if not target_fragments:
            target_fragments = split_list(row.get("digestion_fragments"))

        absorption_values.append(absorption_score(target_fragments))

    df["absorption_feasibility_score"] = absorption_values

    df["digestion_behavior_score"] = df.apply(
        digestion_behavior_score,
        axis=1
    )

    df["other_bioactivity_bonus"] = df[
        "known_other_active_fragment_count"
    ].apply(other_bioactivity_bonus)

    df["uncertainty_penalty"] = df.apply(
        uncertainty_penalty,
        axis=1
    )

    df["oral_candidate_score"] = df.apply(
        lambda r: max(
            0,
            min(
                100,
                round(
                    0.25 * r["known_fragment_ace_score"]
                    + 0.15 * r["predicted_unknown_fragment_ace_score"]
                    + 0.15 * r["parent_predicted_ace_score"]
                    + 0.10 * r["parent_ic50_score"]
                    + 0.15 * r["parent_toxicity_safety_score"]
                    + 0.10 * r["absorption_feasibility_score"]
                    + 0.10 * r["digestion_behavior_score"]
                    + r["other_bioactivity_bonus"]
                    - r["uncertainty_penalty"],
                    2
                )
            )
        ),
        axis=1
    )

    df["candidate_class"] = df["oral_candidate_score"].apply(classify)
    df["evidence_level"] = df.apply(evidence_level, axis=1)
    df["candidate_explanation_tr"] = df.apply(make_explanation, axis=1)

    df = df.sort_values(
        by="oral_candidate_score",
        ascending=False
    )

    df.to_excel(OUTPUT_FILE, index=False)

    print("\nBİTTİ")
    print("Final candidate count:", len(df))
    print("Dosya:", OUTPUT_FILE)

    print("\nTop 20:")
    print(df[[
        "sequence",
        "known_ace_active_fragments",
        "best_predicted_unknown_fragment",
        "parent_predicted_ace_probability",
        "best_unknown_fragment_predicted_ace_probability",
        "oral_candidate_score",
        "candidate_class",
        "evidence_level"
    ]].head(20))


if __name__ == "__main__":
    main()