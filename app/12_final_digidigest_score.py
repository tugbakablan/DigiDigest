import pandas as pd


BASE_FILE = "09_FINAL_ORAL_CANDIDATE_SCORE.xlsx"
STABILITY_FILE = "10_PARENT_STABILITY.xlsx"
FRAGMENT_TOX_FILE = "11_FRAGMENT_TOXICITY_LAYER.xlsx"

OUTPUT_FILE = "12_DIGIDIGEST_FINAL_RESULTS.xlsx"


def safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(str(x).replace(",", "."))
    except Exception:
        return None


def final_confidence(row):
    evidence = str(row.get("evidence_level", "")).lower()

    if (
        "known fragment" in evidence
        and safe_float(row.get("parent_predicted_ace_probability")) is not None
    ):
        return "high_confidence"

    frag_prob = safe_float(
        row.get("best_unknown_fragment_predicted_ace_probability")
    )

    if frag_prob is not None and frag_prob >= 0.80:
        return "medium_high_confidence"

    if frag_prob is not None and frag_prob >= 0.60:
        return "medium_confidence"

    return "low_confidence"


def final_class(score):
    if score >= 85:
        return "Tier 1 oral ACE candidate"
    if score >= 75:
        return "Tier 2 oral ACE candidate"
    if score >= 65:
        return "Tier 3 oral ACE candidate"
    if score >= 50:
        return "Tier 4 oral ACE candidate"
    return "Low priority oral ACE candidate"


def digestion_quality(row):
    known_frag = safe_float(row.get("known_fragment_ace_score"))
    pred_frag = safe_float(row.get("predicted_unknown_fragment_ace_score"))

    if known_frag is not None and known_frag >= 90:
        return "strong confirmed digestion release"

    if pred_frag is not None and pred_frag >= 75:
        return "strong predicted digestion release"

    if pred_frag is not None and pred_frag >= 55:
        return "moderate digestion release"

    return "limited digestion support"


def final_comment(row):
    comments = []

    score = safe_float(row.get("final_digidigest_score"))
    stability = str(row.get("stability_label", ""))
    frag_safety = str(row.get("fragment_safety_label", ""))

    if score is not None:
        if score >= 85:
            comments.append(
                "Bu peptit güçlü oral ACE inhibitör aday profili göstermektedir."
            )
        elif score >= 70:
            comments.append(
                "Bu peptit umut verici oral ACE aday özellikleri göstermektedir."
            )
        elif score >= 50:
            comments.append(
                "Bu peptit orta düzey oral ACE aday sinyali taşımaktadır."
            )

    if "supportive" in stability:
        comments.append(
            "Tahmini stabilite oral kullanım açısından destekleyici görünüyor."
        )
    elif "very_limited" in stability:
        comments.append(
            "Düşük tahmini stabilite oral kullanım potansiyelini sınırlayabilir."
        )

    if "no toxic" in frag_safety:
        comments.append(
            "Sindirim sonrası toksik fragment sinyali gözlenmedi."
        )
    elif "toxic" in frag_safety:
        comments.append(
            "Sindirim sonrası toksik fragment oluşumu dikkat gerektirebilir."
        )

    return " ".join(comments)


def main():
    base = pd.read_excel(BASE_FILE)
    stability = pd.read_excel(STABILITY_FILE)
    fragtox = pd.read_excel(FRAGMENT_TOX_FILE)

    for df in [base, stability, fragtox]:
        df["sequence"] = (
            df["sequence"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

    merged = base.merge(
        stability,
        on="sequence",
        how="left"
    )

    merged = merged.merge(
        fragtox,
        on="sequence",
        how="left"
    )

    merged["stability_adjustment"] = (
        merged["stability_adjustment"]
        .fillna(0)
    )

    merged["fragment_toxicity_penalty"] = (
        merged["fragment_toxicity_penalty"]
        .fillna(0)
    )

    merged["fragment_safety_score"] = (
        merged["fragment_safety_score"]
        .fillna(70)
    )

    merged["final_digidigest_score"] = merged.apply(
        lambda r: max(
            0,
            min(
                100,
                round(
                    r["oral_candidate_score"]
                    + r["stability_adjustment"]
                    + r["fragment_toxicity_penalty"]
                    + (
                        (r["fragment_safety_score"] - 70) * 0.05
                    ),
                    2
                )
            )
        ),
        axis=1
    )

    merged["final_confidence"] = merged.apply(
        final_confidence,
        axis=1
    )

    merged["digestion_quality"] = merged.apply(
        digestion_quality,
        axis=1
    )

    merged["final_candidate_class"] = merged[
        "final_digidigest_score"
    ].apply(final_class)

    merged["final_comment_tr"] = merged.apply(
        final_comment,
        axis=1
    )

    merged = merged.sort_values(
        by="final_digidigest_score",
        ascending=False
    ).reset_index(drop=True)

    merged.insert(
        0,
        "final_rank",
        range(1, len(merged) + 1)
    )

    keep_cols = [
        "final_rank",
        "sequence",
        "length",

        "final_digidigest_score",
        "final_candidate_class",
        "final_confidence",

        "oral_candidate_score",
        "candidate_class",
        "evidence_level",

        "parent_predicted_ace_probability",
        "best_unknown_fragment_predicted_ace_probability",

        "known_ace_active_fragments",
        "best_predicted_unknown_fragment",

        "fragment_safety_score",
        "fragment_safety_label",
        "fragment_toxicity_penalty",

        "predicted_half_life_hours",
        "stability_label",
        "stability_adjustment",

        "absorption_feasibility_score",
        "digestion_quality",

        "candidate_explanation_tr",
        "final_comment_tr",
    ]

    keep_cols = [c for c in keep_cols if c in merged.columns]

    final_df = merged[keep_cols]

    final_df.to_excel(OUTPUT_FILE, index=False)

    print("\nBİTTİ")
    print("Toplam final candidate:", len(final_df))
    print("Dosya:", OUTPUT_FILE)

    print("\nTOP 20")
    print(final_df.head(20))

    print("\nCLASS DISTRIBUTION")
    print(final_df["final_candidate_class"].value_counts())


if __name__ == "__main__":
    main()