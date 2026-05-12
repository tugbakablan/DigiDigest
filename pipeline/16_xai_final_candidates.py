import pandas as pd


INPUT_FILE = "12_DIGIDIGEST_FINAL_RESULTS.xlsx"
OUTPUT_FILE = "16_XAI_FINAL_CANDIDATE_EXPLANATIONS.xlsx"

TOP_N = 50


def safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(str(x).replace(",", "."))
    except Exception:
        return None


def explain_parent_signal(row):
    prob = safe_float(row.get("parent_predicted_ace_probability"))

    if prob is None:
        return "Parent ACE-like prediction unavailable."

    percent = round(prob * 100, 2)

    if prob >= 0.80:
        return (
            f"Parent peptide very strong ACE-like signal taşıyor "
            f"(%{percent})."
        )

    if prob >= 0.60:
        return (
            f"Parent peptide orta-yüksek ACE-like sinyal gösteriyor "
            f"(%{percent})."
        )

    if prob >= 0.40:
        return (
            f"Parent peptide zayıf-orta ACE-like sinyal gösteriyor "
            f"(%{percent})."
        )

    return (
        f"Parent peptide düşük ACE-like sinyal gösteriyor "
        f"(%{percent})."
    )


def explain_fragment_signal(row):
    frag = str(row.get("best_predicted_unknown_fragment", "")).strip()

    prob = safe_float(
        row.get("best_unknown_fragment_predicted_ace_probability")
    )

    known = str(row.get("known_ace_active_fragments", "")).strip()

    parts = []

    if known and known.lower() != "nan":
        parts.append(
            f"BIOPEP confirmed aktif ACE fragmentleri bulundu: {known}."
        )

    if frag and frag.lower() != "nan" and prob is not None:
        percent = round(prob * 100, 2)

        if prob >= 0.80:
            parts.append(
                f"{frag} fragmenti çok güçlü ACE-like sinyal verdi "
                f"(%{percent})."
            )
        elif prob >= 0.60:
            parts.append(
                f"{frag} fragmenti orta-yüksek ACE-like sinyal verdi "
                f"(%{percent})."
            )
        elif prob >= 0.40:
            parts.append(
                f"{frag} fragmenti zayıf-orta ACE-like sinyal verdi "
                f"(%{percent})."
            )

    if not parts:
        return "Belirgin güçlü digestion fragment sinyali bulunamadı."

    return " ".join(parts)


def explain_absorption(row):
    score = safe_float(
        row.get("absorption_feasibility_score")
    )

    if score is None:
        return "Absorption değerlendirmesi yok."

    if score >= 85:
        return (
            "Aktif fragment uzunlukları oral emilim açısından "
            "çok avantajlı görünüyor."
        )

    if score >= 70:
        return (
            "Aktif fragment uzunlukları oral emilim açısından "
            "destekleyici."
        )

    if score >= 50:
        return (
            "Fragment boyutları orta düzey emilim potansiyeli taşıyor."
        )

    return (
        "Fragment boyutları oral emilim açısından sınırlayıcı olabilir."
    )


def explain_stability(row):
    label = str(row.get("stability_label", "")).lower()

    hours = safe_float(row.get("predicted_half_life_hours"))

    if "supportive" in label:
        return (
            f"Predicted half-life (~{round(hours, 2)} h) "
            f"oral kullanım için destekleyici görünüyor."
        )

    if "moderate" in label:
        return (
            f"Predicted half-life (~{round(hours, 2)} h) "
            f"orta düzey stabilite gösteriyor."
        )

    if "limited" in label:
        return (
            f"Predicted half-life (~{round(hours, 2)} h) "
            f"sınırlı stabilite gösterebilir."
        )

    return "Stability verisi sınırlı."


def explain_fragment_toxicity(row):
    label = str(row.get("fragment_safety_label", "")).lower()

    toxic_frags = str(row.get("toxic_fragments", "")).strip()

    if "no toxic" in label:
        return (
            "Sindirim sonrası belirgin toksik fragment sinyali "
            "tespit edilmedi."
        )

    if "borderline" in label:
        return (
            "Bazı fragmentlerde sınırda toksisite sinyali gözlendi."
        )

    if "toxic" in label:
        return (
            f"Toksik olabilecek digestion fragmentleri bulundu: "
            f"{toxic_frags}."
        )

    return "Fragment toksisite verisi sınırlı."


def explain_final_logic(row):
    final_score = safe_float(row.get("final_digidigest_score"))

    if final_score is None:
        return "Final skor hesaplanamadı."

    if final_score >= 85:
        return (
            "Sistem bu peptiti güçlü digestion-aware oral ACE "
            "candidate olarak önceliklendirdi."
        )

    if final_score >= 75:
        return (
            "Sistem bu peptiti umut verici oral ACE candidate "
            "olarak değerlendirdi."
        )

    if final_score >= 60:
        return (
            "Peptit orta düzey oral ACE candidate sinyali taşıyor."
        )

    return (
        "Peptit düşük öncelikli oral ACE candidate olarak değerlendirildi."
    )


def build_master_explanation(row):
    sections = [
        explain_parent_signal(row),
        explain_fragment_signal(row),
        explain_absorption(row),
        explain_stability(row),
        explain_fragment_toxicity(row),
        explain_final_logic(row),
    ]

    return " ".join(sections)


def confidence_reason(row):
    confidence = str(row.get("final_confidence", "")).lower()

    if "high" in confidence:
        return (
            "Confidence yüksek çünkü sistem hem digestion "
            "fragmentleri hem ACE-like örüntüleri güçlü gördü."
        )

    if "medium" in confidence:
        return (
            "Confidence orta düzey çünkü bazı biyolojik katmanlar "
            "güçlü sinyal verirken bazıları daha sınırlı kaldı."
        )

    return (
        "Confidence düşük çünkü biyolojik kanıt katmanları "
        "sınırlı kaldı."
    )


def main():
    df = pd.read_excel(INPUT_FILE)

    df = df.sort_values(
        by="final_digidigest_score",
        ascending=False
    ).head(TOP_N)

    explanations = []

    for _, row in df.iterrows():
        explanations.append({
            "final_rank": row.get("final_rank"),
            "sequence": row.get("sequence"),

            "final_digidigest_score":
                row.get("final_digidigest_score"),

            "final_candidate_class":
                row.get("final_candidate_class"),

            "final_confidence":
                row.get("final_confidence"),

            "xai_parent_signal":
                explain_parent_signal(row),

            "xai_fragment_signal":
                explain_fragment_signal(row),

            "xai_absorption":
                explain_absorption(row),

            "xai_stability":
                explain_stability(row),

            "xai_fragment_toxicity":
                explain_fragment_toxicity(row),

            "xai_final_logic":
                explain_final_logic(row),

            "xai_confidence_reason":
                confidence_reason(row),

            "xai_master_explanation":
                build_master_explanation(row),
        })

    out = pd.DataFrame(explanations)

    out.to_excel(OUTPUT_FILE, index=False)

    print("\nBİTTİ")
    print("Top explained candidates:", len(out))
    print("Dosya:", OUTPUT_FILE)

    print("\nTOP 10")
    print(out[[
        "final_rank",
        "sequence",
        "final_digidigest_score",
        "final_candidate_class"
    ]].head(10))


if __name__ == "__main__":
    main()