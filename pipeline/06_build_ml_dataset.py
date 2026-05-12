import pandas as pd


POSITIVE_FILE = "01_ACE_CANDIDATES.xlsx"
NEGATIVE_FILE = "05_NEGATIVE_NON_ACE_BIOPEP.xlsx"
OUTPUT_FILE = "06_ML_TRAINING_DATASET.xlsx"

RANDOM_STATE = 42
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
        "sequence": seq,
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

        "n_terminal": seq[0] if length else "",
        "c_terminal": seq[-1] if length else "",

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


def prepare_positive(pos):
    pos["sequence"] = pos["sequence"].apply(clean_sequence)
    pos["length"] = pos["sequence"].str.len()

    # 1 aa alma; 2-3 aa fragment ACE modeli için kalır.
    pos = pos[pos["length"] >= 2].copy()

    pos = pos.drop_duplicates("sequence").reset_index(drop=True)

    pos["is_ace"] = 1
    pos["ml_label_type"] = "BIOPEP_confirmed_ACE"
    pos["role_for_model"] = pos["length"].apply(
        lambda l: "parent_candidate_positive" if l >= 5 else "short_ACE_reference"
    )

    return pos


def prepare_negative(neg, positive_sequences):
    neg["sequence"] = neg["sequence"].apply(clean_sequence)
    neg["length"] = neg["sequence"].str.len()

    neg = neg[neg["length"] >= 2].copy()
    neg = neg.drop_duplicates("sequence").reset_index(drop=True)

    # Pozitif ACE içinde geçen sequence negatifte kalmasın.
    neg = neg[~neg["sequence"].isin(positive_sequences)].copy()

    neg["is_ace"] = 0
    neg["ml_label_type"] = "BIOPEP_non_ACE_bioactive"
    neg["role_for_model"] = neg["length"].apply(
        lambda l: "parent_candidate_negative" if l >= 5 else "short_non_ACE_reference"
    )

    return neg


def balanced_negative_sample(neg, target_n):
    if len(neg) <= target_n:
        return neg.copy().reset_index(drop=True)

    strat_col = "source_query" if "source_query" in neg.columns else "activity_type"

    sampled_groups = []
    counts = neg[strat_col].value_counts()
    base_quota = max(1, target_n // len(counts))

    for group_name in counts.index:
        group = neg[neg[strat_col] == group_name]
        take_n = min(len(group), base_quota)

        sampled_groups.append(
            group.sample(n=take_n, random_state=RANDOM_STATE)
        )

    sampled = pd.concat(sampled_groups, ignore_index=True)

    remaining = target_n - len(sampled)

    if remaining > 0:
        rest = neg[~neg["sequence"].isin(sampled["sequence"])]

        if len(rest) > 0:
            extra_n = min(remaining, len(rest))
            extra = rest.sample(n=extra_n, random_state=RANDOM_STATE)
            sampled = pd.concat([sampled, extra], ignore_index=True)

    if len(sampled) > target_n:
        sampled = sampled.sample(n=target_n, random_state=RANDOM_STATE)

    return sampled.reset_index(drop=True)


def main():
    pos = pd.read_excel(POSITIVE_FILE)
    neg = pd.read_excel(NEGATIVE_FILE)

    pos = prepare_positive(pos)
    neg = prepare_negative(neg, set(pos["sequence"]))

    n_pos = len(pos)
    neg_balanced = balanced_negative_sample(neg, n_pos)

    dataset = pd.concat(
        [pos, neg_balanced],
        ignore_index=True
    )

    dataset = dataset.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    feature_rows = []

    for _, row in dataset.iterrows():
        features = extract_features(row["sequence"])

        features["is_ace"] = int(row["is_ace"])
        features["ml_label_type"] = row.get("ml_label_type", "")
        features["role_for_model"] = row.get("role_for_model", "")

        features["source_activity_type"] = row.get("activity_type", "")
        features["source_query"] = row.get(
            "source_query",
            "ACE" if row["is_ace"] == 1 else ""
        )
        features["source_name"] = row.get("name", "")
        features["source_label"] = row.get("label_source", "BIOPEP")

        feature_rows.append(features)

    ml_df = pd.DataFrame(feature_rows)

    ml_df.to_excel(OUTPUT_FILE, index=False)

    print("\nBİTTİ")
    print("Positive ACE used:", len(pos))
    print("Negative non-ACE total:", len(neg))
    print("Negative non-ACE used:", len(neg_balanced))
    print("Final ML dataset:", len(ml_df))

    print("\nRole distribution:")
    print(ml_df["role_for_model"].value_counts())

    print("\nLabel distribution:")
    print(ml_df["is_ace"].value_counts())

    print("\nDosya:", OUTPUT_FILE)


if __name__ == "__main__":
    main()