import re
import time
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://biochemia.uwm.edu.pl/biopep/"
OUTPUT_FILE = "05_NEGATIVE_NON_ACE_BIOPEP.xlsx"

HEADERS = {"User-Agent": "Mozilla/5.0"}
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

NEGATIVE_ACTIVITIES = [
    "dpp",
    "ao",
    "ab",
    "af",
    "op",
    "im",
    "at",
    "ren",
    "um",
    "glui",
    "hypc",
    "hypl",
    "neup",
    "xox",
]


def clean_sequence(seq):
    seq = str(seq).strip().upper()
    return "".join(c for c in seq if c in AMINO_ACIDS)


def safe_float(x):
    if x is None:
        return None

    x = str(x).replace(",", ".").strip()
    match = re.search(r"-?\d+(\.\d+)?", x)

    if not match:
        return None

    return float(match.group(0))


def build_search_url(activity_keyword):
    return (
        "https://biochemia.uwm.edu.pl/biopep/"
        f"peptide_data_search.php?txt_search={activity_keyword}"
        "&menu_search=activity&button12.x=16&button12.y=7"
    )


def extract_id_from_url(url):
    match = re.search(r"zm_ID=(\d+)", str(url))

    if match:
        return match.group(1)

    return None


def parse_search_page(activity_keyword):
    url = build_search_url(activity_keyword)

    print(f"\nBIOPEP non-ACE aranıyor: {activity_keyword}")
    print(url)

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=60
        )
        response.raise_for_status()

    except Exception as e:
        print("Sayfa hatası:", activity_keyword, e)
        return pd.DataFrame()

    soup = BeautifulSoup(response.text, "lxml")
    rows = []

    for tr in soup.find_all("tr"):
        cells = tr.find_all("td")

        if len(cells) < 7:
            continue

        link = cells[0].find("a")

        if not link:
            continue

        href = link.get("href", "")

        if "peptide_data_page1.php" not in href:
            continue

        detail_url = urljoin(BASE_URL, href)
        cols = [c.get_text(" ", strip=True) for c in cells]

        external_id = extract_id_from_url(detail_url)

        if not external_id and len(cols) > 1:
            external_id = cols[1]

        name = cols[2] if len(cols) > 2 else ""
        sequence = clean_sequence(cols[3]) if len(cols) > 3 else ""
        chemical_mass = safe_float(cols[4]) if len(cols) > 4 else None
        monoisotopic_mass = safe_float(cols[5]) if len(cols) > 5 else None
        activity_type = cols[6] if len(cols) > 6 else activity_keyword

        if not sequence:
            continue

        full_text = f"{name} {activity_type}".lower()

        # ACE içeren hiçbir şeyi negatif etikete sokma.
        if "ace inhibitor" in full_text or "ace-inhibitor" in full_text:
            continue

        rows.append({
            "external_id": external_id,
            "sequence": sequence,
            "length": len(sequence),
            "activity_type": activity_type,
            "name": name,
            "chemical_mass": chemical_mass,
            "monoisotopic_mass": monoisotopic_mass,
            "source_query": activity_keyword,
            "label_source": "BIOPEP",
            "detail_url": detail_url,
            "ml_negative_label": 1,
            "is_ace": 0,
            "negative_label_type": "non_ACE_bioactive",
        })

    df = pd.DataFrame(rows)

    if df.empty:
        print("Kayıt yok:", activity_keyword)
        return df

    df = df.drop_duplicates(
        subset=["external_id", "sequence"]
    ).reset_index(drop=True)

    print("Gelen non-ACE kayıt:", len(df))

    return df


def main():
    all_dfs = []

    for activity in NEGATIVE_ACTIVITIES:
        df = parse_search_page(activity)

        if not df.empty:
            all_dfs.append(df)

        time.sleep(2)

    if not all_dfs:
        print("Hiç non-ACE kayıt çekilemedi.")
        return

    neg = pd.concat(all_dfs, ignore_index=True)

    neg["sequence"] = neg["sequence"].apply(clean_sequence)
    neg["length"] = neg["sequence"].str.len()

    # Tek aminoasitleri model eğitimine alma.
    # 2-3 aa kalır çünkü fragment ACE modeli için kısa peptit evreni önemli.
    neg = neg[neg["length"] >= 2].copy()

    neg = neg.drop_duplicates(
        subset=["sequence"]
    ).reset_index(drop=True)

    neg.insert(0, "negative_id", range(1, len(neg) + 1))

    neg["candidate_role"] = neg["length"].apply(
        lambda l: "parent_candidate_negative" if l >= 5 else "short_non_ACE_reference"
    )

    keep_cols = [
        "negative_id",
        "external_id",
        "sequence",
        "length",
        "candidate_role",
        "activity_type",
        "source_query",
        "is_ace",
        "ml_negative_label",
        "negative_label_type",
        "name",
        "chemical_mass",
        "monoisotopic_mass",
        "label_source",
        "detail_url",
    ]

    neg = neg[keep_cols]

    neg.to_excel(OUTPUT_FILE, index=False)

    print("\nBİTTİ")
    print("Toplam non-ACE kayıt:", len(neg))
    print("5+ parent negative:", (neg["length"] >= 5).sum())
    print("2-4 short non-ACE reference:", ((neg["length"] >= 2) & (neg["length"] < 5)).sum())
    print("Dosya:", OUTPUT_FILE)


if __name__ == "__main__":
    main()