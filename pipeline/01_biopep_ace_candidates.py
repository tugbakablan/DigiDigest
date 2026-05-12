import re
import time
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://biochemia.uwm.edu.pl/biopep/"
SEARCH_URL = (
    "https://biochemia.uwm.edu.pl/biopep/"
    "peptide_data_search.php?txt_search=ACE&menu_search=activity&button12.x=16&button12.y=7"
)

OUTPUT_FILE = "01_ACE_CANDIDATES.xlsx"

HEADERS = {"User-Agent": "Mozilla/5.0"}
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


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


def get_soup(url, retries=3):
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                headers=HEADERS,
                timeout=60
            )
            response.raise_for_status()
            return BeautifulSoup(response.text, "lxml"), response.text

        except Exception as e:
            print(f"Retry {attempt + 1}/{retries} | {url} | {e}")
            time.sleep(3)

    return None, ""


def extract_ic50_from_detail(html):
    soup = BeautifulSoup(html, "lxml")

    for select in soup.find_all("select"):
        if "IC50" not in select.get_text(" ", strip=True):
            continue

        parent_row = select.find_parent("tr")

        if parent_row:
            inputs = parent_row.find_all("input")
            values = [safe_float(inp.get("value")) for inp in inputs]
            values = [v for v in values if v is not None]

            if values:
                return values[-1]

        next_input = select.find_next("input")

        if next_input:
            return safe_float(next_input.get("value"))

    return None


def parse_detail_page(detail_url):
    soup, html = get_soup(detail_url)

    if soup is None:
        return {
            "sequence_detail": None,
            "length_detail": None,
            "ic50_um": None,
        }

    text = soup.get_text(" ", strip=True)

    sequence_detail = None
    textarea = soup.find("textarea")

    if textarea:
        sequence_detail = clean_sequence(
            textarea.get_text(" ", strip=True)
        )

    length_detail = None
    match = re.search(
        r"Number of amino acid residues\s+(\d+)",
        text,
        re.I
    )

    if match:
        length_detail = int(match.group(1))

    ic50_um = extract_ic50_from_detail(html)

    return {
        "sequence_detail": sequence_detail,
        "length_detail": length_detail,
        "ic50_um": ic50_um,
    }


def extract_external_id(cols, detail_url):
    match = re.search(r"zm_ID=(\d+)", str(detail_url))

    if match:
        return match.group(1)

    if len(cols) > 1:
        return str(cols[1]).strip()

    return None


def parse_search_page():
    soup, html = get_soup(SEARCH_URL)

    if soup is None:
        raise RuntimeError("BIOPEP ACE search sayfası açılamadı.")

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
        cols = [cell.get_text(" ", strip=True) for cell in cells]

        external_id = extract_external_id(cols, detail_url)
        name = cols[2] if len(cols) > 2 else ""
        sequence = clean_sequence(cols[3]) if len(cols) > 3 else ""
        chemical_mass = safe_float(cols[4]) if len(cols) > 4 else None
        monoisotopic_mass = safe_float(cols[5]) if len(cols) > 5 else None
        activity_type = cols[6] if len(cols) > 6 else "ACE inhibitor"

        if not sequence:
            continue

        rows.append({
            "external_id": external_id,
            "sequence": sequence,
            "length": len(sequence),
            "activity_type": "ACE_Inhibitor",
            "raw_activity_type": activity_type,
            "name": name,
            "chemical_mass": chemical_mass,
            "monoisotopic_mass": monoisotopic_mass,
            "detail_url": detail_url,
            "parent_is_biopep_ace": 1,
            "parent_ace_source": "BIOPEP",
            "label_source": "BIOPEP",
        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("BIOPEP search sonucundan kayıt çekilemedi.")

    df = df.drop_duplicates(
        subset=["external_id", "sequence"]
    ).reset_index(drop=True)

    print("Search sayfasından gelen kayıt:", len(df))

    return df


def main():
    df = parse_search_page()

    detail_rows = []

    for i, row in df.iterrows():
        print(
            f"{i + 1}/{len(df)} | "
            f"ID={row['external_id']} | {row['sequence']}"
        )

        detail = parse_detail_page(row["detail_url"])

        detail_rows.append({
            "external_id": row["external_id"],
            **detail
        })

        if (i + 1) % 50 == 0:
            temp_detail = pd.DataFrame(detail_rows)
            temp = df.merge(
                temp_detail,
                on="external_id",
                how="left"
            )
            temp.to_excel(OUTPUT_FILE, index=False)
            print("Ara kayıt alındı.")

        time.sleep(0.25)

    detail_df = pd.DataFrame(detail_rows)

    merged = df.merge(
        detail_df,
        on="external_id",
        how="left"
    )

    merged["sequence"] = merged["sequence_detail"].fillna(
        merged["sequence"]
    )

    merged["sequence"] = merged["sequence"].apply(clean_sequence)

    merged["length"] = merged["length_detail"].fillna(
        merged["length"]
    )

    merged["length"] = merged["length"].astype(int)

    merged["parent_candidate_eligible"] = (
        merged["length"] >= 5
    ).astype(int)

    merged["ml_positive_label"] = 1

    merged["candidate_role"] = merged["parent_candidate_eligible"].apply(
        lambda x: "parent_candidate" if x == 1 else "short_active_reference"
    )

    merged = merged.drop_duplicates(
        subset=["sequence"]
    ).reset_index(drop=True)

    merged.insert(0, "peptide_id", range(1, len(merged) + 1))

    keep_cols = [
        "peptide_id",
        "external_id",
        "sequence",
        "length",
        "parent_candidate_eligible",
        "candidate_role",
        "activity_type",
        "parent_is_biopep_ace",
        "parent_ace_source",
        "ml_positive_label",
        "label_source",
        "ic50_um",
        "name",
        "chemical_mass",
        "monoisotopic_mass",
        "raw_activity_type",
        "detail_url",
    ]

    merged = merged[keep_cols]

    merged.to_excel(OUTPUT_FILE, index=False)

    print("\nBİTTİ")
    print("Toplam ACE kayıt:", len(merged))
    print("Parent candidate eligible:", merged["parent_candidate_eligible"].sum())
    print("Short active reference:", (merged["parent_candidate_eligible"] == 0).sum())
    print("Dosya:", OUTPUT_FILE)


if __name__ == "__main__":
    main()