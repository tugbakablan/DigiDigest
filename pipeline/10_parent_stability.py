import time
import pandas as pd
from playwright.sync_api import sync_playwright


INPUT_FILE = "09_FINAL_ORAL_CANDIDATE_SCORE.xlsx"
OUTPUT_FILE = "10_PARENT_STABILITY.xlsx"

URL = "https://webs.iiitd.edu.in/raghava/plifepred/batch.php"


def clean_sequence(seq):
    return str(seq).strip().upper()


def parse_half_life_seconds(value):
    try:
        seconds = float(str(value).strip())
        return seconds / 3600
    except Exception:
        return None


def stability_label(hours):
    if hours is None:
        return "stability_not_available"
    if hours >= 1:
        return "supportive_stability"
    if hours >= 0.25:
        return "moderate_stability"
    if hours >= 0.05:
        return "limited_stability"
    return "very_limited_stability"


def stability_adjustment(hours):
    if hours is None:
        return 0
    if hours >= 1:
        return 3
    if hours >= 0.25:
        return 1.5
    if hours >= 0.05:
        return -1
    return -2


def main():
    df = pd.read_excel(INPUT_FILE)

    sequences = []

    for seq in df["sequence"].dropna().unique():
        seq = clean_sequence(seq)

        if 5 <= len(seq) <= 50:
            sequences.append(seq)

    print("PlifePred'e gönderilecek parent sequence:", len(sequences))

    if not sequences:
        print("Uygun sequence yok.")
        return

    fasta_text = "\n".join(sequences)
    rows = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        page.goto(URL, timeout=120000)
        page.wait_for_timeout(3000)

        page.fill("textarea", fasta_text)

        labels = [
            "Hydrophobicity",
            "Hydropathicity",
            "Molecular weight",
            "Charge",
            "Hydrophilicity",
        ]

        for label in labels:
            try:
                page.get_by_label(label).check()
            except Exception:
                pass

        print("Run Analysis tıklanıyor...")
        page.locator('input[value="Run Analysis!"]').click()

        time.sleep(15)

        html = page.content()

        with open("plifepred_debug.html", "w", encoding="utf-8") as f:
            f.write(html)

        try:
            tables = pd.read_html(html)
        except Exception:
            print("Tablo okunamadı. plifepred_debug.html dosyasını kontrol et.")
            browser.close()
            return

        print("Bulunan tablo sayısı:", len(tables))

        target_table = None

        for table in tables:
            joined_cols = " ".join([str(c).lower() for c in table.columns])

            if "half" in joined_cols or "life" in joined_cols:
                target_table = table
                break

        if target_table is None:
            print("Half-life tablosu bulunamadı.")
            browser.close()
            return

        print(target_table.head())

        for _, row in target_table.iterrows():
            row_text = " ".join(str(x) for x in row.values)

            sequence = None

            for seq in sequences:
                if seq in row_text:
                    sequence = seq
                    break

            half_life_raw = None

            for value in row.values:
                try:
                    numeric = float(value)

                    if numeric > 0:
                        half_life_raw = numeric
                        break
                except Exception:
                    pass

            if sequence:
                hours = parse_half_life_seconds(half_life_raw)

                rows.append({
                    "sequence": sequence,
                    "predicted_half_life_raw_seconds": half_life_raw,
                    "predicted_half_life_hours": hours,
                    "stability_label": stability_label(hours),
                    "stability_adjustment": stability_adjustment(hours),
                    "stability_source": "PlifePred",
                })

        browser.close()

    out = pd.DataFrame(rows)

    out.to_excel(OUTPUT_FILE, index=False)

    print("\nBİTTİ")
    print("Stability kayıt:", len(out))
    print("Dosya:", OUTPUT_FILE)

    if not out.empty:
        print(out.sort_values("predicted_half_life_hours", ascending=False).head(10))


if __name__ == "__main__":
    main()