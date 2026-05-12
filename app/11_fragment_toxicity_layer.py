import re
import time
from io import StringIO

import pandas as pd
from playwright.sync_api import sync_playwright


INPUT_FILE = "09_FINAL_ORAL_CANDIDATE_SCORE.xlsx"
OUTPUT_FILE = "11_FRAGMENT_TOXICITY_LAYER.xlsx"
RAW_OUTPUT_FILE = "11_FRAGMENT_TOXICITY_RAW_RESULTS.xlsx"
DEBUG_HTML = "fragment_toxinpred_debug.html"

TOXINPRED_URL = "https://webs.iiitd.edu.in/raghava/toxinpred/multi_submit.php"
BATCH_SIZE = 50

AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(seq):
    seq = str(seq).strip().upper()
    return "".join(c for c in seq if c in AA)


def split_fragments(x):
    if pd.isna(x) or str(x).strip() == "":
        return []
    return [clean_sequence(i) for i in str(x).split(",") if clean_sequence(i)]


def parse_float(value):
    try:
        value = str(value).replace(",", ".").strip()
        match = re.search(r"-?\d+(\.\d+)?", value)
        return float(match.group(0)) if match else None
    except Exception:
        return None


def make_input(seqs):
    return "\n".join(seqs)


def safe_page_content(page, retries=5):
    for i in range(retries):
        try:
            page.wait_for_load_state("domcontentloaded", timeout=30000)
            page.wait_for_timeout(3000)
            return page.content()
        except Exception as e:
            print(f"Content retry {i + 1}/{retries}: {e}")
            page.wait_for_timeout(5000)
    return page.content()


def submit_toxinpred(page, seqs):
    page.goto(
        TOXINPRED_URL,
        wait_until="domcontentloaded",
        timeout=120000
    )

    page.wait_for_timeout(3000)

    page.locator("textarea").first.fill(make_input(seqs))

    try:
        page.get_by_label("SVM (Swiss-Prot) based").check(timeout=3000)
    except Exception:
        try:
            page.locator('input[type="radio"]').nth(0).check(timeout=3000)
        except Exception:
            pass

    submits = page.locator('input[type="submit"]')

    try:
        with page.expect_navigation(timeout=180000):
            submits.nth(0).click(force=True, timeout=10000)
    except Exception:
        print("Navigation yakalanamadı, sonuç sayfası ayrıca bekleniyor...")

    page.wait_for_timeout(25000)

    return safe_page_content(page)


def parse_toxinpred_results(html):
    results = []

    try:
        tables = pd.read_html(StringIO(html))
    except Exception:
        return results

    for table in tables:
        table.columns = [str(c).strip() for c in table.columns]
        lower_cols = [c.lower() for c in table.columns]

        has_seq = any("sequence" in c for c in lower_cols)
        has_pred = any("prediction" in c for c in lower_cols)
        has_score = any("score" in c for c in lower_cols)

        if not (has_seq and has_pred):
            continue

        seq_col = None
        pred_col = None
        score_col = None

        for col in table.columns:
            c = str(col).lower()

            if "sequence" in c:
                seq_col = col
            elif "prediction" in c:
                pred_col = col
            elif "svm" in c and "score" in c:
                score_col = col
            elif score_col is None and "score" in c:
                score_col = col

        if seq_col is None or pred_col is None:
            continue

        for _, row in table.iterrows():
            seq = clean_sequence(row.get(seq_col))
            pred = str(row.get(pred_col)).strip()
            score = parse_float(row.get(score_col)) if score_col else None

            if not seq:
                continue

            pred_clean = pred.lower().replace("-", "").replace(" ", "")
            is_toxic = 1 if pred_clean in ["toxin", "toxic"] else 0

            results.append({
                "fragment_sequence": seq,
                "fragment_toxicity_prediction": pred,
                "fragment_toxicity_score": score,
                "fragment_is_toxic": is_toxic,
                "fragment_toxicity_method": "ToxinPred SVM Swiss-Prot"
            })

    return results


def fragment_safety_penalty(toxic_count, total_count, max_score):
    if total_count == 0:
        return 0

    toxic_ratio = toxic_count / total_count

    if toxic_ratio >= 0.75:
        return -20
    if toxic_ratio >= 0.50:
        return -15
    if toxic_ratio >= 0.25:
        return -8
    if toxic_count >= 1:
        return -5

    if max_score is not None and max_score > -0.05:
        return -3

    return 0


def fragment_safety_score(toxic_count, total_count, max_score):
    if total_count == 0:
        return 70

    safe_ratio = (total_count - toxic_count) / total_count
    base = safe_ratio * 100

    if toxic_count == 0 and max_score is not None and max_score > -0.05:
        base -= 5

    return round(max(0, min(100, base)), 2)


def fragment_safety_label(toxic_count, total_count, max_score):
    if total_count == 0:
        return "no digestion fragment evaluated"
    if toxic_count > 0:
        return "toxic digestion fragment detected"
    if max_score is not None and max_score > -0.05:
        return "borderline fragment toxicity signal"
    return "no toxic digestion fragment detected"


def collect_fragments(df):
    fragments = set()

    for _, row in df.iterrows():
        frags = []

        frags += split_fragments(row.get("digestion_fragments"))
        frags += split_fragments(row.get("known_ace_active_fragments"))
        frags += split_fragments(row.get("known_other_active_fragments"))

        best_unknown = row.get("best_predicted_unknown_fragment")

        if pd.notna(best_unknown) and str(best_unknown).strip():
            frags.append(clean_sequence(best_unknown))

        for frag in frags:
            if len(frag) >= 2:
                fragments.add(frag)

    return sorted(fragments)


def main():
    df = pd.read_excel(INPUT_FILE)

    unique_fragments = collect_fragments(df)

    print("Unique fragment count:", len(unique_fragments))

    if not unique_fragments:
        print("Fragment bulunamadı.")
        return

    all_results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        for start in range(0, len(unique_fragments), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(unique_fragments))
            batch = unique_fragments[start:end]

            print(f"\nBatch {start + 1} -> {end}")

            try:
                html = submit_toxinpred(page, batch)

                with open(DEBUG_HTML, "w", encoding="utf-8") as f:
                    f.write(html)

                batch_results = parse_toxinpred_results(html)

                print("Rows found:", len(batch_results))

                all_results.extend(batch_results)

                temp = pd.DataFrame(all_results)

                if not temp.empty:
                    temp = temp.drop_duplicates("fragment_sequence")
                    temp.to_excel(RAW_OUTPUT_FILE, index=False)
                    print("Ara kayıt alındı.")

            except Exception as e:
                print("Batch hata:", e)

            time.sleep(3)

        browser.close()

    raw = pd.DataFrame(all_results)

    if raw.empty:
        print("Hiç fragment toxicity sonucu okunamadı.")
        print("Debug dosyasını kontrol et:", DEBUG_HTML)
        return

    raw = raw.drop_duplicates("fragment_sequence")
    raw.to_excel(RAW_OUTPUT_FILE, index=False)

    tox_map = raw.set_index("fragment_sequence").to_dict("index")

    rows = []

    for _, row in df.iterrows():
        parent_seq = clean_sequence(row.get("sequence"))

        frags = []

        frags += split_fragments(row.get("known_ace_active_fragments"))
        frags += split_fragments(row.get("known_other_active_fragments"))

        best_unknown = row.get("best_predicted_unknown_fragment")

        if pd.notna(best_unknown) and str(best_unknown).strip():
            frags.append(clean_sequence(best_unknown))

        frags = sorted(set(f for f in frags if len(f) >= 2))

        toxic_fragments = []
        safe_fragments = []
        scores = []

        for frag in frags:
            info = tox_map.get(frag)

            if not info:
                continue

            score = info.get("fragment_toxicity_score")

            if score is not None and not pd.isna(score):
                scores.append(score)

            if int(info.get("fragment_is_toxic", 0)) == 1:
                toxic_fragments.append(frag)
            else:
                safe_fragments.append(frag)

        total_count = len(frags)
        toxic_count = len(set(toxic_fragments))
        max_score = max(scores) if scores else None

        rows.append({
            "sequence": parent_seq,
            "evaluated_fragment_count": total_count,
            "toxic_fragment_count": toxic_count,
            "safe_fragment_count": len(set(safe_fragments)),
            "toxic_fragments": ",".join(sorted(set(toxic_fragments))),
            "safe_fragments": ",".join(sorted(set(safe_fragments))),
            "max_fragment_toxicity_score": max_score,
            "fragment_safety_score": fragment_safety_score(
                toxic_count,
                total_count,
                max_score
            ),
            "fragment_safety_label": fragment_safety_label(
                toxic_count,
                total_count,
                max_score
            ),
            "fragment_toxicity_penalty": fragment_safety_penalty(
                toxic_count,
                total_count,
                max_score
            ),
        })

    out = pd.DataFrame(rows)

    out.to_excel(OUTPUT_FILE, index=False)

    print("\nBİTTİ")
    print("Raw fragment toxicity:", RAW_OUTPUT_FILE)
    print("Parent-level fragment toxicity:", OUTPUT_FILE)

    print("\nTop 20 güvenli:")
    print(
        out.sort_values(
            "fragment_safety_score",
            ascending=False
        ).head(20)
    )


if __name__ == "__main__":
    main()