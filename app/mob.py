import joblib
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from urllib.parse import urlencode


MODEL_FILE = "07_ACE_RANDOM_FOREST_MODEL.pkl"
FINAL_FILE = "12_DIGIDIGEST_FINAL_RESULTS.xlsx"

BIOPEP_REPORT_URL = "https://biochemia.uwm.edu.pl/biopep/report_cutting_for_seq.php"
BIOPEP_ACTIVE_URL = "https://biochemia.uwm.edu.pl/biopep/report_finding1.php"
HEADERS = {"User-Agent": "Mozilla/5.0"}

AA = "ACDEFGHIKLMNPQRSTVWY"

HYDROPHOBIC = set("AILMFWYV")
AROMATIC = set("FWY")
POSITIVE = set("KRH")
NEGATIVE = set("DE")
POLAR = set("QNST")
ALIPHATIC = set("ILVA")
PROLINE = set("P")


st.set_page_config(
    page_title="DigiDigest",
    page_icon="🧪",
    layout="wide"
)


st.markdown("""
<style>
.stApp {
    background: #F5F7FA;
    color: #111827;
}

.hero {
    background: linear-gradient(135deg, #0F172A, #1E3A8A);
    color: white;
    padding: 34px;
    border-radius: 24px;
    margin-bottom: 28px;
}

.hero h1 {
    font-size: 48px;
    margin-bottom: 8px;
}

.hero p {
    color: #DBEAFE;
    font-size: 17px;
    max-width: 980px;
}

.input-card, .analysis-card, .table-card {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 22px;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
}

.metric-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    min-height: 120px;
}

.metric-label {
    color: #64748B;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 8px;
}

.metric-value {
    color: #0F172A;
    font-size: 34px;
    font-weight: 800;
}

.metric-caption {
    color: #64748B;
    font-size: 13px;
    margin-top: 6px;
}

.section-title {
    font-size: 25px;
    font-weight: 800;
    color: #0F172A;
    margin-top: 20px;
    margin-bottom: 12px;
}

.subtle {
    color: #64748B;
    font-size: 14px;
}

.xai-box {
    background: #F8FAFC;
    border-left: 5px solid #2563EB;
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 16px;
}

.xai-title {
    color: #1E3A8A;
    font-weight: 800;
    font-size: 18px;
    margin-bottom: 8px;
}

.good {
    color: #047857;
    font-weight: 800;
}

.mid {
    color: #B45309;
    font-weight: 800;
}

.low {
    color: #B91C1C;
    font-weight: 800;
}

.badge {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 800;
    margin-bottom: 10px;
}

.badge-high {
    background: #DCFCE7;
    color: #166534;
}

.badge-mid {
    background: #FEF3C7;
    color: #92400E;
}

.badge-low {
    background: #FEE2E2;
    color: #991B1B;
}

hr {
    margin-top: 24px;
    margin-bottom: 24px;
}
</style>
""", unsafe_allow_html=True)


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


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_FILE)
    return bundle["model"], bundle["features"]


@st.cache_data
def load_final_df():
    try:
        df = pd.read_excel(FINAL_FILE)
        df["sequence"] = df["sequence"].astype(str).str.upper()
        return df
    except Exception:
        return pd.DataFrame()


def predict_ace(seq, model, feature_names):
    features = extract_features(seq)
    X = pd.DataFrame([features])

    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_names].fillna(0)
    prob = float(model.predict_proba(X)[0, 1])
    return prob, features


def biological_reason(seq, features):
    reasons = []

    if features.get("c_terminal_hydrophobic") == 1:
        reasons.append(
            "The peptide has a hydrophobic C-terminal residue, a pattern frequently observed in ACE-inhibitory peptides."
        )

    if features.get("has_aromatic") == 1:
        reasons.append(
            "Aromatic residues may support interaction potential with ACE-like peptide patterns."
        )

    if features.get("has_proline") == 1:
        reasons.append(
            "Proline presence can support ACE-like sequence patterns and may contribute to digestion stability."
        )

    if features.get("hydrophobic_ratio", 0) >= 0.50:
        reasons.append(
            f"The hydrophobic ratio is {features['hydrophobic_ratio']:.2f}, which supports ACE-like behavior."
        )

    if len(seq) <= 4:
        reasons.append(
            "The fragment is short, which may support oral absorption feasibility in this MVP scoring logic."
        )

    if not reasons:
        reasons.append(
            "The model decision is based on multiple weak sequence-level signals rather than one dominant feature."
        )

    return reasons[:4]


def build_digest_url(sequence):
    params = {
        "txt_seq_e": "",
        "txt_seq": sequence,
        "enz1": "13",
        "enz2": "12",
        "enz3": "11",
        "but_report.x": "109",
        "but_report.y": "11",
        "prot": "",
        "e2": "",
        "e3": "",
    }
    return BIOPEP_REPORT_URL + "?" + urlencode(params)


def biopep_digest(sequence):
    url = build_digest_url(sequence)

    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")
    form = soup.find("form", {"name": "find_pep"})

    if not form:
        return [], [], [], [], url

    seq_input = form.find("input", {"name": "seq"})
    raw_fragments = seq_input.get("value", "") if seq_input else ""

    fragments = [clean_sequence(x) for x in raw_fragments.split("-")]
    fragments = [f for f in fragments if f]

    payload = {}

    for inp in form.find_all("input"):
        name = inp.get("name")
        value = inp.get("value", "")
        if name:
            payload[name] = value

    known_ace = []
    known_other = []

    try:
        ar = requests.post(
            BIOPEP_ACTIVE_URL,
            data=payload,
            headers=HEADERS,
            timeout=60
        )
        ar.raise_for_status()

        active_soup = BeautifulSoup(ar.text, "lxml")

        for tr in active_soup.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]

            if len(cells) < 7:
                continue

            frag = clean_sequence(cells[2])
            full_text = " ".join(cells).lower()

            if not frag:
                continue

            if "ace inhibitor" in full_text:
                known_ace.append(frag)
            elif any(x in full_text for x in ["dpp", "antioxidant", "antimicrobial", "renin"]):
                known_other.append(frag)

    except Exception:
        pass

    known_active = set(known_ace + known_other)
    unknown = [f for f in fragments if f not in known_active]

    return fragments, sorted(set(known_ace)), sorted(set(known_other)), sorted(set(unknown)), url


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


def live_oral_score(parent_prob, known_ace, unknown_predictions, fragments):
    parent_score = parent_prob * 100

    if len(known_ace) >= 3:
        known_score = 100
    elif len(known_ace) == 2:
        known_score = 90
    elif len(known_ace) == 1:
        known_score = 70
    else:
        known_score = 0

    best_unknown_prob = max(unknown_predictions.values()) if unknown_predictions else 0
    unknown_score = best_unknown_prob * 100

    target_fragments = known_ace.copy()

    if unknown_predictions:
        best_frag = max(unknown_predictions, key=unknown_predictions.get)
        target_fragments.append(best_frag)

    if not target_fragments:
        target_fragments = fragments

    absorption = absorption_score(target_fragments)

    if known_ace:
        digestion_score = 100
    elif best_unknown_prob >= 0.75:
        digestion_score = 85
    elif best_unknown_prob >= 0.55:
        digestion_score = 65
    elif fragments:
        digestion_score = 45
    else:
        digestion_score = 30

    score = (
        0.20 * parent_score
        + 0.30 * known_score
        + 0.25 * unknown_score
        + 0.15 * absorption
        + 0.10 * digestion_score
    )

    return round(max(0, min(100, score)), 2), absorption, best_unknown_prob


def candidate_class(score):
    if score >= 85:
        return "Tier 1 oral ACE candidate"
    if score >= 75:
        return "Tier 2 oral ACE candidate"
    if score >= 65:
        return "Tier 3 oral ACE candidate"
    if score >= 50:
        return "Tier 4 oral ACE candidate"
    return "Low priority candidate"


def interpretation(score):
    if score >= 80:
        return "This sequence shows strong oral ACE candidate potential."
    if score >= 65:
        return "This sequence shows medium-high oral candidate potential."
    if score >= 50:
        return "This sequence shows moderate oral candidate signal."
    return "This sequence has limited oral candidate support in the live scoring mode."


def analyse_sequence(seq, model, feature_names):
    parent_prob, parent_features = predict_ace(seq, model, feature_names)
    parent_reasons = biological_reason(seq, parent_features)

    fragments, known_ace, known_other, unknowns, digest_url = biopep_digest(seq)

    unknown_predictions = {}
    unknown_reasons = {}

    for frag in unknowns:
        if len(frag) >= 2:
            prob, features = predict_ace(frag, model, feature_names)
            unknown_predictions[frag] = prob
            unknown_reasons[frag] = biological_reason(frag, features)

    live_score, absorption, best_unknown_prob = live_oral_score(
        parent_prob,
        known_ace,
        unknown_predictions,
        fragments
    )

    best_unknown_fragment = None
    best_unknown_reasons = []

    if unknown_predictions:
        best_unknown_fragment = max(unknown_predictions, key=unknown_predictions.get)
        best_unknown_reasons = unknown_reasons.get(best_unknown_fragment, [])

    result = {
        "sequence": seq,
        "parent_ace_percent": round(parent_prob * 100, 2),
        "parent_reasons": parent_reasons,

        "digestion_fragments": fragments,
        "known_ace_fragments": known_ace,
        "known_other_fragments": known_other,
        "unknown_fragments": unknowns,
        "digest_url": digest_url,

        "unknown_predictions": unknown_predictions,
        "unknown_reasons": unknown_reasons,
        "best_unknown_fragment": best_unknown_fragment,
        "best_unknown_percent": round(best_unknown_prob * 100, 2),
        "best_unknown_reasons": best_unknown_reasons,

        "absorption_score": absorption,
        "live_oral_score": live_score,
        "candidate_class": candidate_class(live_score),
        "interpretation": interpretation(live_score),
    }

    return result


def add_to_history(result):
    st.session_state.history = [
        r for r in st.session_state.history
        if r["sequence"] != result["sequence"]
    ]
    st.session_state.history.append(result)


def metric_card(label, value, caption=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_analysis(result):
    score = result["live_oral_score"]
    badge_class = "badge-high" if score >= 80 else "badge-mid" if score >= 60 else "badge-low"

    st.markdown(
        f"""
        <div class="analysis-card">
            <span class="badge {badge_class}">{result['candidate_class']}</span>
            <h2>{result['sequence']}</h2>
            <p class="subtle">{result['interpretation']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Live oral score", f"{score}", "Combined live prioritization score")

    with c2:
        metric_card("Parent ACE-like", f"%{result['parent_ace_percent']}", "Sequence-level model probability")

    with c3:
        metric_card("Absorption feasibility", f"{result['absorption_score']}", "Fragment-length based estimate")

    with c4:
        metric_card("Known ACE fragments", f"{len(result['known_ace_fragments'])}", "BIOPEP-supported active fragments")

    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">1. Parent peptide AI prediction</div>', unsafe_allow_html=True)
    st.write(f"**Sequence:** {result['sequence']}")
    st.write(f"**ACE-like probability:** %{result['parent_ace_percent']}")

    for reason in result["parent_reasons"]:
        st.write("• " + reason)

    st.markdown('<div class="section-title">2. Live BIOPEP digestion</div>', unsafe_allow_html=True)
    st.write(
        "**Digestion fragments:** "
        + (", ".join(result["digestion_fragments"]) if result["digestion_fragments"] else "Not found")
    )
    st.caption(result["digest_url"])

    st.markdown('<div class="section-title">3. Known active fragment lookup</div>', unsafe_allow_html=True)
    st.write(
        "**Known ACE-active fragments:** "
        + (", ".join(result["known_ace_fragments"]) if result["known_ace_fragments"] else "None")
    )
    st.write(
        "**Known other bioactive fragments:** "
        + (", ".join(result["known_other_fragments"]) if result["known_other_fragments"] else "None")
    )

    st.markdown('<div class="section-title">4. Unknown fragment AI inference</div>', unsafe_allow_html=True)

    if result["unknown_predictions"]:
        pred_table = pd.DataFrame([
            {
                "unknown_fragment": frag,
                "length": len(frag),
                "predicted_ACE_probability_%": round(prob * 100, 2),
                "main_biological_interpretation": " ".join(result["unknown_reasons"].get(frag, [])[:2])
            }
            for frag, prob in result["unknown_predictions"].items()
        ]).sort_values("predicted_ACE_probability_%", ascending=False)

        st.dataframe(pred_table, use_container_width=True)

        st.markdown('<div class="xai-box">', unsafe_allow_html=True)
        st.markdown('<div class="xai-title">Best unknown fragment explanation</div>', unsafe_allow_html=True)
        st.write(
            f"**{result['best_unknown_fragment']}** is the strongest predicted unknown fragment "
            f"with **%{result['best_unknown_percent']} ACE-like probability**."
        )

        for reason in result["best_unknown_reasons"]:
            st.write("• " + reason)

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.write("No unknown fragment was available for AI inference.")

    st.markdown('<div class="section-title">5. Live oral candidate interpretation</div>', unsafe_allow_html=True)

    st.markdown('<div class="xai-box">', unsafe_allow_html=True)
    st.write(f"**Final interpretation:** {result['interpretation']}")
    st.write(
        "The live oral score combines parent ACE-like probability, BIOPEP-confirmed active fragments, "
        "predicted ACE-like unknown fragments, digestion behavior and fragment-length based absorption feasibility."
    )
    st.write(
        "This live mode does not automatically run ToxinPred or PlifePred. Full offline pipeline results include "
        "toxicity, stability, fragment safety and the final DigiDigest score."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


model, feature_names = load_model()
final_df = load_final_df()

if "history" not in st.session_state:
    st.session_state.history = []


st.markdown(
    """
    <div class="hero">
        <h1>🧪 DigiDigest</h1>
        <p>
        A digestion-aware decision-support platform for prioritizing oral ACE-inhibitory peptide candidates.
        Enter a parent peptide sequence, run live BIOPEP digestion, inspect active fragments and review the XAI explanation.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


tab1, tab2, tab3 = st.tabs([
    "Live Analysis",
    "Saved Candidate Table",
    "Methodology"
])


with tab1:

    st.subheader("Live candidate analysis")

    seq_input = st.text_input(
        "Parent peptide sequence",
        "IIVFGRQLL"
    )

    run = st.button("Analyse candidate")

    if run:
        seq = clean_sequence(seq_input)

        if len(seq) < 2:
            st.error("Please enter a valid peptide sequence with at least 2 amino acids.")
        else:
            with st.spinner("Running live DigiDigest analysis..."):
                result = analyse_sequence(seq, model, feature_names)

            add_to_history(result)
            render_analysis(result)


with tab2:
    st.markdown('<div class="table-card">', unsafe_allow_html=True)
    st.subheader("Saved candidate ranking")

    if not st.session_state.history:
        st.info("No candidate has been analysed yet.")
    else:
        rows = []

        for r in st.session_state.history:
            rows.append({
                "sequence": r["sequence"],
                "live_oral_score": r["live_oral_score"],
                "candidate_class": r["candidate_class"],
                "parent_ACE_%": r["parent_ace_percent"],
                "best_unknown_fragment": r["best_unknown_fragment"],
                "best_unknown_ACE_%": r["best_unknown_percent"],
                "known_ACE_fragment_count": len(r["known_ace_fragments"]),
                "absorption_score": r["absorption_score"],
            })

        table = pd.DataFrame(rows)
        table = table.sort_values("live_oral_score", ascending=False).reset_index(drop=True)
        table.insert(0, "rank", range(1, len(table) + 1))

        min_score = st.slider("Minimum oral candidate score", 0, 100, 0)

        filtered_table = table[table["live_oral_score"] >= min_score].copy()

        st.dataframe(filtered_table, use_container_width=True)

        st.download_button(
            "Download saved candidates as CSV",
            data=filtered_table.to_csv(index=False).encode("utf-8"),
            file_name="digidigest_saved_candidates.csv",
            mime="text/csv"
        )

        selected = st.selectbox(
            "Select a candidate to review its XAI explanation",
            filtered_table["sequence"].tolist()
        )

        selected_result = next(
            (r for r in st.session_state.history if r["sequence"] == selected),
            None
        )

        if selected_result:
            st.markdown("---")
            render_analysis(selected_result)

    st.markdown('</div>', unsafe_allow_html=True)


with tab3:
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)

    st.subheader("Methodology")

    st.markdown("""
    DigiDigest is a lab decision-support MVP for oral ACE-inhibitory peptide prioritization.

    **Live analysis layers**

    1. Parent peptide ACE-like prediction  
    2. BIOPEP gastrointestinal digestion simulation  
    3. Known ACE-active fragment lookup  
    4. Unknown fragment ACE-like inference  
    5. Fragment-length based absorption feasibility  
    6. Live oral candidate score  
    7. XAI explanation and saved candidate ranking  

    **Full offline pipeline layers**

    1. BIOPEP ACE candidate extraction  
    2. Non-ACE bioactive negative dataset construction  
    3. ACE-like Random Forest model training  
    4. Parent peptide ACE-like prediction  
    5. Digestion fragment analysis  
    6. Known ACE fragment lookup  
    7. Unknown fragment ACE-like prediction  
    8. Parent toxicity layer  
    9. Fragment toxicity layer  
    10. Stability / half-life confidence layer  
    11. Absorption feasibility  
    12. Final DigiDigest oral candidate ranking  
    13. XAI interpretation layer  

    **Important note:**  
    DigiDigest does not replace experimental validation. It ranks peptide candidates before lab testing.
    """)

    st.markdown('</div>', unsafe_allow_html=True)