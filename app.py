import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import re
import io
import time
import socket
import textwrap
import plotly.graph_objects as go
from Bio import Entrez, Medline
from scipy import stats as scipy_stats

# ─── App Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EIF2AK Discovery Engine",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Constants ────────────────────────────────────────────────────────────────
Entrez.email   = "samikshapasalkar2212@gmail.com"
Entrez.api_key = "9de22485baf54ae653d2825299784fcfb008"
EIF2AK_GENES   = ["EIF2AK1", "EIF2AK2", "EIF2AK3", "EIF2AK4"]

# ─── EIF2AK Alias Dictionary ─────────────────────────────────────────────────
# Comprehensive alias → canonical symbol mapping for EIF2AK1-4.
# Sources: HGNC, NCBI Gene, UniProt, Ensembl (GRCh38.p14).
# Ensembl base IDs are stored without version suffixes; version matching
# is handled by the matcher (strip ".NN" before lookup).
_EIF2AK_ALIASES: dict[str, str] = {
    # ── EIF2AK1 (HRI) ─────────────────────────────────────────────────────
    "EIF2AK1":          "EIF2AK1",
    "HRI":              "EIF2AK1",
    "HRI1":             "EIF2AK1",
    "ENSG00000086232":  "EIF2AK1",   # Ensembl gene ID (GRCh38)
    "27102":            "EIF2AK1",   # NCBI Entrez Gene ID
    "24921":            "EIF2AK1",   # HGNC ID (numeric part)
    "HGNC:24921":       "EIF2AK1",
    # ── EIF2AK2 (PKR) ─────────────────────────────────────────────────────
    "EIF2AK2":          "EIF2AK2",
    "PKR":              "EIF2AK2",
    "PRKR":             "EIF2AK2",
    # NOTE: bare "EIF2AK" is intentionally excluded — it is ambiguous and
    # causes false-positive substring matches on "EIF2AK3", "EIF2AK4", etc.
    "PPP1R83":          "EIF2AK2",
    "P68":              "EIF2AK2",
    "ENSG00000055332":  "EIF2AK2",
    "5610":             "EIF2AK2",   # Entrez
    "9437":             "EIF2AK2",   # HGNC numeric
    "HGNC:9437":        "EIF2AK2",
    # ── EIF2AK3 (PERK) ────────────────────────────────────────────────────
    "EIF2AK3":          "EIF2AK3",
    "PERK":             "EIF2AK3",
    "PEK":              "EIF2AK3",
    "WRS":              "EIF2AK3",
    "ENSG00000172071":  "EIF2AK3",
    "9451":             "EIF2AK3",   # Entrez
    "3255":             "EIF2AK3",   # HGNC numeric
    "HGNC:3255":        "EIF2AK3",
    # ── EIF2AK4 (GCN2) ────────────────────────────────────────────────────
    "EIF2AK4":          "EIF2AK4",
    "GCN2":             "EIF2AK4",
    "GCN2L1":           "EIF2AK4",
    "ENSG00000128829":  "EIF2AK4",
    "440275":           "EIF2AK4",   # Entrez
    "3257":             "EIF2AK4",   # HGNC numeric
    "HGNC:3257":        "EIF2AK4",
}

# Pre-compiled regex to capture an Ensembl gene ID (with optional version)
# or an HGNC-prefixed ID embedded anywhere in a compound label.
_ENSG_RE   = re.compile(r"\b(ENSG\d{11})(?:\.\d+)?\b", re.IGNORECASE)
_HGNC_RE   = re.compile(r"\bHGNC:(\d+)\b",              re.IGNORECASE)
_ENTREZ_RE = re.compile(r"\b(\d{4,6})\b")   # 4-6 digit numeric Entrez IDs


def match_eif2ak(label: str) -> str | None:
    """Return the canonical EIF2AK symbol (EIF2AK1-4) for *label*, or None.

    Handles:
    - Exact and case-insensitive symbol / alias matches
    - Ensembl IDs with version suffixes  (ENSG00000172071.15)
    - Compound labels                    (EIF2AK3|PERK, 9451 (EIF2AK3))
    - HGNC-prefixed IDs                 (HGNC:3255)
    - Bare Entrez / HGNC numeric IDs    (9451, 440275)
    """
    if not isinstance(label, str) or not label.strip():
        return None
    s = label.strip()

    # 1. Direct case-insensitive lookup of the whole label
    canonical = _EIF2AK_ALIASES.get(s.upper(), _EIF2AK_ALIASES.get(s))
    if canonical:
        return canonical

    s_up = s.upper()

    # 2. Scan for known aliases as substrings (handles "EIF2AK3|PERK" etc.)
    #    Sort by alias length descending so longer / more specific tokens win
    #    (e.g. "EIF2AK3" is matched before any 3-letter alias like "PKR").
    for alias, canon in sorted(_EIF2AK_ALIASES.items(),
                                key=lambda kv: len(kv[0]), reverse=True):
        if alias and alias in s_up:
            return canon

    # 3. Extract embedded Ensembl gene ID (strip version suffix)
    for m in _ENSG_RE.finditer(s):
        canon = _EIF2AK_ALIASES.get(m.group(1).upper())
        if canon:
            return canon

    # 4. Extract embedded HGNC:NNNN
    for m in _HGNC_RE.finditer(s):
        canon = _EIF2AK_ALIASES.get(f"HGNC:{m.group(1)}")
        if canon:
            return canon

    # 5. Extract embedded numeric Entrez / HGNC IDs (4-6 digits)
    for m in _ENTREZ_RE.finditer(s):
        canon = _EIF2AK_ALIASES.get(m.group(1))
        if canon:
            return canon

    return None


def compute_mannwhitney_bh(
    tpm_df: pd.DataFrame,
    ctrl_cols: list[str],
    dis_cols: list[str],
    min_n_per_group: int = 2,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Per-gene Control-vs-Disease significance testing on the TPM matrix.

    This performs **real, exact statistics on the actual sample values** —
    nothing here is predicted, imputed, simulated, or approximated:

    - Mann-Whitney U (two-sided, `scipy.stats.mannwhitneyu`) is run
      independently for every gene using only that gene's real TPM values
      in `ctrl_cols` vs `dis_cols`. `method="auto"` lets SciPy pick the
      *exact* permutation distribution for small/tie-free samples and the
      asymptotic normal approximation only once the exact computation
      becomes intractable (SciPy's own documented crossover) — this is
      the standard, textbook-correct behaviour, not a shortcut.
    - Genes that cannot be validly tested (fewer than `min_n_per_group`
      real replicates in either group, or identical values in both groups
      so no U-statistic exists) are left as NaN rather than filled with a
      guessed p-value of 0, 1, or anything else.
    - Benjamini-Hochberg FDR correction (`scipy.stats.false_discovery_control`,
      method="bh") is then applied *only* across the genes that actually
      produced a real p-value. Untested (NaN) genes are excluded from the
      correction entirely, since folding them in would silently distort
      the correction for every other gene.

    Returns
    -------
    (p_values, q_values, n_tested) as pandas Series aligned to tpm_df.index.
    n_tested is the same valid/invalid mask, exposed as an int Series
    (1 = real test ran, 0 = not enough data to test) so the UI can be
    fully transparent about which genes have a genuine p-value vs. none.
    """
    idx = tpm_df.index
    pvals = pd.Series(np.nan, index=idx, dtype=float)
    tested = pd.Series(0, index=idx, dtype=int)

    n_ctrl, n_dis = len(ctrl_cols), len(dis_cols)
    if n_ctrl >= min_n_per_group and n_dis >= min_n_per_group:
        ctrl_arr = tpm_df[ctrl_cols].apply(pd.to_numeric, errors="coerce").values
        dis_arr  = tpm_df[dis_cols].apply(pd.to_numeric, errors="coerce").values

        for i in range(len(idx)):
            c = ctrl_arr[i]
            d = dis_arr[i]
            # Drop any NaNs from coercion so the test only ever sees real
            # numeric measurements — never a filled-in / guessed value.
            c = c[~np.isnan(c)]
            d = d[~np.isnan(d)]
            if len(c) < min_n_per_group or len(d) < min_n_per_group:
                continue
            try:
                _, p = scipy_stats.mannwhitneyu(c, d, alternative="two-sided",
                                                 method="auto")
            except ValueError:
                # Raised by SciPy when every value in both groups is
                # identical (no variation → no valid U-statistic). Left
                # as NaN rather than assigning an arbitrary p-value.
                continue
            pvals.iloc[i] = p
            tested.iloc[i] = 1

    qvals = pd.Series(np.nan, index=idx, dtype=float)
    valid_mask = pvals.notna()
    if valid_mask.sum() > 0:
        adj = scipy_stats.false_discovery_control(pvals[valid_mask].values, method="bh")
        qvals.loc[valid_mask] = adj

    return pvals, qvals, tested


def find_eif2ak_rows(index: pd.Index) -> pd.Series:
    """Scan a gene index and return a Series mapping original_label → canonical
    for every row that matches any EIF2AK gene.  Empty if none found."""
    hits = {}
    for label in index:
        canon = match_eif2ak(str(label))
        if canon:
            hits[label] = canon
    return pd.Series(hits, dtype=str)

# Biopython's Entrez functions have no per-call timeout argument — under the
# hood they use urllib, which honours Python's *global* default socket
# timeout (socket.setdefaulttimeout). That setting is process-wide, so a
# short timeout used for a quick "is NCBI up?" probe would otherwise also
# cut off slower, legitimate esearch/esummary requests later. We therefore
# never use socket.setdefaulttimeout() for the probe itself (see
# _ncbi_reachable below, which uses its own throwaway socket), and instead
# set one generous global timeout for actual NCBI data calls.
NCBI_DATA_TIMEOUT = 30  # seconds — applies to real esearch/esummary/efetch calls

def _ncbi_reachable(host: str = "eutils.ncbi.nlm.nih.gov", port: int = 443,
                     timeout: int = 3) -> bool:
    """Quick reachability probe using its own socket/timeout — does NOT
    touch the global socket default, so it can't shrink the timeout used
    by real NCBI data requests made elsewhere in the app."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def _entrez_call(fn, *args, retries: int = 3, **kwargs):
    """Call an Entrez function (esearch/esummary/efetch) with a couple of
    retries on transient network errors — NCBI occasionally hiccups partway
    through a long paginated fetch, and a single dropped request shouldn't
    discard everything already retrieved."""
    last_err = None
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    raise last_err

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Sora:wght@300;400;500;600;700&display=swap');

:root {
    --bg:       #050b18;
    --surface:  #0c1424;
    --surface2: #111d30;
    --surface3: #162340;
    --accent:   #00ffcc;
    --accent2:  #8b5cf6;
    --accent3:  #f472b6;
    --text:     #dde4f0;
    --muted:    #5a6a80;
    --border:   #1a2d45;
    --success:  #10b981;
    --warning:  #f59e0b;
    --danger:   #ef4444;
    --glow:     rgba(0,255,204,0.15);
}

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}
.stApp { background-color: var(--bg); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060d1c 0%, #0a1525 100%);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stRadio label { color: var(--text) !important; font-size:0.88rem; }

/* Hero */
.hero-banner {
    background: linear-gradient(135deg, #08132a 0%, #0e1e3a 40%, #06111f 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top:-60%; right:-8%;
    width:500px; height:500px;
    background: radial-gradient(circle, rgba(0,255,204,0.07) 0%, transparent 65%);
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom:-60%; left:-5%;
    width:400px; height:400px;
    background: radial-gradient(circle, rgba(139,92,246,0.05) 0%, transparent 65%);
    pointer-events: none;
}
.hero-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--accent);
    margin: 0 0 0.35rem 0;
    letter-spacing: -0.5px;
    text-shadow: 0 0 30px rgba(0,255,204,0.3);
}
.hero-subtitle { color: var(--muted); font-size: 0.9rem; margin:0; }
.gene-chips { display:flex; gap:0.45rem; flex-wrap:wrap; margin-top:1rem; }
.gene-chip {
    background: rgba(0,255,204,0.08);
    border: 1px solid rgba(0,255,204,0.25);
    color: var(--accent);
    padding: 0.18rem 0.7rem;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1rem;
}
.card-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 0.7rem;
}

/* Evidence */
.evidence-badge {
    background: rgba(0,255,204,0.06);
    border-left: 3px solid var(--accent);
    padding: 0.6rem 1rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.83rem;
    color: var(--text);
    margin-top: 0.5rem;
    line-height: 1.6;
}

/* Step indicator */
.step-indicator { display:flex; align-items:center; gap:0.75rem; margin-bottom:1.4rem; }
.step-num {
    background: var(--accent2);
    color: white;
    width:26px; height:26px;
    border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-family:'JetBrains Mono',monospace;
    font-size:0.72rem; font-weight:700; flex-shrink:0;
}
.step-label { font-size:0.9rem; color:var(--text); font-weight:500; }

/* Metrics */
.metric-row { display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:1.5rem; }
.metric-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.4rem;
    flex: 1; min-width: 110px;
}
.metric-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem; font-weight: 700;
    color: var(--accent);
}
.metric-lbl { font-size:0.72rem; color:var(--muted); margin-top:0.2rem; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent2), #5b21b6) !important;
    color: white !important; border:none !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    padding: 0.5rem 1.4rem !important;
    font-weight: 600 !important; letter-spacing: 0.4px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover { opacity:0.88 !important; transform:translateY(-1px) !important; }

/* Inputs */
.stTextInput > div > div > input,
.stSelectbox > div > div > div,
.stMultiSelect > div > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

/* Misc */
.stDataFrame { border-radius:10px; overflow:hidden; }
hr { border-color: var(--border) !important; }
.stAlert { border-radius:8px !important; }
.section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem; color: var(--text);
    margin: 1.8rem 0 1rem 0;
    padding-bottom: 0.45rem;
    border-bottom: 1px solid var(--border);
}

/* Code block */
.code-block {
    background: #0a0f1e;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #a8b8d0;
    white-space: pre-wrap;
    word-break: break-word;
    line-height: 1.65;
    overflow-x: auto;
}
.kw  { color: #8b5cf6; font-weight:600; }
.fn  { color: #00ffcc; }
.str { color: #fbbf24; }
.cmt { color: #4a5a6a; font-style:italic; }
.num { color: #f472b6; }

/* Spotlight gene card */
.gene-spotlight {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    text-align: center;
}
.gene-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem; font-weight:700;
    color: var(--accent); margin-bottom:0.5rem;
}
.gene-lfc {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem; font-weight:700;
}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="padding:1.1rem 0 1.5rem 0; text-align:center;">
  <div style="font-family:'JetBrains Mono',monospace; font-size:1.05rem; color:#00ffcc; font-weight:700;">
    🧬 EIF2AK Engine
  </div>
  <div style="font-size:0.72rem; color:#5a6a80; margin-top:0.3rem;">Discovery &amp; Analysis Platform</div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["📄 Literature Evidence", "🔬 GEO AI-Agent & Explorer", "📊 TPM Normalisation"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size:0.74rem; color:#475569; padding:0.4rem 0;">
<b style="color:#94a3b8;">Gene Family</b><br>
EIF2AK1 · EIF2AK2<br>EIF2AK3 · EIF2AK4<br><br>
<b style="color:#94a3b8;">Integrated Services</b><br>
PubMed · PubTator3<br>GEO · GEOparse<br>TPM Norm · mygene
</div>
""", unsafe_allow_html=True)

if st.sidebar.checkbox("🛠 Debug: Session Keys", value=False):
    st.sidebar.write(list(st.session_state.keys()))
    if "clean_counts" in st.session_state:
        s = st.session_state["clean_counts"]
        st.sidebar.success(f"clean_counts: {s.shape[0]}g × {s.shape[1]}s")

# ── Network connectivity indicator ────────────────────────────────────────────
if _ncbi_reachable(timeout=3):
    st.sidebar.markdown(
        '<div style="font-size:0.73rem; color:#10b981; margin-top:0.5rem;">🟢 NCBI reachable</div>',
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        '<div style="font-size:0.73rem; color:#ef4444; margin-top:0.5rem;">'
        '🔴 NCBI unreachable — check network/VPN</div>',
        unsafe_allow_html=True
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: LITERATURE EVIDENCE  —  PubTator3 clone
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📄 Literature Evidence":

    st.markdown("""
    <div class="hero-banner">
      <div class="hero-title">Literature Evidence Mining</div>
      <div class="hero-subtitle">
        A faithful PubTator3 client — keyword · entity · relation search,
        live against the real NCBI PubTator3 API
      </div>
      <div class="gene-chips">
        <span class="gene-chip">EIF2AK1</span>
        <span class="gene-chip">EIF2AK2</span>
        <span class="gene-chip">EIF2AK3</span>
        <span class="gene-chip">EIF2AK4</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── PubTator3 constants (mirrors the real system exactly) ──────────────────
    PT3_ENTITY_TYPES = {
        "GENE":     {"label": "Gene/Protein",     "color": "#a78bfa", "emoji": "🧬"},  # purple
        "DISEASE":  {"label": "Disease",          "color": "#fb923c", "emoji": "🦠"},  # orange
        "CHEMICAL": {"label": "Chemical",         "color": "#34d399", "emoji": "💊"},  # green
        "VARIANT":  {"label": "Genetic Variant",  "color": "#a8755a", "emoji": "🔬"},  # brown
        "SPECIES":  {"label": "Species",          "color": "#60a5fa", "emoji": "🐾"},  # blue
        "CELLLINE": {"label": "Cell Line",        "color": "#2dd4bf", "emoji": "🧫"},  # teal
    }
    # PubTator3 BioC "type" infons sometimes appear with different casing/aliases
    PT3_TYPE_ALIASES = {
        "GENE": "GENE", "DISEASE": "DISEASE", "CHEMICAL": "CHEMICAL",
        "MUTATION": "VARIANT", "VARIANT": "VARIANT", "DNAMUTATION": "VARIANT",
        "PROTEINMUTATION": "VARIANT", "SNP": "VARIANT",
        "SPECIES": "SPECIES", "CELLLINE": "CELLLINE", "CELL_LINE": "CELLLINE",
    }
    PT3_RELATION_TYPES = [
        ("treat",              "Treat",                "Chemical treats disease"),
        ("cause",               "Cause",                "Entity causes disease/effect"),
        ("prevent",             "Prevent",              "Chemical prevents disease"),
        ("inhibit",             "Inhibit",              "Entity inhibits another"),
        ("stimulate",           "Stimulate",            "Entity stimulates another"),
        ("interact",            "Interact",             "Entities interact"),
        ("associate",           "Associate",            "Entities are associated (any pair)"),
        ("positive_correlate",  "Positive correlate",   "Positive correlation"),
        ("negative_correlate",  "Negative correlate",   "Negative correlation"),
        ("cotreat",             "Co-treat",              "Co-treatment relationship"),
        ("compare",             "Compare",              "Entities compared"),
        ("convert",             "Convert",              "Entity converts to another"),
        ("drug_interact",       "Drug interact",        "Drug-drug interaction"),
    ]
    PT3_E2_TYPES = ["chemical", "disease", "gene", "variant"]

    PT3_API_ROOT = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"
    PT3_HEADERS = {
        # NCBI asks API clients to identify themselves with a descriptive
        # User-Agent (same courtesy E-utilities requests via Entrez.email
        # above) — some NCBI edge services are stricter about a default
        # python-requests UA than others.
        "User-Agent": "EIF2AK-Discovery-Engine/1.0 (Streamlit; +samikshapasalkar2212@gmail.com)",
        "Accept": "application/json",
    }

    # ── Low level HTTP helper with retry ────────────────────────────────────────
    def _pt3_get(path: str, params: dict, timeout: int = 20, retries: int = 2):
        url = f"{PT3_API_ROOT}{path}"
        last_err = None
        for attempt in range(retries + 1):
            try:
                resp = requests.get(url, params=params, headers=PT3_HEADERS, timeout=timeout)
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except ValueError:
                        last_err = "PubTator3 returned a non-JSON response"
                        break
                elif resp.status_code == 429:
                    last_err = "Rate limited by PubTator3 (HTTP 429) — try again shortly"
                    time.sleep(2.0 * (attempt + 1))
                    continue
                else:
                    last_err = f"HTTP {resp.status_code} from PubTator3"
            except requests.exceptions.Timeout:
                last_err = "PubTator3 request timed out"
            except requests.exceptions.ConnectionError:
                last_err = "Could not reach PubTator3 (network/DNS issue)"
            except Exception as e:
                last_err = str(e)
            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))
        return {"_error": last_err}

    # ── 1. FindEntityID — /entity/autocomplete/ ─────────────────────────────────
    @st.cache_data(show_spinner=False, ttl=3600)
    def pt3_autocomplete(query: str, concept: str = None, limit: int = 10):
        """
        Real endpoint: GET /research/pubtator3-api/entity/autocomplete/?query=...&limit=...
        Returns a list of {"_id"/"id": "@TYPE_Name", "name": ..., "biotype": ...}
        suggestions. NCBI's live response is a JSON array of suggestion objects;
        we normalise key access defensively since exact key names have shifted
        across PubTator3 releases.
        """
        params = {"query": query, "limit": limit}
        if concept:
            params["concept"] = concept
        data = _pt3_get("/entity/autocomplete/", params)
        if isinstance(data, dict) and data.get("_error"):
            return []
        items = data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        out = []
        for it in items:
            if not isinstance(it, dict):
                continue
            eid = it.get("_id") or it.get("id") or it.get("entity_id") or ""
            name = it.get("name") or it.get("label") or it.get("text") or eid
            biotype = it.get("biotype") or it.get("type") or ""
            if eid:
                out.append({"id": eid, "name": name, "biotype": biotype})
        return out

    # ── 2. FindRelatedEntity — /relations ────────────────────────────────────────
    @st.cache_data(show_spinner=False, ttl=3600)
    def pt3_related_entities(e1: str, e2_type: str, rel_type: str, limit: int = 10):
        """
        Real endpoint: GET /research/pubtator3-api/relations?e1=...&e2=...&type=...&limit=...
        Returns {"relations": [{"entity_id", "relation_type", "details": {...}}, ...]}
        """
        params = {"e1": e1, "e2": e2_type, "type": rel_type, "limit": limit}
        data = _pt3_get("/relations", params)
        if isinstance(data, dict) and data.get("_error"):
            return [], data.get("_error")
        rels = data.get("relations", []) if isinstance(data, dict) else []
        return rels, None

    # ── 3a. SearchPubTatorID — /search/ (keyword OR relations: query) ──────────
    @st.cache_data(show_spinner=False, ttl=1800)
    def pt3_search(text_query: str, page_num: int = 1):
        """
        Real endpoint: GET /research/pubtator3-api/search/?text=...&page=...
        `text_query` can be:
          - plain keywords:            "EIF2AK3 endoplasmic reticulum stress"
          - a single entity:           "@GENE_EIF2AK3"
          - boolean entity/keyword mix:"@DISEASE_Diabetes_Mellitus AND stress"
          - a relation query:          "relations:treat|@CHEMICAL_X|@DISEASE_Y"
        Response shape (per NCBI's documented schema + live observation):
          {"results": [ {pmid, title, journal, date/pub_date, authors,
                         text_hl/snippet, score, ...}, ... ],
           "page": 1, "count": N, "total_pages": M, "facets": {...}}
        We normalise defensively since PubTator3 has used both snake_case and
        camelCase keys across versions.
        """
        params = {"text": text_query, "page": page_num}
        data = _pt3_get("/search/", params, timeout=25)
        return data

    # ── 3b. Export annotations — /publications/export/biocjson ─────────────────
    @st.cache_data(show_spinner=False, ttl=3600)
    def pt3_export_biocjson(pmids: list, full_text: bool = False):
        """
        Real endpoint:
          GET /research/pubtator3-api/publications/export/biocjson
              ?pmids=1,2,3&full=true|false
        Returns BioC-JSON: a list of documents, each with "passages", each
        passage carrying "text", "offset", and "annotations" (with "infons"
        for type/identifier and "locations" for character offsets), plus
        top-level "relations" per document where BioREx found them.
        Chunked to PubTator3's documented 100-PMID batch limit.
        """
        results = {}
        if not pmids:
            return results
        chunk = 100
        for start in range(0, len(pmids), chunk):
            batch = pmids[start:start + chunk]
            params = {
                "pmids": ",".join(str(p) for p in batch),
                "full": "true" if full_text else "false",
            }
            data = _pt3_get("/publications/export/biocjson", params, timeout=30)
            if isinstance(data, dict) and data.get("_error"):
                continue
            docs = data if isinstance(data, list) else data.get("PubTator3", [])
            for doc in docs:
                pmid = str(doc.get("id", doc.get("pmid", "")))
                if not pmid:
                    continue
                annotations, relations = [], doc.get("relations", [])
                for passage in doc.get("passages", []):
                    for ann in passage.get("annotations", []):
                        infons = ann.get("infons", {})
                        raw_type = str(infons.get("type", "")).upper()
                        norm_type = PT3_TYPE_ALIASES.get(raw_type, raw_type)
                        locs = ann.get("locations", [{}])
                        offset = locs[0].get("offset") if locs else None
                        length = locs[0].get("length") if locs else None
                        annotations.append({
                            "text": ann.get("text", ""),
                            "type": norm_type,
                            "identifier": infons.get("identifier", ""),
                            "offset": offset,
                            "length": length,
                            "section": passage.get("infons", {}).get("section_type", "")
                                       or passage.get("infons", {}).get("type", ""),
                        })
                if pmid:
                    results[pmid] = {"annotations": annotations, "relations": relations}
        return results

    # ── Entity legend ────────────────────────────────────────────────────────
    def render_entity_legend():
        chips = "".join(
            f'<span style="display:inline-flex;align-items:center;gap:0.3rem;'
            f'background:{m["color"]}1a;border:1px solid {m["color"]}55;'
            f'color:{m["color"]};padding:0.15rem 0.6rem;border-radius:14px;'
            f'font-family:JetBrains Mono,monospace;font-size:0.7rem;'
            f'font-weight:600;margin-right:0.4rem;">{m["emoji"]} {m["label"]}</span>'
            for m in PT3_ENTITY_TYPES.values()
        )
        st.markdown(f'<div style="margin:0.6rem 0 1rem 0;">{chips}</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SEARCH MODE SELECTOR — mirrors PubTator3's keyword / entity / relation
    # search modes exactly
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔍 Search PubTator3</div>', unsafe_allow_html=True)

    search_mode = st.radio(
        "Search mode",
        ["🔤 Keyword", "🧬 Entity", "🔗 Relation"],
        horizontal=True,
        help="Keyword: free text, like PubMed. Entity: search by a normalised "
             "bio-concept (gene/disease/chemical/...). Relation: find articles "
             "supporting a specific relation between two entities — "
             "all three are real PubTator3 search modes.",
    )

    final_query = None   # the `text=` value sent to /search/
    query_label = None    # human-readable label for the results header

    # ---- Keyword mode --------------------------------------------------------
    if search_mode == "🔤 Keyword":
        col1, col2 = st.columns([2, 2])
        with col1:
            disease = st.text_input(
                "🦠 Disease / Condition (optional)",
                placeholder="Leave blank to search ALL diseases",
            )
        with col2:
            gene_choice = st.multiselect("🧬 Gene(s)", EIF2AK_GENES, default=EIF2AK_GENES)

        if not disease:
            st.info("💡 No disease entered — will search **all disease contexts** for selected EIF2AK genes.")

        gene_part = " OR ".join(gene_choice) if gene_choice else ""
        if gene_part and disease:
            final_query = f"({gene_part}) AND {disease}"
        elif gene_part:
            final_query = gene_part
        else:
            final_query = disease
        query_label = final_query

    # ---- Entity mode ----------------------------------------------------------
    elif search_mode == "🧬 Entity":
        st.markdown(
            '<div class="card" style="font-size:0.82rem; color:#94a3b8;">'
            'Type a bio-concept name (gene, disease, chemical, species, '
            'variant, or cell line). This calls PubTator3\'s real '
            '<code>FindEntityID</code> autocomplete endpoint to resolve it to a '
            'normalised entity ID (e.g. <code>@GENE_EIF2AK3</code>), then '
            'searches for articles annotated with that exact concept.</div>',
            unsafe_allow_html=True,
        )
        ac1, ac2 = st.columns([3, 1])
        with ac1:
            entity_free_text = st.text_input(
                "Entity name", value="EIF2AK3",
                placeholder="e.g. EIF2AK3, PERK, type 2 diabetes, metformin…",
            )
        with ac2:
            entity_concept = st.selectbox(
                "Restrict to type",
                ["(any)", "gene", "disease", "chemical", "species", "mutation"],
            )

        resolved_id, resolved_name = None, None
        if entity_free_text.strip():
            concept_param = None if entity_concept == "(any)" else entity_concept
            with st.spinner("🔎 Resolving entity via PubTator3 autocomplete…"):
                suggestions = pt3_autocomplete(entity_free_text.strip(), concept_param, limit=10)
            if suggestions:
                opts = [f'{s["name"]}  ·  {s["id"]}' for s in suggestions]
                pick = st.selectbox("Matched PubTator3 entities", opts)
                idx = opts.index(pick)
                resolved_id = suggestions[idx]["id"]
                resolved_name = suggestions[idx]["name"]
            else:
                st.warning(
                    "No PubTator3 entity match found yet — refine the spelling, "
                    "or this search will fall back to plain keyword search."
                )

        final_query = resolved_id if resolved_id else entity_free_text.strip()
        query_label = resolved_name or entity_free_text.strip()

    # ---- Relation mode ----------------------------------------------------------
    else:
        st.markdown(
            '<div class="card" style="font-size:0.82rem; color:#94a3b8;">'
            'Real PubTator3 relation search: pick a starting entity, a relation '
            'type, and the type of entity it should relate to. This calls the '
            'live <code>FindRelatedEntity</code> endpoint to get the top related '
            'concepts, then <code>SearchPubTatorID</code> to fetch articles '
            'supporting each relation — exactly how the official site\'s '
            '"Relations" search works.</div>',
            unsafe_allow_html=True,
        )
        rc1, rc2 = st.columns([3, 1])
        with rc1:
            rel_entity_text = st.text_input(
                "Entity 1 (e.g. a gene)", value="EIF2AK3",
                placeholder="e.g. EIF2AK3, metformin, type 2 diabetes…",
            )
        with rc2:
            rel_entity_concept = st.selectbox(
                "Entity 1 type", ["gene", "disease", "chemical", "species", "mutation"],
            )

        rel_resolved_id, rel_resolved_name = None, None
        if rel_entity_text.strip():
            with st.spinner("🔎 Resolving entity…"):
                rel_suggestions = pt3_autocomplete(rel_entity_text.strip(), rel_entity_concept, limit=10)
            if rel_suggestions:
                opts2 = [f'{s["name"]}  ·  {s["id"]}' for s in rel_suggestions]
                pick2 = st.selectbox("Matched entity", opts2)
                idx2 = opts2.index(pick2)
                rel_resolved_id = rel_suggestions[idx2]["id"]
                rel_resolved_name = rel_suggestions[idx2]["name"]
            else:
                st.warning("No PubTator3 entity match found for Entity 1 — try a different spelling.")

        rc3, rc4 = st.columns(2)
        with rc3:
            rel_type_pick = st.selectbox(
                "Relation type",
                [f"{label} — {desc}" for _, label, desc in PT3_RELATION_TYPES],
            )
            rel_type_value = PT3_RELATION_TYPES[
                [f"{l} — {d}" for _, l, d in PT3_RELATION_TYPES].index(rel_type_pick)
            ][0]
        with rc4:
            rel_e2_type = st.selectbox("Related entity type (Entity 2)", PT3_E2_TYPES)

        related_pick_id, related_pick_name = None, None
        if rel_resolved_id:
            with st.spinner("🔗 Finding related entities via PubTator3…"):
                related, rel_err = pt3_related_entities(
                    rel_resolved_id, rel_e2_type, rel_type_value, limit=10
                )
            if rel_err:
                st.error(f"PubTator3 relations API error: {rel_err}")
            elif related:
                rel_opts = []
                for r in related:
                    rid = r.get("entity_id", "")
                    details = r.get("details", {}) or {}
                    rname = details.get("name") or rid
                    score = details.get("score")
                    label = f"{rname}  ·  {rid}" + (f"  (score {score:.2f})" if isinstance(score, (int, float)) else "")
                    rel_opts.append((label, rid, rname))
                pick3 = st.selectbox("Top related entities", [o[0] for o in rel_opts])
                sel = next(o for o in rel_opts if o[0] == pick3)
                related_pick_id, related_pick_name = sel[1], sel[2]
            else:
                st.info("No related entities returned for this combination yet — try a different relation type.")

        if rel_resolved_id and related_pick_id:
            final_query = f"relations:{rel_type_value}|{rel_resolved_id}|{related_pick_id}"
            query_label = f"{rel_resolved_name} —[{rel_type_value}]→ {related_pick_name}"
        else:
            final_query = None
            query_label = None

    st.markdown("---")
    colB, colC = st.columns([1, 1])
    with colB:
        max_pages_fetch = st.slider("Result pages to fetch", 1, 10, 2,
                                     help="PubTator3 paginates ~10 results per page server-side.")
    with colC:
        run_lit = st.button("🔍 Search PubTator3", type="primary", use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # EXECUTE SEARCH
    # ─────────────────────────────────────────────────────────────────────────
    if run_lit:
        if not final_query:
            st.warning("Enter or resolve a query first.")
        else:
            all_results, total_count = [], None
            with st.spinner(f"📡 Querying PubTator3 — `{final_query}`…"):
                for pg in range(1, max_pages_fetch + 1):
                    data = pt3_search(final_query, pg)
                    if isinstance(data, dict) and data.get("_error"):
                        st.error(f"PubTator3 search error: {data['_error']}")
                        break
                    if not isinstance(data, dict):
                        break
                    page_results = data.get("results", [])
                    if total_count is None:
                        total_count = data.get("count", data.get("total", len(page_results)))
                    all_results.extend(page_results)
                    if not page_results or len(page_results) < 10:
                        break  # short page ⇒ no more results

            if not all_results:
                st.error(
                    "No results from PubTator3 for this query. Try a broader "
                    "keyword, a different entity spelling, or another relation type."
                )
            else:
                pmids = [str(r.get("pmid", r.get("_id", r.get("id", "")))) for r in all_results]
                pmids = [p for p in pmids if p]

                with st.spinner("🔬 Fetching full BioC-JSON entity + relation annotations…"):
                    annot_cache = pt3_export_biocjson(pmids, full_text=False)

                annotated_count = sum(1 for p in pmids if p in annot_cache)
                relation_count = sum(len(v.get("relations", [])) for v in annot_cache.values())

                # ── Metrics row ──────────────────────────────────────────────
                st.markdown(f"""
                <div class="metric-row">
                  <div class="metric-box">
                    <div class="metric-val">{total_count if total_count is not None else len(all_results)}</div>
                    <div class="metric-lbl">Total PubTator3 Hits</div>
                  </div>
                  <div class="metric-box">
                    <div class="metric-val">{len(all_results)}</div>
                    <div class="metric-lbl">Fetched This Session</div>
                  </div>
                  <div class="metric-box">
                    <div class="metric-val">{annotated_count}</div>
                    <div class="metric-lbl">Annotated PMIDs</div>
                  </div>
                  <div class="metric-box">
                    <div class="metric-val">{relation_count}</div>
                    <div class="metric-lbl">Relations Found</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                render_entity_legend()

                # ── Year histogram (matches PubTator3's results-by-year chart) ──
                years = []
                for r in all_results:
                    d = str(r.get("date") or r.get("pub_date") or r.get("pubdate") or "")
                    m = re.search(r"(19|20)\d{2}", d)
                    if m:
                        years.append(int(m.group(0)))
                if years:
                    yr_counts = pd.Series(years).value_counts().sort_index()
                    hist_fig = go.Figure(go.Bar(
                        x=yr_counts.index.astype(str), y=yr_counts.values,
                        marker_color="#00ffcc",
                    ))
                    hist_fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Sora, sans-serif", color="#dde4f0", size=12),
                        height=220, margin=dict(l=40, r=20, t=10, b=30),
                        xaxis_title=None, yaxis_title="Articles",
                    )
                    hist_fig.update_xaxes(gridcolor="#1a2d45")
                    hist_fig.update_yaxes(gridcolor="#1a2d45")
                    st.markdown('<div class="section-header">📊 Results by Publication Year</div>',
                                unsafe_allow_html=True)
                    st.plotly_chart(hist_fig, use_container_width=True)

                # ── Facet filters (Section / Journal / Type — real PubTator3 facets) ──
                st.markdown('<div class="section-header">🧰 Facet Filters</div>', unsafe_allow_html=True)
                journals = sorted({r.get("journal", "") for r in all_results if r.get("journal")})
                fcol1, fcol2 = st.columns(2)
                with fcol1:
                    journal_filter = st.multiselect("Journal", journals, default=[])
                with fcol2:
                    text_filter = st.text_input("🔎 Filter results further (title/snippet contains)",
                                                 placeholder="keyword…")

                filtered_results = all_results
                if journal_filter:
                    filtered_results = [r for r in filtered_results if r.get("journal", "") in journal_filter]
                if text_filter:
                    tl = text_filter.lower()
                    filtered_results = [
                        r for r in filtered_results
                        if tl in str(r.get("title", "")).lower()
                        or tl in str(r.get("text_hl", r.get("snippet", ""))).lower()
                    ]

                st.session_state["pt3_results"] = filtered_results
                st.session_state["pt3_annot_cache"] = annot_cache

                # ── Build a flat table with direct links (replaces the old
                # separate "Article Detail View" expanders below) ──────────
                rows = []
                for r in filtered_results:
                    pmid = str(r.get("pmid", r.get("_id", r.get("id", ""))))
                    annot = annot_cache.get(pmid, {})
                    rows.append({
                        "PMID": pmid,
                        "Title": r.get("title", "N/A"),
                        "Journal": r.get("journal", "N/A"),
                        "Date": r.get("date", r.get("pub_date", "N/A")),
                        "Relations": len(annot.get("relations", [])),
                        "PubMed": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                        "PubTator3": f"https://www.ncbi.nlm.nih.gov/research/pubtator3/publication/{pmid}" if pmid else None,
                    })
                df_lit = pd.DataFrame(rows)

                st.markdown('<div class="section-header">📋 Results Table</div>', unsafe_allow_html=True)
                st.dataframe(
                    df_lit,
                    use_container_width=True,
                    height=420,
                    hide_index=True,
                    column_config={
                        "PubMed": st.column_config.LinkColumn("PubMed", display_text="Open ↗"),
                        "PubTator3": st.column_config.LinkColumn("PubTator3", display_text="Open ↗"),
                    },
                )

                buf = io.StringIO()
                df_lit.to_csv(buf, index=False)
                st.download_button("⬇ Download CSV", buf.getvalue(),
                                   file_name="pubtator3_results.csv", mime="text/csv")

                st.markdown("---")
                st.caption(
                    "🔌 Live data from the official PubTator3 REST API "
                    "(`pubtator3-api/search`, `/relations`, `/entity/autocomplete`, "
                    "`/publications/export/biocjson`) — operated by NCBI/NLM. "
                    "No local keyword-matching fallback is used; if the API has "
                    "no annotations for a PMID, the entity list will simply be empty."
                )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: GEO AI-AGENT & EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 GEO AI-Agent & Explorer":

    st.markdown("""
    <div class="hero-banner">
      <div class="hero-title">GEO AI-Agent &amp; Explorer</div>
      <div class="hero-subtitle">
        Search NCBI GEO · RNA-seq datasets · Metadata &amp; sample-group detection
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 2-A  GEO SEARCH (mirrors ncbi.nlm.nih.gov/gds exactly) ─────────────────
    st.markdown('<div class="section-header">🔍 Search GEO DataSets</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="font-size:0.82rem; color:#94a3b8; line-height:1.7;">
      Queries the same Entrez <code>gds</code> database that powers
      <a href="https://www.ncbi.nlm.nih.gov/gds" target="_blank" style="color:#00ffcc;">ncbi.nlm.nih.gov/gds</a>,
      using the identical field tags the GEO Advanced Search Builder uses
      (<code>[Organism]</code>, <code>[DataSet Type]</code>,
      <code>[Number of Samples]</code>, <code>[Publication Date]</code>, <code>[Supplementary Files]</code>).
      No entry-type restriction is applied by default — exactly like the plain
      ncbi.nlm.nih.gov/gds search box, results span DataSets, Series, Platforms
      and Samples together.
    </div>
    """, unsafe_allow_html=True)

    # Official GEO "DataSet Type" fixed-list values (same list shown on the
    # GEO Advanced Search page when you click the DataSet Type field dropdown).
    GEO_STUDY_TYPES = [
        "Expression profiling by array",
        "Expression profiling by high throughput sequencing",
        "Expression profiling by genome tiling array",
        "Expression profiling by RT-PCR",
        "Expression profiling by SNP array",
        "Expression profiling by MPSS",
        "Expression profiling by SAGE",
        "Genome variation profiling by array",
        "Genome variation profiling by high throughput sequencing",
        "Genome variation profiling by SNP array",
        "Genome variation profiling by genome tiling array",
        "Genome binding/occupancy profiling by array",
        "Genome binding/occupancy profiling by high throughput sequencing",
        "Genome binding/occupancy profiling by genome tiling array",
        "Methylation profiling by array",
        "Methylation profiling by high throughput sequencing",
        "Methylation profiling by SNP array",
        "Methylation profiling by genome tiling array",
        "Non-coding RNA profiling by array",
        "Non-coding RNA profiling by high throughput sequencing",
        "Protein profiling by array",
        "Protein profiling by mass spec",
        "SNP genotyping by SNP array",
        "Third-party reanalysis",
        "Other",
    ]
    GEO_COMMON_ORGANISMS = [
        "Any organism", "Homo sapiens", "Mus musculus", "Rattus norvegicus",
        "Danio rerio", "Drosophila melanogaster", "Caenorhabditis elegans",
        "Saccharomyces cerevisiae", "Arabidopsis thaliana", "Custom (type below)",
    ]

    c1, c2 = st.columns([3, 2])
    with c1:
        geo_query = st.text_input(
            "🔎 Search term — same syntax as the GEO search box ([All Fields])",
            placeholder='e.g. diabetes OR "ER stress"',
        )
    with c2:
        inc_eif = st.checkbox("Auto-include EIF2AK in query", value=True)

    with st.expander("🔧 Advanced filters (GEO Advanced Search Builder fields)"):
        # No Entry Type selector: results aren't restricted by entry type,
        # matching the default ncbi.nlm.nih.gov/gds search (Series/GSE makes
        # up the vast majority of current GEO records — standalone curated
        # DataSets/GDS records are rarely produced anymore).
        fc2, fc3 = st.columns(2)
        with fc2:
            organism_choice = st.selectbox("Organism", GEO_COMMON_ORGANISMS)
            organism_custom = ""
            if organism_choice == "Custom (type below)":
                organism_custom = st.text_input("Custom organism (NCBI Taxonomy name)",
                                                 placeholder="e.g. Sus scrofa")
        with fc3:
            supp_file = st.text_input(
                "Supplementary Files contain", placeholder="e.g. CEL, BAM, FASTQ",
                help="Maps to GEO's [Supplementary Files] field",
            )

        study_types = st.multiselect(
            "DataSet / Study Type", GEO_STUDY_TYPES,
            help="Maps to GEO's [DataSet Type] field — same fixed list as the GEO website. "
                 "Leave empty for any type.",
        )

        fc4, fc5 = st.columns(2)
        with fc4:
            use_sample_filter = st.checkbox("Filter by Number of Samples", value=False)
            sample_range = st.slider("Samples range", 1, 5000, (1, 5000),
                                      disabled=not use_sample_filter)
        with fc5:
            use_date_filter = st.checkbox("Filter by Publication Date", value=False)
            dc1, dc2 = st.columns(2)
            with dc1:
                pdat_from = st.text_input("From (YYYY/MM)", placeholder="2015/01",
                                           disabled=not use_date_filter)
            with dc2:
                pdat_to = st.text_input("To (YYYY/MM)", placeholder="2025/12",
                                         disabled=not use_date_filter)

    sort_choice = st.selectbox(
        "Sort by", ["Best Match (relevance)", "Most Recent", "Most Samples", "Title (A→Z)"],
    )

    st.caption(
        "ℹ️ Every matching result is fetched — the count will match "
        "ncbi.nlm.nih.gov/gds exactly, no matter how large. Very broad queries "
        "page through NCBI in batches, so they take longer."
    )

    run_geo = st.button("🔍 Search GEO Datasets")


    def _build_geo_query() -> str:
        """Build an Entrez query string using GEO's own field-tag syntax —
        identical to what the GEO Advanced Search Builder would produce."""
        terms = []
        if geo_query.strip():
            terms.append(geo_query.strip())
        if inc_eif:
            terms.append("EIF2AK")
        # Wrap each free-text term in parentheses before AND-ing, so any OR/NOT
        # the user typed keeps correct precedence (same rule GEO's own docs give).
        base = " AND ".join(f"({t})" for t in terms) if len(terms) > 1 else (terms[0] if terms else "")

        filters = []
        # No [Entry Type] filter — matches the default ncbi.nlm.nih.gov/gds
        # search, which spans DataSets, Series, Platforms and Samples together.
        organism_final = organism_custom.strip() if organism_choice == "Custom (type below)" else organism_choice
        if organism_final and organism_final != "Any organism":
            filters.append(f"{organism_final}[Organism]")
        if study_types:
            st_clause = " OR ".join(f'"{t}"[DataSet Type]' for t in study_types)
            filters.append(f"({st_clause})")
        if supp_file.strip():
            filters.append(f"{supp_file.strip()}[Supplementary Files]")
        if use_sample_filter:
            filters.append(f"{sample_range[0]}:{sample_range[1]}[Number of Samples]")
        if use_date_filter and pdat_from.strip() and pdat_to.strip():
            filters.append(f"{pdat_from.strip()}:{pdat_to.strip()}[Publication Date]")

        parts = ([base] if base else []) + filters
        return " AND ".join(parts)

    @st.cache_data(show_spinner=False)
    def search_geo(query: str):
        """
        Fetch EVERY matching GEO record for the query — the exact count NCBI
        itself reports, with no artificial cap, however large it is.

        Two-stage pagination keeps this reliable at any scale:
        - esearch IDs are paged 500 at a time via Entrez's history server
          (WebEnv/QueryKey), which avoids re-running the search query on
          every page and avoids URL-length limits on huge ID lists.
        - esummary is then called in batches of 200 IDs (NCBI's documented
          ceiling for a single summary request).
        A short sleep between calls keeps requests under the ~10/sec ceiling
        that comes with an API key. For very broad queries (tens of
        thousands of hits) this means fetching genuinely will take longer —
        there's no way to report the true total without retrieving it.
        """
        from Bio import Entrez as _Entrez
        _Entrez.email   = "samikshapasalkar2212@gmail.com"
        _Entrez.api_key = "9de22485baf54ae653d2825299784fcfb008"

        # Reachability check uses its own socket — it will NOT shrink the
        # timeout used by the real esearch/esummary calls below (that bug
        # used to abort large fetches partway through with a read timeout).
        if not _ncbi_reachable(timeout=5):
            st.error(
                "❌ **Cannot reach NCBI servers.** Check your internet connection or VPN."
            )
            return pd.DataFrame(), 0
        socket.setdefaulttimeout(NCBI_DATA_TIMEOUT)

        ID_PAGE  = 500   # esearch IDs per page (via history server)
        SUM_PAGE = 200   # esummary IDs per call

        try:
            # Establish the search on NCBI's history server and get the
            # true total count up front.
            handle = _entrez_call(_Entrez.esearch, db="gds", term=query,
                                   retmax=0, usehistory="y")
            record = _Entrez.read(handle); handle.close()
            total  = int(record.get("Count", 0))
            if total == 0:
                return pd.DataFrame(), 0
            webenv = record["WebEnv"]
            qkey   = record["QueryKey"]

            status = st.empty()

            # ── Page through ALL ids ──────────────────────────────────────
            all_ids = []
            for start in range(0, total, ID_PAGE):
                status.markdown(f"📡 Fetching IDs… {min(start + ID_PAGE, total):,} / {total:,}")
                handle = _entrez_call(_Entrez.esearch, db="gds", term=query,
                                       retstart=start, retmax=ID_PAGE,
                                       usehistory="y", webenv=webenv, query_key=qkey)
                rec = _Entrez.read(handle); handle.close()
                ids = rec.get("IdList", [])
                if not ids:
                    break
                all_ids.extend(ids)
                time.sleep(0.11)

            # ── Fetch summaries for every id ────────────────────────────────
            rows = []
            for i in range(0, len(all_ids), SUM_PAGE):
                status.markdown(
                    f"📋 Fetching record details… {min(i + SUM_PAGE, len(all_ids)):,} / {len(all_ids):,}"
                )
                chunk      = all_ids[i:i + SUM_PAGE]
                handle2    = _entrez_call(_Entrez.esummary, db="gds", id=",".join(chunk))
                summaries  = _Entrez.read(handle2); handle2.close()
                for s in summaries:
                    acc        = s.get("Accession", "")
                    entry_type = (s.get("entryType", "") or "").upper()
                    if not entry_type:
                        entry_type = ("GDS" if acc.startswith("GDS") else
                                      "GSE" if acc.startswith("GSE") else
                                      "GSM" if acc.startswith("GSM") else
                                      "GPL" if acc.startswith("GPL") else "?")
                    pmids = s.get("PubMedIds", [])
                    rows.append({
                        "Accession":  acc,
                        "Entry Type": entry_type,
                        "Title":      s.get("title", ""),
                        "Organism":   s.get("taxon", ""),
                        "Study Type": s.get("gdsType", ""),
                        "Platform":   str(s.get("GPL", "")),
                        "Samples":    s.get("n_samples", ""),
                        "Release":    s.get("PDAT", ""),
                        "Supp Files": s.get("suppFile", ""),
                        "PubMed":     f"{len(pmids)} linked" if pmids else "—",
                        "GEO Link":   f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc}",
                    })
                time.sleep(0.11)  # stay comfortably under the API-key rate limit

            status.empty()
            return pd.DataFrame(rows), total
        except Exception as e:
            st.error(f"GEO search error: {e}")
            return pd.DataFrame(), 0


    def _sort_results(df: pd.DataFrame, choice: str) -> pd.DataFrame:
        if df.empty:
            return df
        if choice == "Most Recent":
            return (df.assign(_d=pd.to_datetime(df["Release"], errors="coerce"))
                      .sort_values("_d", ascending=False).drop(columns="_d"))
        if choice == "Most Samples":
            return (df.assign(_n=pd.to_numeric(df["Samples"], errors="coerce"))
                      .sort_values("_n", ascending=False).drop(columns="_n"))
        if choice == "Title (A→Z)":
            return df.sort_values("Title", key=lambda s: s.str.lower())
        return df  # Best Match — keep NCBI's own relevance order

    DISPLAY_COLS = ["Accession", "Entry Type", "Title", "Organism", "Study Type",
                     "Platform", "Samples", "Release", "Supp Files", "PubMed",
                     "GEO Link"]
    COL_CONFIG = {
        "GEO Link": st.column_config.LinkColumn("GEO Link", display_text="View on GEO ↗"),
    }

    if run_geo:
        full_q = _build_geo_query()
        if not full_q.strip():
            st.warning("Enter a search term or enable at least one filter first.")
        else:
            st.info(f"🔎 Entrez query (same syntax GEO's own Advanced Search uses): `{full_q}`")
            with st.spinner("Searching GEO — fetching every matching result…"):
                geo_df, total = search_geo(full_q)
            geo_df = _sort_results(geo_df, sort_choice)
            st.session_state["geo_df"] = geo_df
            st.session_state["geo_total"] = total
            if geo_df.empty:
                st.warning(f"No results found ({total} raw hits). Try broadening your filters.")
            else:
                # Entry-type breakdown, mirroring the "Entry type" facet on the
                # ncbi.nlm.nih.gov/gds sidebar.
                et_counts = geo_df["Entry Type"].value_counts()
                et_badges = " &nbsp; ".join(
                    f"<b>{et}</b> ({n:,})" for et, n in et_counts.items()
                )
                st.markdown(f"""
                <div class="metric-row">
                  <div class="metric-box"><div class="metric-val">{total:,}</div>
                    <div class="metric-lbl">Total Results (exact NCBI count)</div></div>
                </div>
                <div class="card" style="font-size:0.82rem; color:#94a3b8;">
                  <div class="card-title">Entry type breakdown</div>
                  {et_badges}
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(geo_df[DISPLAY_COLS], use_container_width=True,

                             column_config=COL_CONFIG, hide_index=True)
                buf = io.StringIO(); geo_df[DISPLAY_COLS].to_csv(buf, index=False)
                st.download_button("⬇ Download Results (CSV)", buf.getvalue(),
                                    file_name="geo_search_results.csv", mime="text/csv",
                                    key="dl_geo_results")
    elif "geo_df" in st.session_state and not st.session_state["geo_df"].empty:
        st.info("Showing cached results — press Search to refresh.")
        geo_df = st.session_state["geo_df"]
        st.dataframe(geo_df[DISPLAY_COLS], use_container_width=True,
                     column_config=COL_CONFIG, hide_index=True)

    # ── 2-B  DATASET METADATA AGENT ───────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🤖 Dataset Metadata Agent</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <div class="card-title">How It Works</div>
      Enter a GSE or GDS accession. The agent will:<br>
      <ol style="margin:0.6rem 0 0 1rem; color:#9ca3af; font-size:0.87rem; line-height:1.8;">
        <li>Fetch dataset metadata from GEO via <code>GEOparse</code></li>
        <li>Analyse sample titles &amp; characteristics to detect Control / Disease groups</li>
      </ol>
    </div>
    """, unsafe_allow_html=True)

    acc_input = st.text_input("GEO Accession (e.g., GSE12345 or GDS1234)",
                               placeholder="GSE…")
    fetch_meta = st.button("🧠 Fetch Metadata")


    @st.cache_data(show_spinner=False)
    def fetch_geo_metadata(accession: str):
        """Use GEOparse to pull dataset metadata."""
        try:
            import GEOparse
            # silent mode
            import logging
            logging.getLogger("GEOparse").setLevel(logging.ERROR)
            gse = GEOparse.get_GEO(geo=accession, silent=True)
            return gse
        except Exception as e:
            return str(e)

    def analyse_gse(gse) -> dict:
        """
        Inspect a GEOparse GSE/GDS object and return a structured summary
        with detected columns, sample group labels, and supplementary file info.
        """
        info = {
            "title":      getattr(gse, "metadata", {}).get("title", ["Unknown"])[0],
            "summary":    getattr(gse, "metadata", {}).get("summary", [""])[0][:300],
            "organism":   getattr(gse, "metadata", {}).get("organism", ["Unknown"])[0],
            "platform":   getattr(gse, "metadata", {}).get("platform_id", ["Unknown"])[0],
            "samples":    {},
        }
        gsms = getattr(gse, "gsms", {})
        for sid, gsm in list(gsms.items())[:30]:
            title_s = gsm.metadata.get("title", [""])[0]
            char    = gsm.metadata.get("characteristics_ch1", [])
            source  = gsm.metadata.get("source_name_ch1", [""])[0]
            info["samples"][sid] = {
                "title":   title_s,
                "source":  source,
                "chars":   char,
            }
        return info

    def guess_groups(samples: dict) -> dict:
        """Heuristically guess Control vs Disease from sample titles / chars."""
        ctrl_kw = {"control","normal","healthy","wild","wt","untreated","vehicle","mock"}
        dis_kw  = {"disease","patient","treated","tumor","cancer","diabetes","infected",
                   "knockdown","knockout","mutant","overexpression","stimulated"}
        groups = {}
        for sid, meta in samples.items():
            label = (meta["title"] + " " + " ".join(meta["chars"])).lower()
            if any(k in label for k in ctrl_kw):
                groups[sid] = "control"
            elif any(k in label for k in dis_kw):
                groups[sid] = "disease"
            else:
                groups[sid] = "unknown"
        return groups

    if fetch_meta and acc_input.strip():
        acc = acc_input.strip().upper()
        with st.spinner(f"Fetching metadata for {acc}…"):
            gse_obj = fetch_geo_metadata(acc)

        if isinstance(gse_obj, str):
            st.error(f"GEOparse error: {gse_obj}")
        else:
            info   = analyse_gse(gse_obj)
            info["accession"] = acc
            groups = guess_groups(info["samples"])

            ctrl_n = sum(1 for g in groups.values() if g == "control")
            dis_n  = sum(1 for g in groups.values() if g == "disease")
            unk_n  = sum(1 for g in groups.values() if g == "unknown")

            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-box"><div class="metric-val">{acc}</div>
                <div class="metric-lbl">Accession</div></div>
              <div class="metric-box"><div class="metric-val">{len(info['samples'])}</div>
                <div class="metric-lbl">Samples</div></div>
              <div class="metric-box"><div class="metric-val">{ctrl_n}</div>
                <div class="metric-lbl">Control (detected)</div></div>
              <div class="metric-box"><div class="metric-val">{dis_n}</div>
                <div class="metric-lbl">Disease (detected)</div></div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📋 Dataset Metadata"):
                st.markdown(f"**Title:** {info['title']}")
                st.markdown(f"**Organism:** {info['organism']}  •  **Platform:** {info['platform']}")
                st.markdown(f"**Summary:** {info['summary']}")

            with st.expander("🧬 Sample Groups (AI-guessed)"):
                sample_data = [
                    {"Sample ID": sid, "Title": m["title"],
                     "Detected Group": groups.get(sid,"unknown"),
                     "Characteristics": "; ".join(m["chars"][:3])}
                    for sid, m in info["samples"].items()
                ]
                st.dataframe(pd.DataFrame(sample_data), use_container_width=True)

            st.session_state["geo_meta"]   = info
            st.session_state["geo_groups"] = groups



# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: TPM NORMALISATION  (multi-file upload → clean → TPM)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 TPM Normalisation":

    st.markdown("""
    <div class="hero-banner">
      <div class="hero-title">TPM Normalisation</div>
      <div class="hero-subtitle">
        Upload raw GEO files (any format) → auto-clean → download cleaned matrix →
        calculate TPM → EIF2AK spotlight
      </div>
      <div class="gene-chips">
        <span class="gene-chip">Step 1 · Upload &amp; Clean</span>
        <span class="gene-chip">Step 2 · Download Cleaned</span>
        <span class="gene-chip">Step 3 · Calculate TPM</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Helper: parse one uploaded file into a raw DataFrame
    # Handles the most common GEO supplementary file flavours:
    #   • plain CSV / TSV count tables (HTSeq, featureCounts, STAR ReadsPerGene)
    #   • soft-style tables (lines starting with !, ^, #  are comments)
    #   • GEO Series Matrix txt files (lines like "!Sample_title …")
    #   • files where the first numeric column is the gene ID index
    # ──────────────────────────────────────────────────────────────────────────
    def _parse_geo_file(f) -> pd.DataFrame | None:
        """Return a raw (genes × samples) DataFrame or None on failure."""
        name = f.name.lower()
        raw  = f.read()
        f.seek(0)

        # ── gz → decompress in memory ──────────────────────────────────────
        if name.endswith(".gz"):
            import gzip
            raw  = gzip.decompress(raw)
            name = name[:-3]          # strip .gz for extension sniffing below

        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            return None

        lines = text.splitlines()

        # ── GEO Series Matrix (.txt with "!…" header lines) ───────────────
        # These files embed a tab-delimited expression table between
        # "!series_matrix_table_begin" and "!series_matrix_table_end" markers.
        if any("series_matrix_table_begin" in l.lower() for l in lines):
            start = next(
                (i for i, l in enumerate(lines)
                 if "series_matrix_table_begin" in l.lower()), None
            )
            end = next(
                (i for i, l in enumerate(lines)
                 if "series_matrix_table_end" in l.lower()), None
            )
            if start is not None:
                data_lines = lines[start + 1 : end if end else None]
                text = "\n".join(data_lines)
            # fall through to TSV parsing below

        # ── Strip comment / metadata lines (!, ^, #) ─────────────────────
        data_lines = [l for l in text.splitlines()
                      if l.strip() and not l.startswith(("!", "^", "#"))]
        if not data_lines:
            return None
        text = "\n".join(data_lines)

        # ── Detect delimiter ──────────────────────────────────────────────
        first_data = data_lines[0]
        sep = "\t" if first_data.count("\t") >= first_data.count(",") else ","

        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, index_col=0,
                             header=0, low_memory=False)
        except Exception:
            return None

        # Drop columns that are entirely non-numeric (annotation columns)
        num_cols = [c for c in df.columns
                    if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.5]
        if not num_cols:
            return None
        return df[num_cols]


    def _clean_count_matrix(df: pd.DataFrame) -> pd.DataFrame:
        """
        Standard cleaning pipeline for a raw GEO count table:
        1. Force all values to numeric (coerce → NaN)
        2. Drop genes where > 80 % of values are NaN
        3. Fill remaining NaN with 0
        4. Drop genes with all-zero counts
        5. Remove duplicate gene IDs (keep first)
        6. Strip whitespace from gene names
        7. Convert floats that look like ints (e.g. 142.0 → 142)
        """
        df = df.copy()
        df.index = df.index.astype(str).str.strip()
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(thresh=int(0.2 * len(df.columns)))   # keep ≥ 20 % non-NaN
        df = df.fillna(0)
        df = df[df.sum(axis=1) > 0]                          # drop all-zero rows
        df = df[~df.index.duplicated(keep="first")]
        # Round to int where possible (count data is always integer)
        if (df % 1 == 0).all().all():
            df = df.astype(int)
        return df


    def _merge_files(dfs: list[pd.DataFrame],
                     names: list[str]) -> pd.DataFrame:
        """
        Merge multiple count DataFrames by gene index (outer join → fill 0).
        If a file has only one sample column, use the filename as sample name.
        """
        renamed = []
        for df, fname in zip(dfs, names):
            sample_stem = fname.rsplit(".", 1)[0].replace(".gz", "")
            if df.shape[1] == 1:
                df = df.rename(columns={df.columns[0]: sample_stem})
            renamed.append(df)
        merged = renamed[0]
        for other in renamed[1:]:
            merged = merged.join(other, how="outer")
        return merged.fillna(0)


    # ══════════════════════════════════════════════════════════════════════════
    # PHASE A — Upload raw files & produce cleaned matrix
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="step-indicator">
      <div class="step-num">1</div>
      <div class="step-label">Upload raw GEO count files (any format, multiple OK)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="font-size:0.83rem; color:#94a3b8; line-height:1.8;">
      <div class="card-title">Supported file types</div>
      <b style="color:#00ffcc;">CSV / TSV</b> — featureCounts, HTSeq, STAR ReadsPerGene<br>
      <b style="color:#00ffcc;">TXT</b> — any tab/comma-delimited count table or GEO Series Matrix<br>
      <b style="color:#00ffcc;">.gz</b> — any of the above gzip-compressed (common on GEO FTP)<br>
      Multi-file upload: each file = one or more samples; all will be merged by gene ID.
    </div>
    """, unsafe_allow_html=True)

    raw_uploads = st.file_uploader(
        "Drop raw count files here (CSV, TSV, TXT, .gz — multiple allowed)",
        type=["csv", "tsv", "txt", "gz"],
        accept_multiple_files=True,
        key="raw_upload_files",
    )

    if raw_uploads:
        parsed_dfs, file_names, parse_errors = [], [], []

        for uf in raw_uploads:
            df_raw = _parse_geo_file(uf)
            if df_raw is not None and not df_raw.empty:
                parsed_dfs.append(df_raw)
                file_names.append(uf.name)
            else:
                parse_errors.append(uf.name)

        if parse_errors:
            st.warning(
                f"⚠️ Could not parse {len(parse_errors)} file(s): "
                + ", ".join(parse_errors)
            )

        if not parsed_dfs:
            st.error("No parseable count files found. Check file format.")
            st.stop()

        # Merge multiple files
        if len(parsed_dfs) == 1:
            raw_merged = parsed_dfs[0]
        else:
            raw_merged = _merge_files(parsed_dfs, file_names)

        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-box"><div class="metric-val">{len(raw_uploads)}</div>
            <div class="metric-lbl">Files uploaded</div></div>
          <div class="metric-box"><div class="metric-val">{raw_merged.shape[0]:,}</div>
            <div class="metric-lbl">Raw genes</div></div>
          <div class="metric-box"><div class="metric-val">{raw_merged.shape[1]}</div>
            <div class="metric-lbl">Samples detected</div></div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("👁 Preview raw merged table (first 10 rows)"):
            st.dataframe(raw_merged.head(10), use_container_width=True)

        # ── Clean ──────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("""
        <div class="step-indicator">
          <div class="step-num">2</div>
          <div class="step-label">Auto-clean → preview → download cleaned matrix</div>
        </div>
        """, unsafe_allow_html=True)

        cleaned_df = _clean_count_matrix(raw_merged)

        dropped_genes = raw_merged.shape[0] - cleaned_df.shape[0]
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-box"><div class="metric-val">{cleaned_df.shape[0]:,}</div>
            <div class="metric-lbl">Genes after cleaning</div></div>
          <div class="metric-box"><div class="metric-val">{dropped_genes:,}</div>
            <div class="metric-lbl">Genes removed (zero / NaN)</div></div>
          <div class="metric-box"><div class="metric-val">{cleaned_df.shape[1]}</div>
            <div class="metric-lbl">Samples</div></div>
          <div class="metric-box"><div class="metric-val">
            {int(cleaned_df.values.sum()):,}</div>
            <div class="metric-lbl">Total counts</div></div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("👁 Preview cleaned matrix (first 15 rows)"):
            st.dataframe(cleaned_df.head(15), use_container_width=True)

        # Download cleaned CSV
        buf_clean = io.StringIO()
        cleaned_df.to_csv(buf_clean)
        st.download_button(
            "⬇️ Download Cleaned Count Matrix (CSV)",
            data=buf_clean.getvalue(),
            file_name="cleaned_count_matrix.csv",
            mime="text/csv",
            key="dl_cleaned",
        )

        # Stash in session for TPM step
        st.session_state["clean_counts"] = cleaned_df
        st.session_state["sample_names"] = list(cleaned_df.columns)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE B — TPM input: either from Phase A session or manual CSV upload
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("""
    <div class="step-indicator">
      <div class="step-num">3</div>
      <div class="step-label">Load cleaned matrix for TPM (from above or upload saved CSV)</div>
    </div>
    """, unsafe_allow_html=True)

    if "clean_counts" in st.session_state:
        counts_df    = st.session_state["clean_counts"]
        sample_names = list(counts_df.columns)
        st.success(
            f"✅ Ready: {counts_df.shape[0]:,} genes × {counts_df.shape[1]} samples"
        )
    else:
        st.info(
            "Upload your raw files above — or load a previously saved cleaned CSV here."
        )
        up_csv = st.file_uploader(
            "Upload saved cleaned count matrix (CSV, genes as rows)",
            type=["csv"],
            key="tpm_csv_upload",
        )
        if up_csv:
            try:
                counts_df    = pd.read_csv(up_csv, index_col=0)
                counts_df    = counts_df.apply(pd.to_numeric, errors="coerce").fillna(0)
                sample_names = list(counts_df.columns)
                st.session_state["clean_counts"] = counts_df
                st.session_state["sample_names"] = sample_names
                st.success(
                    f"✅ Loaded {counts_df.shape[0]:,} genes × {counts_df.shape[1]} samples"
                )
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                st.stop()
        else:
            st.stop()

    # ── Sample grouping ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div class="step-indicator">
      <div class="step-num">4</div>
      <div class="step-label">Label samples: Control / Disease / Exclude</div>
    </div>
    """, unsafe_allow_html=True)

    def _auto_label(name: str) -> str:
        n = name.lower()
        if any(k in n for k in ["ctrl","control","normal","healthy","wt","untreated","mock","vehicle"]):
            return "Control"
        if any(k in n for k in ["disease","treated","tumor","cancer","patient","kd","ko","mut","stim"]):
            return "Disease"
        return "Control"

    if ("group_df" not in st.session_state or
            set(st.session_state["group_df"]["Sample"].tolist()) != set(sample_names)):
        st.session_state["group_df"] = pd.DataFrame({
            "Sample": sample_names,
            "Group":  [_auto_label(s) for s in sample_names],
        })

    edited_groups = st.data_editor(
        st.session_state["group_df"],
        column_config={
            "Sample": st.column_config.TextColumn("Sample", disabled=True),
            "Group":  st.column_config.SelectboxColumn(
                "Group", options=["Control", "Disease", "Exclude"], required=True
            ),
        },
        use_container_width=True, num_rows="fixed", key="group_editor",
    )
    st.session_state["group_df"] = edited_groups

    valid_groups = edited_groups[edited_groups["Group"] != "Exclude"]
    ctrl_samples = valid_groups[valid_groups["Group"] == "Control"]["Sample"].tolist()
    dis_samples  = valid_groups[valid_groups["Group"] == "Disease"]["Sample"].tolist()
    c1, c2, c3 = st.columns(3)
    c1.metric("Control", len(ctrl_samples))
    c2.metric("Disease",  len(dis_samples))
    c3.metric("Excluded", len(sample_names) - len(ctrl_samples) - len(dis_samples))

    # ── Gene ID mapping ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div class="step-indicator">
      <div class="step-num">5</div>
      <div class="step-label">Gene ID mapping (Ensembl / Entrez → Symbol, optional)</div>
    </div>
    """, unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def _map_ids(gene_tuple: tuple):
        try:
            import mygene
            mg    = mygene.MyGeneInfo()
            probe = [g for g in gene_tuple[:15] if g]
            if not probe:
                return None
            if all(str(g).startswith("ENSG") for g in probe):
                scope = "ensembl.gene"
            elif all(str(g).isdigit() for g in probe):
                scope = "entrezgene"
            else:
                return None
            res = mg.querymany(list(gene_tuple), scopes=scope,
                               fields="symbol", species="human", returnall=False)
            return {r["query"]: r["symbol"] for r in res if "symbol" in r}
        except Exception as ex:
            st.warning(f"mygene: {ex}")
            return None

    map_toggle   = st.toggle("Auto-map gene IDs to symbols", value=True)
    gene_mapping = None
    if map_toggle:
        with st.spinner("Mapping gene IDs…"):
            gene_mapping = _map_ids(tuple(counts_df.index.tolist()))
        if gene_mapping:
            st.success(f"✅ Mapped {len(gene_mapping):,} IDs to gene symbols.")
        else:
            st.info("Gene IDs already look like symbols — skipping mapping.")

    # ── Gene lengths ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div class="step-indicator">
      <div class="step-num">6</div>
      <div class="step-label">Gene lengths (optional — default 1 000 bp)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="font-size:0.83rem; color:#94a3b8; line-height:1.8;">
      <div class="card-title">Why it matters</div>
      TPM corrects for both library size <em>and</em> gene length. Without real lengths
      every gene is assumed to be 1 000 bp — library-size correction still works
      but length bias is not removed. Upload a two-column CSV:
      <code>gene_id, length_bp</code>.
    </div>
    """, unsafe_allow_html=True)

    len_file        = st.file_uploader(
        "Gene lengths CSV  (gene_id, length_bp) — optional",
        type=["csv", "tsv", "txt"], key="len_upload_v2",
    )
    gene_lengths_bp = None
    if len_file:
        try:
            sep    = "\t" if len_file.name.endswith((".tsv", ".txt")) else ","
            len_df = pd.read_csv(len_file, sep=sep, index_col=0, header=0)
            len_df.columns = [c.strip().lower() for c in len_df.columns]
            cands  = [c for c in len_df.columns
                      if any(k in c for k in ["length", "len", "size", "bp"])]
            col    = cands[0] if cands else len_df.columns[0]
            gene_lengths_bp = pd.to_numeric(len_df[col], errors="coerce").dropna()
            st.success(f"✅ Gene lengths loaded for {len(gene_lengths_bp):,} genes.")
        except Exception as e:
            st.warning(f"Could not parse lengths file ({e}) — using 1 000 bp default.")
    else:
        st.info("No lengths file — using 1 000 bp for all genes.")

    # ── RUN TPM ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div class="step-indicator">
      <div class="step-num">7</div>
      <div class="step-label">Run TPM Normalisation</div>
    </div>
    """, unsafe_allow_html=True)

    run_tpm = st.button("🚀 Run TPM Normalisation", key="run_tpm_btn")

    if run_tpm:
        keep_cols = ctrl_samples + dis_samples
        if not keep_cols:
            st.error("Label at least one Control or Disease sample first.")
            st.stop()

        counts_sub = counts_df[keep_cols].copy()
        counts_sub = counts_sub.apply(pd.to_numeric, errors="coerce").fillna(0)

        if gene_mapping:
            counts_sub.index = [gene_mapping.get(g, g) for g in counts_sub.index]
        counts_sub = counts_sub[~counts_sub.index.duplicated(keep="first")]

        gl    = (gene_lengths_bp.reindex(counts_sub.index).fillna(1000)
                 if gene_lengths_bp is not None
                 else pd.Series(1000.0, index=counts_sub.index))
        gl_kb = gl / 1000.0

        with st.spinner("🔬 Calculating TPM…"):
            try:
                # RPK → scaling factor → TPM  (standard three-step formula)
                rpk     = counts_sub.div(gl_kb, axis=0)
                scaling = rpk.sum(axis=0) / 1e6
                tpm_df  = rpk.div(scaling, axis=1)
                tpm_df.index.name = "Gene"

                log2_tpm_df = np.log2(tpm_df + 1)
                log2_tpm_df.index.name = "Gene"

                # Per-group mean TPM + approximate log2 FC
                ctrl_cols = [c for c in ctrl_samples if c in tpm_df.columns]
                dis_cols  = [c for c in dis_samples  if c in tpm_df.columns]

                parts = {"Gene": tpm_df.index}
                if ctrl_cols:
                    parts["mean_TPM_control"] = tpm_df[ctrl_cols].mean(axis=1).values
                if dis_cols:
                    parts["mean_TPM_disease"] = tpm_df[dis_cols].mean(axis=1).values
                if ctrl_cols and dis_cols:
                    parts["log2_FC_approx"] = np.log2(
                        (tpm_df[dis_cols].mean(axis=1) + 1) /
                        (tpm_df[ctrl_cols].mean(axis=1) + 1)
                    ).values

                # Real per-gene significance testing on the actual TPM values
                # (Mann-Whitney U, exact/auto) + Benjamini-Hochberg FDR across
                # every gene that received a genuine test. See
                # compute_mannwhitney_bh() docstring for exactly what is and
                # isn't computed — nothing here is estimated or simulated.
                if ctrl_cols and dis_cols:
                    pvals, qvals, tested = compute_mannwhitney_bh(
                        tpm_df, ctrl_cols, dis_cols, min_n_per_group=2
                    )
                    parts["p_value_MannWhitney"] = pvals.values
                    parts["q_value_BH_FDR"]      = qvals.values
                    parts["n_tested"]            = tested.values

                summary_df = pd.DataFrame(parts).set_index("Gene")

                st.session_state["tpm_matrix"]  = tpm_df
                st.session_state["log2_tpm"]    = log2_tpm_df
                st.session_state["tpm_summary"] = summary_df
                st.session_state["tpm_ctrl_cols"] = ctrl_cols
                st.session_state["tpm_dis_cols"]  = dis_cols
                st.success("✅ TPM Normalisation complete!")

            except Exception as e:
                st.error(f"TPM calculation error: {e}")
                st.stop()

    # ── Display TPM Results ───────────────────────────────────────────────────
    if "tpm_matrix" in st.session_state:
        tpm_df     = st.session_state["tpm_matrix"]
        log2_tpm   = st.session_state["log2_tpm"]
        summary_df = st.session_state["tpm_summary"]
        ctrl_cols  = st.session_state.get("tpm_ctrl_cols", [])
        dis_cols   = st.session_state.get("tpm_dis_cols", [])

        _expressed = tpm_df.values[tpm_df.values > 0]
        med_tpm    = float(np.median(_expressed)) if _expressed.size > 0 else 0.0
        high_ex    = int((tpm_df.mean(axis=1) >= 10).sum())

        sig_metric_html = ""
        if "q_value_BH_FDR" in summary_df.columns:
            n_tested_total = int(summary_df.get("n_tested", pd.Series(dtype=int)).sum())
            n_sig = int((summary_df["q_value_BH_FDR"] < 0.05).sum())
            sig_metric_html = f"""
          <div class="metric-box"><div class="metric-val">{n_sig:,}</div>
            <div class="metric-lbl">Significant genes (q&lt;0.05, BH-FDR)</div></div>
          <div class="metric-box"><div class="metric-val">{n_tested_total:,}</div>
            <div class="metric-lbl">Genes actually tested</div></div>"""

        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-box"><div class="metric-val">{len(tpm_df):,}</div>
            <div class="metric-lbl">Total Genes</div></div>
          <div class="metric-box"><div class="metric-val">{len(tpm_df.columns)}</div>
            <div class="metric-lbl">Samples</div></div>
          <div class="metric-box"><div class="metric-val">{med_tpm:.1f}</div>
            <div class="metric-lbl">Median TPM (expressed)</div></div>
          <div class="metric-box"><div class="metric-val">{high_ex:,}</div>
            <div class="metric-lbl">Genes TPM ≥ 10</div></div>{sig_metric_html}
        </div>
        """, unsafe_allow_html=True)

        if "q_value_BH_FDR" in summary_df.columns and int(summary_df.get("n_tested", pd.Series(dtype=int)).sum()) < len(summary_df):
            st.caption(
                "ℹ️ Some genes show **N/A** for p-value / q-value — this means "
                "there weren't at least 2 real replicates in *both* Control and "
                "Disease for that gene (or every value was identical in both "
                "groups), so Mann-Whitney U genuinely cannot be computed. "
                "These are left blank rather than filled with a guessed number."
            )

        tab1, tab2, tab3 = st.tabs(
            ["📋 TPM Matrix", "📈 log₂(TPM+1)", "📊 Group Summary"]
        )

        with tab1:
            st.markdown('<div class="section-header">TPM Normalised Matrix</div>',
                        unsafe_allow_html=True)
            st.dataframe(tpm_df.round(3), use_container_width=True, height=380)
            b1 = io.StringIO(); tpm_df.to_csv(b1)
            st.download_button("⬇ Download TPM Matrix (CSV)", b1.getvalue(),
                               file_name="eif2ak_tpm_matrix.csv",
                               mime="text/csv", key="dl_tpm_matrix")

        with tab2:
            st.markdown('<div class="section-header">log₂(TPM + 1) Matrix</div>',
                        unsafe_allow_html=True)
            st.dataframe(log2_tpm.round(4), use_container_width=True, height=380)
            b2 = io.StringIO(); log2_tpm.to_csv(b2)
            st.download_button("⬇ Download log₂(TPM+1) CSV", b2.getvalue(),
                               file_name="eif2ak_log2tpm.csv",
                               mime="text/csv", key="dl_log2tpm")

        with tab3:
            st.markdown('<div class="section-header">Group Mean TPM Summary</div>',
                        unsafe_allow_html=True)
            has_stats = "q_value_BH_FDR" in summary_df.columns
            if has_stats:
                c1f, c2f, c3f = st.columns(3)
            else:
                c1f, c2f = st.columns(2)

            with c1f:
                min_tpm = st.slider("Min mean TPM (Control or Disease)",
                                    0.0, 50.0, 0.0, 0.5, key="min_tpm_f")
            with c2f:
                lfc_filter = (
                    st.slider("Min |log₂ FC|", 0.0, 5.0, 0.0, 0.25, key="lfc_f")
                    if "log2_FC_approx" in summary_df.columns else 0.0
                )
            sig_only = False
            if has_stats:
                with c3f:
                    sig_only = st.checkbox(
                        "Only significant (BH q < 0.05)", value=False, key="sig_only_f",
                        help="Filters to genes with a real Mann-Whitney U test "
                             "whose Benjamini-Hochberg adjusted q-value is below 0.05. "
                             "Genes with no valid test (NaN) are excluded by this filter."
                    )

            filt_sum = summary_df.copy()
            if "mean_TPM_control" in filt_sum and "mean_TPM_disease" in filt_sum:
                filt_sum = filt_sum[
                    (filt_sum["mean_TPM_control"] >= min_tpm) |
                    (filt_sum["mean_TPM_disease"]  >= min_tpm)
                ]
            if "log2_FC_approx" in filt_sum and lfc_filter > 0:
                filt_sum = filt_sum[filt_sum["log2_FC_approx"].abs() >= lfc_filter]
            if sig_only and "q_value_BH_FDR" in filt_sum:
                filt_sum = filt_sum[filt_sum["q_value_BH_FDR"] < 0.05]

            # p/q-values need scientific notation, not a flat .round(4) —
            # a real p-value of 3.2e-7 must not be silently displayed as 0.0000.
            display_sum = filt_sum.copy()
            fmt_map = {c: "{:.3f}" for c in display_sum.select_dtypes("number").columns}
            for pc in ("p_value_MannWhitney", "q_value_BH_FDR"):
                if pc in display_sum.columns:
                    fmt_map[pc] = "{:.3e}"
            if "n_tested" in display_sum.columns:
                fmt_map["n_tested"] = "{:.0f}"

            st.dataframe(
                display_sum.style.format(fmt_map, na_rep="N/A"),
                use_container_width=True, height=380,
            )
            st.caption(
                "p_value_MannWhitney = two-sided Mann-Whitney U test on the real "
                "per-sample TPM values (exact method for small tie-free samples, "
                "asymptotic otherwise — SciPy's own auto-selection, no shortcuts). "
                "q_value_BH_FDR = Benjamini-Hochberg FDR-adjusted p-value across all "
                "genes with a valid test. n_tested = 1 if that gene had ≥2 real "
                "replicates in both groups and could be tested, 0 if not (shown as N/A)."
            )
            b3 = io.StringIO(); summary_df.to_csv(b3)
            st.download_button("⬇ Download Summary CSV", b3.getvalue(),
                               file_name="eif2ak_tpm_summary.csv",
                               mime="text/csv", key="dl_tpm_sum")

            # ── CHART: Volcano plot (log₂FC vs -log10 q-value) ────────────────
            has_volcano_data = (
                "log2_FC_approx" in summary_df.columns
                and "q_value_BH_FDR" in summary_df.columns
            )
            if has_volcano_data:
                st.markdown("---")
                st.markdown(
                    '<div class="section-header">🌋 Volcano Plot — Fold Change vs Significance</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="card" style="font-size:0.83rem; color:#94a3b8; '
                    'line-height:1.7;">Every tested gene plotted by log₂ fold change '
                    '(x-axis) against -log₁₀(BH q-value) (y-axis). Genes above the '
                    'dashed line pass q&lt;0.05. EIF2AK1-4 are highlighted as gold '
                    'stars so you can see where they fall relative to the rest of '
                    'the transcriptome.</div>',
                    unsafe_allow_html=True,
                )
                try:
                    volc_df = summary_df.copy()
                    volc_df = volc_df[
                        volc_df["log2_FC_approx"].notna()
                        & volc_df["q_value_BH_FDR"].notna()
                        & (volc_df["q_value_BH_FDR"] > 0)
                    ]
                    if volc_df.empty:
                        st.info("No genes with both a valid log₂ FC and q-value to plot.")
                    else:
                        volc_eif2ak_map = find_eif2ak_rows(volc_df.index)
                        is_spot = volc_df.index.isin(volc_eif2ak_map.index)

                        neg_log10_q = -np.log10(volc_df["q_value_BH_FDR"].astype(float))
                        bg_mask = ~is_spot

                        volcano_fig = go.Figure()
                        volcano_fig.add_trace(go.Scattergl(
                            x=volc_df.loc[bg_mask, "log2_FC_approx"],
                            y=neg_log10_q[bg_mask],
                            mode="markers",
                            name="All genes",
                            marker=dict(size=5, color="#3b82f6", opacity=0.35,
                                        line=dict(width=0)),
                            text=volc_df.index[bg_mask],
                            hovertemplate="<b>%{text}</b><br>log₂FC: %{x:.3f}<br>"
                                          "-log₁₀(q): %{y:.3f}<extra></extra>",
                        ))
                        if is_spot.any():
                            spot_labels = volc_df.index[is_spot].map(
                                lambda lbl: volc_eif2ak_map.get(lbl, lbl)
                            )
                            volcano_fig.add_trace(go.Scatter(
                                x=volc_df.loc[is_spot, "log2_FC_approx"],
                                y=neg_log10_q[is_spot],
                                mode="markers+text",
                                name="EIF2AK1-4",
                                marker=dict(size=16, color="#facc15", symbol="star",
                                            line=dict(width=1, color="#050b18")),
                                text=spot_labels,
                                textposition="top center",
                                textfont=dict(color="#facc15", size=11),
                                hovertemplate="<b>%{text}</b><br>log₂FC: %{x:.3f}<br>"
                                              "-log₁₀(q): %{y:.3f}<extra></extra>",
                            ))
                        volcano_fig.add_hline(
                            y=-np.log10(0.05), line_dash="dash",
                            line_color="#94a3b8", line_width=1,
                            annotation_text="q = 0.05", annotation_font_color="#94a3b8",
                        )
                        volcano_fig.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="Sora, sans-serif", color="#dde4f0", size=12),
                            xaxis_title="log₂ Fold Change (Disease vs Control, approx)",
                            yaxis_title="-log₁₀(BH q-value)",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                        xanchor="left", x=0),
                            height=480,
                            margin=dict(l=60, r=20, t=50, b=50),
                        )
                        volcano_fig.update_xaxes(gridcolor="#1a2d45")
                        volcano_fig.update_yaxes(gridcolor="#1a2d45")
                        st.plotly_chart(volcano_fig, use_container_width=True)
                        st.caption(
                            "⭐ Gold stars = EIF2AK1-4  •  Dashed line = q = 0.05 significance threshold"
                        )
                except Exception as _volc_err:
                    st.warning(f"Volcano plot error: {_volc_err}")

            # ── CHART: EIF2AK-specific significance bar chart ──────────────────
            has_eif2ak_sig_data = (
                "q_value_BH_FDR" in summary_df.columns
            )
            if has_eif2ak_sig_data:
                st.markdown("---")
                st.markdown(
                    '<div class="section-header">📌 EIF2AK Family — Significance (-log₁₀ q-value)</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="card" style="font-size:0.83rem; color:#94a3b8; '
                    'line-height:1.7;">A dedicated view of just EIF2AK1-4, since '
                    'they can be hard to read on the full volcano plot above. Bars '
                    'above the dashed line pass q&lt;0.05 (BH-FDR).</div>',
                    unsafe_allow_html=True,
                )
                try:
                    sig_eif2ak_map = find_eif2ak_rows(summary_df.index)
                    if sig_eif2ak_map.empty:
                        st.info("No EIF2AK genes detected in the summary table.")
                    else:
                        sig_rows = summary_df.loc[sig_eif2ak_map.index].copy()
                        sig_rows.index = sig_eif2ak_map.values
                        sig_rows = sig_rows[~sig_rows.index.duplicated(keep="first")]
                        sig_rows = sig_rows.reindex(
                            [g for g in EIF2AK_GENES if g in sig_rows.index]
                        )

                        q_series = pd.to_numeric(
                            sig_rows.get("q_value_BH_FDR", pd.Series(dtype=float)),
                            errors="coerce",
                        )
                        valid = q_series.notna() & (q_series > 0)
                        bar_x = sig_rows.index[valid]
                        bar_y = -np.log10(q_series[valid].astype(float))
                        bar_colors = ["#22c55e" if q_series[g] < 0.05 else "#64748b"
                                      for g in bar_x]

                        if len(bar_x) == 0:
                            st.info(
                                "None of EIF2AK1-4 have a valid q-value (not enough "
                                "replicates in both groups to test)."
                            )
                        else:
                            sig_bar_fig = go.Figure()
                            sig_bar_fig.add_trace(go.Bar(
                                x=list(bar_x),
                                y=bar_y.tolist(),
                                marker_color=bar_colors,
                                text=[f"q={q_series[g]:.2e}" for g in bar_x],
                                textposition="outside",
                                hovertemplate="<b>%{x}</b><br>-log₁₀(q): %{y:.3f}<extra></extra>",
                            ))
                            sig_bar_fig.add_hline(
                                y=-np.log10(0.05), line_dash="dash",
                                line_color="#94a3b8", line_width=1,
                                annotation_text="q = 0.05", annotation_font_color="#94a3b8",
                            )
                            sig_bar_fig.update_layout(
                                template="plotly_dark",
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(family="Sora, sans-serif", color="#dde4f0", size=12),
                                xaxis_title="Gene",
                                yaxis_title="-log₁₀(BH q-value)",
                                showlegend=False,
                                height=380,
                                margin=dict(l=60, r=20, t=50, b=50),
                            )
                            sig_bar_fig.update_xaxes(gridcolor="#1a2d45")
                            sig_bar_fig.update_yaxes(gridcolor="#1a2d45")
                            st.plotly_chart(sig_bar_fig, use_container_width=True)
                            st.caption(
                                "🟢 Green = q &lt; 0.05  •  ⚪ Grey = not significant  •  "
                                "Dashed line = q = 0.05 threshold"
                            )
                except Exception as _sigbar_err:
                    st.warning(f"EIF2AK significance bar chart error: {_sigbar_err}")

            # ── CHART: p-value histogram (diagnostic QC) ────────────────────────
            if "p_value_MannWhitney" in summary_df.columns:
                with st.expander("🔬 Diagnostic: p-value distribution (QC)"):
                    st.markdown(
                        '<div style="font-size:0.83rem; color:#94a3b8; line-height:1.7;">'
                        'Histogram of raw Mann-Whitney p-values across all tested genes. '
                        'Under the null hypothesis (no real differences) this should look '
                        'roughly flat/uniform. A peak near 0 suggests genuine differential '
                        'expression signal in the dataset; a flat or right-skewed shape '
                        'suggests little real signal, or that assumptions of the test may '
                        'be violated.</div>',
                        unsafe_allow_html=True,
                    )
                    try:
                        pvals_all = pd.to_numeric(
                            summary_df["p_value_MannWhitney"], errors="coerce"
                        ).dropna()
                        if pvals_all.empty:
                            st.info("No valid p-values available to plot.")
                        else:
                            hist_fig = go.Figure()
                            hist_fig.add_trace(go.Histogram(
                                x=pvals_all,
                                xbins=dict(start=0.0, end=1.0, size=0.05),
                                marker_color="#818cf8",
                                opacity=0.85,
                                hovertemplate="p-range: %{x}<br>Count: %{y}<extra></extra>",
                            ))
                            hist_fig.update_layout(
                                template="plotly_dark",
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(family="Sora, sans-serif", color="#dde4f0", size=12),
                                xaxis_title="Mann-Whitney p-value",
                                yaxis_title="Number of genes",
                                bargap=0.05,
                                height=340,
                                margin=dict(l=60, r=20, t=30, b=50),
                            )
                            hist_fig.update_xaxes(gridcolor="#1a2d45", range=[0, 1])
                            hist_fig.update_yaxes(gridcolor="#1a2d45")
                            st.plotly_chart(hist_fig, use_container_width=True)
                            st.caption(
                                f"n = {len(pvals_all):,} genes with a valid test  •  "
                                "Flat ≈ little signal  •  Peak near 0 ≈ real differential expression"
                            )
                    except Exception as _hist_err:
                        st.warning(f"P-value histogram error: {_hist_err}")

        # ── EIF2AK Spotlight ──────────────────────────────────────────────────
        st.markdown('<div class="section-header">🌟 EIF2AK Gene Spotlight (TPM)</div>',
                    unsafe_allow_html=True)

        # ── Robust alias-based EIF2AK matching ────────────────────────────────
        # find_eif2ak_rows scans every row label through the full alias/Ensembl
        # dictionary, handling version suffixes, compound labels, bare Entrez IDs,
        # HGNC IDs and any case variation.
        eif2ak_id_map = find_eif2ak_rows(summary_df.index)
        # eif2ak_id_map: original_label → canonical_symbol (EIF2AK1-4)

        if eif2ak_id_map.empty:
            st.info(
                "No EIF2AK genes detected with any known alias or ID "
                "(symbols, PERK/PKR/HRI/GCN2, Ensembl IDs, Entrez IDs…). "
                "Ensure your count matrix contains EIF2AK family genes."
            )
        else:
            # Build the spotlight table with canonical symbols as the index
            # and a 'Matched As' column showing what the original label was.
            spotlight_rows = summary_df.loc[eif2ak_id_map.index].copy()
            spotlight_rows.insert(0, "Matched As", eif2ak_id_map.values)
            spotlight_rows.index = eif2ak_id_map.values   # canonical symbols
            spotlight_rows.index.name = "Canonical Gene"
            spotlight_rows = spotlight_rows[~spotlight_rows.index.duplicated(keep="first")]

            # canon → original label (for downstream tpm_df row-lookup)
            canon_to_original = dict(zip(eif2ak_id_map.values, eif2ak_id_map.index))

            def _hi_expr(row):
                try:
                    vals = [v for k, v in row.items()
                            if isinstance(k, str) and "tpm" in k.lower()
                            and isinstance(v, (int, float))]
                    if vals and max(vals) >= 1:
                        return (["background-color:rgba(16,185,129,0.18);"
                                  "color:#10b981;font-weight:600"] * len(row))
                except Exception:
                    pass
                return [""] * len(row)

            _spot_fmt = {c: "{:.3f}" for c in spotlight_rows.select_dtypes("number").columns}
            for _pc in ("p_value_MannWhitney", "q_value_BH_FDR"):
                if _pc in spotlight_rows.columns:
                    _spot_fmt[_pc] = "{:.3e}"
            if "n_tested" in spotlight_rows.columns:
                _spot_fmt["n_tested"] = "{:.0f}"

            st.dataframe(
                spotlight_rows.style.apply(_hi_expr, axis=1).format(_spot_fmt, na_rep="N/A"),
                use_container_width=True,
            )
            b4 = io.StringIO(); spotlight_rows.to_csv(b4)
            st.download_button("⬇ Download EIF2AK Spotlight CSV", b4.getvalue(),
                               file_name="eif2ak_spotlight.csv",
                               mime="text/csv", key="dl_spot")

            st.markdown('<div class="section-header">📊 Gene-level TPM Cards</div>',
                        unsafe_allow_html=True)
            spot_r    = spotlight_rows.reset_index()
            gene_cols_ui = st.columns(min(len(spot_r), 4))
            for i, row in spot_r.iterrows():
                gene_name = row.get("Canonical Gene", str(row.name))
                ctrl_tpm  = row.get("mean_TPM_control", None)
                dis_tpm   = row.get("mean_TPM_disease",  None)
                lfc_val   = row.get("log2_FC_approx",   None)
                p_val     = row.get("p_value_MannWhitney", None)
                q_val     = row.get("q_value_BH_FDR",       None)
                matched_as = row.get("Matched As", gene_name)
                alias_note = (f"<div style='font-size:0.67rem;color:#5a6a80;"
                              f"margin-bottom:0.3rem;'>id: {matched_as}</div>"
                              if matched_as != gene_name else "")

                try:
                    ctrl_s = f"{float(ctrl_tpm):.2f}" if ctrl_tpm is not None else "N/A"
                    dis_s  = f"{float(dis_tpm):.2f}"  if dis_tpm  is not None else "N/A"
                except (TypeError, ValueError):
                    ctrl_s = dis_s = "N/A"

                try:
                    lfc_f   = float(lfc_val)
                    lfc_s   = f"{lfc_f:+.3f}"
                    lfc_col = "#10b981" if lfc_f > 0 else "#ef4444"
                    expr_l  = "🟢 Up" if lfc_f > 0.5 else ("🔴 Down" if lfc_f < -0.5 else "⚪ Stable")
                except (TypeError, ValueError):
                    lfc_s   = "N/A"
                    lfc_col = "#00ffcc"
                    expr_l  = "❓ Unknown"

                try:
                    q_f = float(q_val)
                    p_f = float(p_val)
                    if q_f < 0.001:
                        stars = "***"
                    elif q_f < 0.01:
                        stars = "**"
                    elif q_f < 0.05:
                        stars = "*"
                    else:
                        stars = "ns"
                    stats_line = (
                        f"<div style='font-size:0.72rem;color:#94a3b8;margin-top:0.3rem;'>"
                        f"p={p_f:.2e} · q={q_f:.2e} <b>{stars}</b></div>"
                    )
                except (TypeError, ValueError):
                    stats_line = (
                        "<div style='font-size:0.72rem;color:#5a6a80;margin-top:0.3rem;'>"
                        "p/q: N/A (not enough replicates to test)</div>"
                    )

                with gene_cols_ui[i % 4]:
                    st.markdown(f"""
                    <div class="gene-spotlight">
                      <div class="gene-name">{gene_name}</div>
                      {alias_note}
                      <div class="gene-lfc" style="color:{lfc_col};">{lfc_s}</div>
                      <div style="font-size:0.7rem;color:#5a6a80;margin:0.15rem 0 0.4rem 0;">
                        log₂ FC (approx)</div>
                      <div style="font-size:0.77rem;color:#94a3b8;">Ctrl TPM: {ctrl_s}</div>
                      <div style="font-size:0.77rem;color:#94a3b8;">Dis TPM:  {dis_s}</div>
                      <div style="margin-top:0.35rem;font-size:0.8rem;">{expr_l}</div>
                      {stats_line}
                    </div>
                    """, unsafe_allow_html=True)

            # ── CHART 1: EIF2AK 4-gene violin — Control vs Disease ────────────
            st.markdown("---")
            st.markdown(
                '<div class="section-header">🎻 EIF2AK Family — TPM Distribution by Group</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="card" style="font-size:0.83rem; color:#94a3b8; '
                'line-height:1.7;">Split violin per gene: Control (left, blue) vs '
                'Disease (right, pink). Each dot is one sample. Mean line shown '
                'inside each violin.</div>',
                unsafe_allow_html=True,
            )

            spot_genes = list(spotlight_rows.index)   # canonical EIF2AK names
            if not (ctrl_cols and dis_cols):
                st.info(
                    "Both Control and Disease samples are needed for the violin. "
                    "Label samples in both groups to unlock this chart."
                )
            else:
                try:
                    gene_fig = go.Figure()
                    group_defs = [("Control", ctrl_cols, "#60a5fa"),
                                  ("Disease",  dis_cols,  "#f472b6")]

                    for grp_name, grp_cols, grp_color in group_defs:
                        grp_vals, grp_genes = [], []
                        for canon in spot_genes:
                            orig = canon_to_original.get(canon, canon)
                            if orig in tpm_df.index:
                                # log2(TPM+1) so zeros don't collapse the violin
                                vals = np.log2(
                                    tpm_df.loc[orig, grp_cols].astype(float).values + 1
                                )
                                grp_vals.extend(vals.tolist())
                                grp_genes.extend([canon] * len(vals))

                        gene_fig.add_trace(go.Violin(
                            x=grp_genes,
                            y=grp_vals,
                            name=grp_name,
                            legendgroup=grp_name,
                            side="negative" if grp_name == "Control" else "positive",
                            line_color=grp_color,
                            fillcolor=grp_color,
                            opacity=0.65,
                            points="all",
                            pointpos=-0.8 if grp_name == "Control" else 0.8,
                            jitter=0.3,
                            marker=dict(size=6, color=grp_color,
                                        line=dict(width=0.5, color="#050b18")),
                            meanline_visible=True,
                            box_visible=True,
                            box_width=0.15,
                            scalemode="count",
                        ))

                    gene_fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Sora, sans-serif", color="#dde4f0", size=12),
                        yaxis_title="log₂(TPM + 1)",
                        xaxis_title="Gene",
                        violingap=0.3,
                        violingroupgap=0.1,
                        violinmode="group",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                    xanchor="left", x=0),
                        height=460,
                        margin=dict(l=60, r=20, t=50, b=50),
                    )
                    gene_fig.update_xaxes(gridcolor="#1a2d45")
                    gene_fig.update_yaxes(gridcolor="#1a2d45")
                    st.plotly_chart(gene_fig, use_container_width=True)
                    st.caption("🔵 Left = Control  •  🩷 Right = Disease  •  Inner box = IQR  •  Line = mean")
                except Exception as _vio_err:
                    st.warning(f"Violin chart error: {_vio_err}")

            # ── CHART 2: EIF2AK grouped bar — mean TPM Control vs Disease ─────
            st.markdown("---")
            st.markdown(
                '<div class="section-header">📊 EIF2AK Family — Mean TPM: Control vs Disease</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="card" style="font-size:0.83rem; color:#94a3b8; '
                'line-height:1.7;">Mean TPM for each EIF2AK gene side-by-side for '
                'Control (blue) and Disease (pink). Error bars show ±1 SD across '
                'samples in each group.</div>',
                unsafe_allow_html=True,
            )

            if not spot_genes:
                st.info("No EIF2AK genes detected in the TPM matrix.")
            else:
                try:
                    bar_genes, ctrl_means, ctrl_sds, dis_means, dis_sds = [], [], [], [], []
                    for canon in spot_genes:
                        orig = canon_to_original.get(canon, canon)
                        if orig not in tpm_df.index:
                            continue
                        bar_genes.append(canon)
                        if ctrl_cols:
                            cv = tpm_df.loc[orig, ctrl_cols].astype(float).values
                            ctrl_means.append(float(cv.mean()))
                            ctrl_sds.append(float(cv.std(ddof=1)) if len(cv) > 1 else 0.0)
                        else:
                            ctrl_means.append(0.0); ctrl_sds.append(0.0)
                        if dis_cols:
                            dv = tpm_df.loc[orig, dis_cols].astype(float).values
                            dis_means.append(float(dv.mean()))
                            dis_sds.append(float(dv.std(ddof=1)) if len(dv) > 1 else 0.0)
                        else:
                            dis_means.append(0.0); dis_sds.append(0.0)

                    bar_fig = go.Figure()
                    bar_fig.add_trace(go.Bar(
                        name="Control",
                        x=bar_genes,
                        y=ctrl_means,
                        error_y=dict(type="data", array=ctrl_sds, visible=True,
                                     color="#60a5fa", thickness=1.5, width=6),
                        marker_color="#60a5fa",
                        opacity=0.85,
                        hovertemplate="<b>%{x}</b> — Control<br>Mean TPM: %{y:.2f}<extra></extra>",
                    ))
                    bar_fig.add_trace(go.Bar(
                        name="Disease",
                        x=bar_genes,
                        y=dis_means,
                        error_y=dict(type="data", array=dis_sds, visible=True,
                                     color="#f472b6", thickness=1.5, width=6),
                        marker_color="#f472b6",
                        opacity=0.85,
                        hovertemplate="<b>%{x}</b> — Disease<br>Mean TPM: %{y:.2f}<extra></extra>",
                    ))
                    bar_fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Sora, sans-serif", color="#dde4f0", size=12),
                        barmode="group",
                        bargap=0.25,
                        bargroupgap=0.08,
                        yaxis_title="Mean TPM",
                        xaxis_title="Gene",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                    xanchor="left", x=0),
                        height=400,
                        margin=dict(l=60, r=20, t=50, b=50),
                    )
                    bar_fig.update_xaxes(gridcolor="#1a2d45")
                    bar_fig.update_yaxes(gridcolor="#1a2d45")
                    st.plotly_chart(bar_fig, use_container_width=True)
                    st.caption("Error bars = ±1 SD across samples in each group")
                except Exception as _bar_err:
                    st.warning(f"Bar chart error: {_bar_err}")

            # ── CHART 3: Strip / Dot plot — every real sample shown ───────────
            st.markdown("---")
            st.markdown(
                '<div class="section-header">🔵 EIF2AK Family — Strip Plot (Individual Samples)</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="card" style="font-size:0.83rem; color:#94a3b8; '
                'line-height:1.7;">Every dot is one real sample — no smoothing, '
                'no estimation. The horizontal bar marks the group mean. '
                'With small sample sizes this is the most honest representation '
                'of your actual data.</div>',
                unsafe_allow_html=True,
            )

            if not spot_genes:
                st.info("No EIF2AK genes detected in the TPM matrix.")
            else:
                try:
                    strip_fig = go.Figure()
                    group_defs_s = [("Control", ctrl_cols, "#60a5fa"),
                                    ("Disease",  dis_cols,  "#f472b6")]

                    for grp_name, grp_cols, grp_color in group_defs_s:
                        if not grp_cols:
                            continue
                        for canon in spot_genes:
                            orig = canon_to_original.get(canon, canon)
                            if orig not in tpm_df.index:
                                continue
                            vals = np.log2(
                                tpm_df.loc[orig, grp_cols].astype(float).values + 1
                            )
                            # offset Control left, Disease right within each gene
                            x_offset = -0.18 if grp_name == "Control" else 0.18
                            x_jitter = np.random.uniform(-0.06, 0.06, size=len(vals))
                            x_pos = [canon] * len(vals)

                            # individual dots
                            strip_fig.add_trace(go.Scatter(
                                x=[f"{canon}" for _ in vals],
                                y=vals.tolist(),
                                mode="markers",
                                name=grp_name,
                                legendgroup=grp_name,
                                showlegend=(canon == spot_genes[0]),
                                marker=dict(
                                    color=grp_color,
                                    size=9,
                                    opacity=0.85,
                                    line=dict(width=0.8, color="#050b18"),
                                    symbol="circle",
                                ),
                                hovertemplate=(
                                    f"<b>{canon}</b> — {grp_name}<br>"
                                    "log₂(TPM+1): %{y:.3f}<extra></extra>"
                                ),
                                offsetgroup=grp_name,
                            ))

                            # mean crossbar
                            strip_fig.add_shape(
                                type="line",
                                xref="x", yref="y",
                                x0=canon, x1=canon,
                                y0=float(vals.mean()), y1=float(vals.mean()),
                                line=dict(color=grp_color, width=3),
                            )

                    strip_fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Sora, sans-serif", color="#dde4f0", size=12),
                        yaxis_title="log₂(TPM + 1)",
                        xaxis_title="Gene",
                        boxmode="group",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                    xanchor="left", x=0),
                        height=420,
                        margin=dict(l=60, r=20, t=50, b=50),
                    )
                    strip_fig.update_xaxes(gridcolor="#1a2d45")
                    strip_fig.update_yaxes(gridcolor="#1a2d45")
                    st.plotly_chart(strip_fig, use_container_width=True)
                    st.caption("Each dot = one sample  •  Horizontal bar = group mean")
                except Exception as _strip_err:
                    st.warning(f"Strip plot error: {_strip_err}")


