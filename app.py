import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import re
import io
import time
import textwrap
from Bio import Entrez, Medline

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
HF_TOKEN       = "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"
EIF2AK_GENES   = ["EIF2AK1", "EIF2AK2", "EIF2AK3", "EIF2AK4"]
BIOMEDBERT_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

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
    ["📄 Literature Evidence", "🔬 GEO AI-Agent & Explorer", "📊 Differential Expression"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size:0.74rem; color:#475569; padding:0.4rem 0;">
<b style="color:#94a3b8;">Gene Family</b><br>
EIF2AK1 · EIF2AK2<br>EIF2AK3 · EIF2AK4<br><br>
<b style="color:#94a3b8;">Integrated Services</b><br>
PubMed · BioBERT<br>GEO · GEOparse<br>PyDESeq2 · mygene
</div>
""", unsafe_allow_html=True)

if st.sidebar.checkbox("🛠 Debug: Session Keys", value=False):
    st.sidebar.write(list(st.session_state.keys()))
    if "clean_counts" in st.session_state:
        s = st.session_state["clean_counts"]
        st.sidebar.success(f"clean_counts: {s.shape[0]}g × {s.shape[1]}s")

# ── Network connectivity indicator ────────────────────────────────────────────
import socket as _socket
try:
    _socket.setdefaulttimeout(3)
    _socket.getaddrinfo("eutils.ncbi.nlm.nih.gov", 443)
    st.sidebar.markdown(
        '<div style="font-size:0.73rem; color:#10b981; margin-top:0.5rem;">🟢 NCBI reachable</div>',
        unsafe_allow_html=True
    )
except OSError:
    st.sidebar.markdown(
        '<div style="font-size:0.73rem; color:#ef4444; margin-top:0.5rem;">'
        '🔴 NCBI unreachable — check network/VPN</div>',
        unsafe_allow_html=True
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: LITERATURE EVIDENCE
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📄 Literature Evidence":

    st.markdown("""
    <div class="hero-banner">
      <div class="hero-title">Literature Evidence Mining</div>
      <div class="hero-subtitle">
        PubMed full-text abstracts · BioBERT semantic similarity · Ranked evidence extraction
      </div>
      <div class="gene-chips">
        <span class="gene-chip">EIF2AK1</span>
        <span class="gene-chip">EIF2AK2</span>
        <span class="gene-chip">EIF2AK3</span>
        <span class="gene-chip">EIF2AK4</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 2])
    with col1:
        disease = st.text_input(
            "🦠 Disease / Condition",
            placeholder="Leave blank to search ALL diseases",
            help="Leave blank to search across all disease types for selected EIF2AK genes",
        )
    with col2:
        gene_choice = st.multiselect("🧬 Gene(s)", EIF2AK_GENES, default=EIF2AK_GENES)
    drug = ""  # Drug field removed

    if not disease:
        st.info("💡 No disease entered — will search **all disease contexts** for selected EIF2AK genes.")

    max_results = st.slider("Max abstracts to fetch", 50, 1000, 100)
    run_lit = st.button("🔍 Search & Extract Evidence")

    # ── PubMed fetch ──────────────────────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def fetch_pubmed(query: str, max_r: int = 20):
        import socket
        from Bio import Entrez as _Entrez, Medline as _Medline
        # Re-apply credentials inside cached function (avoids cache/env issues)
        _Entrez.email   = "samikshapasalkar2212@gmail.com"
        _Entrez.api_key = "9de22485baf54ae653d2825299784fcfb008"
        # Quick connectivity check
        try:
            socket.setdefaulttimeout(5)
            socket.getaddrinfo("eutils.ncbi.nlm.nih.gov", 443)
        except OSError:
            st.error(
                "❌ **Cannot reach NCBI servers.**\n\n"
                "Possible causes:\n"
                "- No internet connection\n"
                "- VPN / firewall blocking outbound traffic\n"
                "- Corporate / university network restrictions\n\n"
                "Try switching networks or disabling VPN, then search again."
            )
            return []
        try:
            handle = _Entrez.esearch(db="pubmed", term=query, retmax=max_r, sort="relevance")
            record = _Entrez.read(handle); handle.close()
            ids = record.get("IdList", [])
            if not ids:
                return []
            handle2 = _Entrez.efetch(db="pubmed", id=",".join(ids),
                                     rettype="medline", retmode="text")
            records = list(_Medline.parse(handle2)); handle2.close()
            return records
        except Exception as e:
            st.error(f"PubMed fetch error: {e}")
            return []

    # ── Sentence splitter ─────────────────────────────────────────────────────
    def split_sentences(text: str):
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sents if len(s.strip()) > 20]

    # ── BioM-ELECTRA QA pipeline loader (cached across session) ──────────────
    @st.cache_resource(show_spinner=False)
    def load_qa_pipeline():
        try:
            from transformers import pipeline as hf_pipeline
            import torch
            model_name = "sultan/BioM-ELECTRA-Large-SQuAD2"
            qa = hf_pipeline(
                "question-answering",
                model=model_name,
                tokenizer=model_name,
                device=0 if torch.cuda.is_available() else -1,
            )
            return qa, "bioelectra"
        except Exception as e1:
            # Fallback to a lighter distilbert biomedical QA model
            try:
                from transformers import pipeline as hf_pipeline
                import torch
                model_name = "distilbert-base-cased-distilled-squad"
                qa = hf_pipeline(
                    "question-answering",
                    model=model_name,
                    tokenizer=model_name,
                    device=0 if torch.cuda.is_available() else -1,
                )
                return qa, "distilbert"
            except Exception as e2:
                return None, f"QA model unavailable ({e1})"

    # ── Keyword-based fallback evidence extractor (no model needed) ──────────
    def extract_evidence_keywords(abstract: str, genes: list, disease: str) -> str:
        """
        Rule-based evidence extractor that works even without a QA model.
        Finds the most relevant sentence(s) mentioning the gene(s) and/or disease.
        Returns a human-readable evidence snippet.
        """
        if not abstract or len(abstract.strip()) < 30:
            return "No abstract available"

        sentences = split_sentences(abstract)
        gene_terms  = [g.lower() for g in genes] + ["eif2ak", "eif2", "kinase"]
        dis_terms   = [d.strip().lower() for d in disease.split()] if disease else []

        scored = []
        for sent in sentences:
            sl = sent.lower()
            gene_hits = sum(1 for t in gene_terms if t in sl)
            dis_hits  = sum(1 for t in dis_terms  if t in sl) if dis_terms else 0
            # Bonus for mechanistic / association language
            mech_hits = sum(1 for kw in [
                "regulates", "activates", "inhibits", "expression", "mutation",
                "associated", "involved", "pathway", "signaling", "phosphorylation",
                "upregulated", "downregulated", "overexpressed", "stress response",
                "role", "function", "mediates", "promotes", "suppresses",
                "linked", "implicated", "encodes", "protein kinase",
            ] if kw in sl)
            score = gene_hits * 3 + dis_hits * 2 + mech_hits
            if score > 0:
                scored.append((score, sent))

        if not scored:
            # Last resort: return the first two sentences as context
            fallback = " ".join(sentences[:2]) if sentences else abstract[:300]
            return f"[Keyword match] {fallback}"

        scored.sort(key=lambda x: x[0], reverse=True)
        # Return top 1–2 most relevant sentences
        top = scored[:2]
        snippet = " ".join(s for _, s in top)
        return f"[Keyword match] {snippet}"

    def find_evidence_qa(qa_pipeline_tuple, abstract: str, question: str,
                         genes: list, disease: str,
                         min_score: float = 0.01, max_context: int = 3000) -> str:
        """
        Extract evidence from abstract using QA model with keyword fallback.
        Never returns bare 'N/A' — always provides some evidence text.
        """
        if not abstract or len(abstract.strip()) < 30:
            return "No abstract available"

        qa_model, model_tag = qa_pipeline_tuple if isinstance(qa_pipeline_tuple, tuple) else (None, "none")

        # ── Try QA model first ──────────────────────────────────────────────
        if qa_model is not None:
            sentences = split_sentences(abstract)
            best = {"score": 0.0, "answer": ""}
            try:
                result = qa_model(question=question, context=abstract[:max_context])
                if result["score"] > best["score"]:
                    best = {"score": result["score"], "answer": result["answer"]}
            except Exception:
                pass
            for i, _ in enumerate(sentences):
                window = " ".join(sentences[max(0, i - 1): i + 2])
                try:
                    result = qa_model(question=question, context=window)
                    if result["score"] > best["score"]:
                        best = {"score": result["score"], "answer": result["answer"]}
                except Exception:
                    continue
            if best["score"] >= min_score and best["answer"].strip():
                return f"{best['answer']}  [QA score: {best['score']:.3f}]"
            # QA model ran but confidence too low — fall through to keyword extractor

        # ── Keyword fallback (always runs if QA model fails or low confidence) ──
        return extract_evidence_keywords(abstract, genes, disease)

    # ── Main search logic ─────────────────────────────────────────────────────
    if run_lit:
        if not gene_choice:
            st.warning("Select at least one gene.")
        else:
            # Build PubMed query — genes always included; disease is optional
            gene_str = " OR ".join(f'"{g}"[Text Word]' for g in gene_choice)
            if disease:
                query = f'({gene_str}) AND ("{disease}"[Title/Abstract])'
            else:
                # Broad search: all EIF2AK genes across all disease/biological contexts
                query = (
                    f'({gene_str}) AND '
                    f'(disease OR cancer OR disorder OR syndrome OR infection OR '
                    f'diabetes OR neurodegeneration OR inflammation OR '
                    f'tumor OR stress OR pathway)[Title/Abstract]'
                )

            # Build QA question — used by BioM-ELECTRA when model is available
            genes_str_q = " or ".join(gene_choice)
            question = (
                f"What is the role of {genes_str_q} in {disease}?"
                if disease else
                f"What disease or biological process is {genes_str_q} involved in?"
            )

            with st.spinner("📡 Querying PubMed…"):
                records = fetch_pubmed(query, max_results)

            if not records:
                st.error("No results found. Try broader terms or enable Relaxed filter.")
            else:
                with st.spinner("⚙️ Loading evidence model (QA + keyword fallback)…"):
                    qa_model_tuple = load_qa_pipeline()

                model_label = "BioM-ELECTRA" if (
                    isinstance(qa_model_tuple, tuple) and qa_model_tuple[1] == "bioelectra"
                ) else ("DistilBERT" if (
                    isinstance(qa_model_tuple, tuple) and qa_model_tuple[1] == "distilbert"
                ) else "Keyword")

                st.markdown(f"""
                <div class="metric-row">
                  <div class="metric-box">
                    <div class="metric-val">{len(records)}</div>
                    <div class="metric-lbl">Abstracts Found</div>
                  </div>
                  <div class="metric-box">
                    <div class="metric-val">{len(gene_choice)}</div>
                    <div class="metric-lbl">Genes Queried</div>
                  </div>
                  <div class="metric-box">
                    <div class="metric-val">{model_label}</div>
                    <div class="metric-lbl">Evidence Model</div>
                  </div>
                  <div class="metric-box">
                    <div class="metric-val">PubMed</div>
                    <div class="metric-lbl">Source DB</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                rows = []
                prog = st.progress(0, text="Extracting evidence…")
                for i, rec in enumerate(records):
                    title    = rec.get("TI", "N/A")
                    authors  = "; ".join(rec.get("AU", [])[:3])
                    if len(rec.get("AU", [])) > 3:
                        authors += " et al."
                    date     = rec.get("DP", "N/A")
                    abstract = rec.get("AB", "")
                    pmid     = rec.get("PMID", "")
                    # Case-insensitive gene detection in abstract
                    abs_lower = abstract.lower()
                    genes_in  = [g for g in EIF2AK_GENES if g.lower() in abs_lower]
                    # Always extract evidence — keyword fallback ensures never empty
                    evidence = find_evidence_qa(
                        qa_model_tuple, abstract, question,
                        genes=gene_choice, disease=disease
                    )
                    rows.append({
                        "PMID":        pmid,
                        "Title":       title,
                        "Authors":     authors,
                        "Date":        date,
                        "Genes Mentioned": ", ".join(genes_in) if genes_in else "—",
                        "Abstract":    (abstract[:350] + "…") if len(abstract) > 350 else abstract,
                        "AI Evidence": evidence,
                    })
                    prog.progress((i+1)/len(records),
                                  text=f"Processing {i+1}/{len(records)}…")
                    time.sleep(0.05)

                prog.empty()
                df_lit = pd.DataFrame(rows)
                st.session_state["lit_df"] = df_lit

                st.markdown('<div class="section-header">📋 Evidence Table</div>',
                            unsafe_allow_html=True)
                filt = st.text_input("🔎 Filter table", placeholder="gene, disease, keyword…")
                df_show = df_lit[
                    df_lit.apply(lambda r: filt.lower() in r.to_string().lower(), axis=1)
                ] if filt else df_lit

                st.dataframe(df_show, use_container_width=True, height=400)

                buf = io.StringIO()
                df_lit.to_csv(buf, index=False)
                st.download_button("⬇ Download CSV", buf.getvalue(),
                                   file_name="eif2ak_literature.csv",
                                   mime="text/csv")

                st.markdown('<div class="section-header">🔬 Evidence Detail View</div>',
                            unsafe_allow_html=True)
                for _, row in df_show.head(5).iterrows():
                    with st.expander(f"📄 {row['Title'][:90]}…"):
                        st.markdown(
                            f"**Authors:** {row['Authors']}  •  **Date:** {row['Date']}  "
                            f"•  **Genes:** {row['Genes Mentioned']}"
                        )
                        st.markdown(
                            f"**PMID:** [{row['PMID']}]"
                            f"(https://pubmed.ncbi.nlm.nih.gov/{row['PMID']}/)"
                        )
                        st.markdown("**Abstract:**")
                        st.markdown(row["Abstract"])
                        st.markdown(f"""
                        <div class="evidence-badge">
                          🤖 <b>AI Evidence ({model_label}):</b><br>
                          {row['AI Evidence']}
                        </div>
                        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: GEO AI-AGENT & EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 GEO AI-Agent & Explorer":

    st.markdown("""
    <div class="hero-banner">
      <div class="hero-title">GEO AI-Agent &amp; Explorer</div>
      <div class="hero-subtitle">
        Search NCBI GEO · RNA-seq datasets · AI-generated preprocessing scripts ·
        Cleaning sandbox for DESeq2
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 2-A  GEO SEARCH ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔍 Search GEO Datasets</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        geo_disease = st.text_input("🦠 Disease / Condition",
                                    placeholder="e.g., diabetes")
    with c2:
        geo_extra = st.text_input("🔑 Additional keywords",
                                  placeholder="e.g., EIF2AK3 ER stress")

    inc_eif = st.checkbox("Auto-include EIF2AK in query", value=True)
    relaxed = st.checkbox(
        "Relaxed filter (include microarray / other organisms if RNA-seq scarce)",
        value=False
    )
    run_geo = st.button("🔍 Search GEO Datasets")

    @st.cache_data(show_spinner=False)
    def search_geo(query: str, max_r: int = 100, relaxed: bool = False):
        import socket
        from Bio import Entrez as _Entrez
        _Entrez.email   = "samikshapasalkar2212@gmail.com"
        _Entrez.api_key = "9de22485baf54ae653d2825299784fcfb008"
        try:
            socket.setdefaulttimeout(5)
            socket.getaddrinfo("eutils.ncbi.nlm.nih.gov", 443)
        except OSError:
            st.error(
                "❌ **Cannot reach NCBI servers.** Check your internet connection or VPN."
            )
            return pd.DataFrame(), 0
        try:
            handle  = _Entrez.esearch(db="gds", term=query, retmax=max_r, sort="relevance")
            record  = _Entrez.read(handle); handle.close()
            ids     = record.get("IdList", [])
            total   = int(record.get("Count", 0))
            if not ids:
                return pd.DataFrame(), total

            handle2    = _Entrez.esummary(db="gds", id=",".join(ids))
            summaries  = _Entrez.read(handle2); handle2.close()

            rows = []
            for s in summaries:
                gds_type  = s.get("gdsType", "")
                organism  = s.get("taxon", "")
                is_rnaseq = "high throughput sequencing" in gds_type.lower()
                is_human  = "homo sapiens" in organism.lower()
                if not relaxed:
                    if not (is_rnaseq and is_human):
                        continue
                else:
                    if not (is_human or is_rnaseq):
                        continue
                acc = s.get("Accession", "")
                rows.append({
                    "Accession": acc,
                    "Title":     s.get("title", ""),
                    "Organism":  organism,
                    "Type":      gds_type,
                    "Samples":   s.get("n_samples", ""),
                    "Date":      s.get("PDAT", ""),
                    "GEO Link":  f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc}",
                    "Platform":  str(s.get("GPL", "")),
                })
            return pd.DataFrame(rows), total
        except Exception as e:
            st.error(f"GEO search error: {e}")
            return pd.DataFrame(), 0

    if run_geo:
        parts = []
        if geo_disease: parts.append(f'"{geo_disease}"')
        if geo_extra:   parts.append(geo_extra)
        if inc_eif:     parts.append("EIF2AK")
        full_q = " AND ".join(parts) if parts else "EIF2AK"
        st.info(f"🔎 Query: `{full_q}`")
        with st.spinner("Searching GEO…"):
            geo_df, total = search_geo(full_q, max_r=100, relaxed=relaxed)
        st.session_state["geo_df"] = geo_df
        st.session_state["geo_total"] = total
        if geo_df.empty:
            st.warning(
                f"No RNA-seq / Homo sapiens datasets after filtering "
                f"({total} raw hits). Try enabling Relaxed filter or simplify terms."
            )
        else:
            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-box"><div class="metric-val">{len(geo_df)}</div>
                <div class="metric-lbl">Filtered Datasets</div></div>
              <div class="metric-box"><div class="metric-val">{total}</div>
                <div class="metric-lbl">Raw GEO Hits</div></div>
              <div class="metric-box">
                <div class="metric-val">{'RNA-seq' if not relaxed else 'Relaxed'}</div>
                <div class="metric-lbl">Filter Mode</div></div>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(
                geo_df[["Accession","Title","Organism","Type","Samples","Date","GEO Link"]],
                use_container_width=True
            )
    elif "geo_df" in st.session_state and not st.session_state["geo_df"].empty:
        st.info("Showing cached results — press Search to refresh.")
        geo_df = st.session_state["geo_df"]
        st.dataframe(
            geo_df[["Accession","Title","Organism","Type","Samples","Date","GEO Link"]],
            use_container_width=True
        )

    # ── 2-B  AI SCRIPTING AGENT ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🤖 AI Scripting Agent (Geo2R Translator)</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <div class="card-title">How It Works</div>
      Enter a GSE or GDS accession. The agent will:<br>
      <ol style="margin:0.6rem 0 0 1rem; color:#9ca3af; font-size:0.87rem; line-height:1.8;">
        <li>Fetch dataset metadata from GEO via <code>GEOparse</code></li>
        <li>Analyse the supplementary file structure to detect columns &amp; sample groups</li>
        <li>Generate a virtual Geo2R-style <b>R script</b> (for reference)</li>
        <li>Automatically translate that logic into a <b>Pandas / PyDESeq2 Python script</b></li>
      </ol>
    </div>
    """, unsafe_allow_html=True)

    acc_input = st.text_input("GEO Accession (e.g., GSE12345 or GDS1234)",
                               placeholder="GSE…")
    fetch_meta = st.button("🧠 Fetch Metadata & Generate Scripts")

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
            "supp_files": getattr(gse, "metadata", {}).get("supplementary_file", []),
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

    def build_r_script(info: dict, groups: dict) -> str:
        """Generate a Geo2R-style R script for reference."""
        ctrl  = [s for s,g in groups.items() if g == "control"][:3]
        dis   = [s for s,g in groups.items() if g == "disease"][:3]
        lines = [
            f'# Geo2R-style script — auto-generated by EIF2AK Discovery Engine',
            f'# Dataset: {info["title"][:60]}',
            f'# Organism: {info["organism"]}',
            f'# Platform: {info["platform"]}',
            '',
            'library(GEOquery)',
            'library(DESeq2)',
            '',
            f'gse <- getGEO("{info.get("accession","GSE_XXXX")}", GSEMatrix=TRUE)',
            'gse <- gse[[1]]',
            '',
            '# Build count matrix',
            'counts <- exprs(gse)',
            'counts <- round(counts)   # Ensure integer counts',
            '',
            '# Assign sample groups',
            f'ctrl_samples  <- c({", ".join(repr(s) for s in ctrl)}{"..." if len(ctrl)<len([s for s,g in groups.items() if g=="control"]) else ""})',
            f'dis_samples   <- c({", ".join(repr(s) for s in dis )}{"..." if len(dis )<len([s for s,g in groups.items() if g=="disease"]) else ""})',
            '',
            'col_data <- data.frame(',
            '  row.names  = c(ctrl_samples, dis_samples),',
            '  condition  = factor(c(rep("control", length(ctrl_samples)),',
            '                        rep("disease", length(dis_samples))))',
            ')',
            '',
            '# Run DESeq2',
            'dds <- DESeqDataSetFromMatrix(',
            '  countData = counts[, rownames(col_data)],',
            '  colData   = col_data,',
            '  design    = ~ condition',
            ')',
            'dds <- DESeq(dds)',
            'res <- results(dds, contrast=c("condition","disease","control"))',
            'res <- res[order(res$padj), ]',
            'write.csv(as.data.frame(res), "deseq2_results.csv")',
        ]
        return "\n".join(lines)

    def build_python_script(info: dict, groups: dict) -> str:
        """Translate R/Geo2R logic into a Pandas + PyDESeq2 Python script."""
        ctrl = [s for s,g in groups.items() if g=="control"][:3]
        dis  = [s for s,g in groups.items() if g=="disease"][:3]
        lines = [
            '# PyDESeq2 preprocessing script — auto-generated by EIF2AK Discovery Engine',
            f'# Dataset: {info["title"][:60]}',
            f'# Organism: {info["organism"]}',
            '',
            'import pandas as pd',
            'import numpy as np',
            'from pydeseq2.dds import DeseqDataSet',
            'from pydeseq2.ds  import DeseqStats',
            '',
            '# ── 1. Load count matrix ──────────────────────────────────────',
            '# Replace with your actual file path (downloaded from GEO supp files)',
            'counts_raw = pd.read_csv("counts.csv", index_col=0)',
            '',
            '# ── 2. Integer conversion (required by DESeq2) ───────────────',
            'counts_raw = counts_raw.apply(pd.to_numeric, errors="coerce").fillna(0)',
            'counts_raw = counts_raw.round(0).astype(int)',
            '',
            '# ── 3. Remove zero-count genes ───────────────────────────────',
            'counts_raw = counts_raw[counts_raw.sum(axis=1) > 0]',
            '',
            '# ── 4. Assign sample groups ──────────────────────────────────',
            f'ctrl_samples = {ctrl + ["..."]}  # extend with full list',
            f'dis_samples  = {dis  + ["..."]}  # extend with full list',
            'keep = ctrl_samples + dis_samples',
            '',
            'count_matrix = counts_raw[keep].T  # samples x genes',
            'metadata = pd.DataFrame({',
            '    "condition": ["control"]*len(ctrl_samples) + ["disease"]*len(dis_samples)',
            '}, index=keep)',
            '',
            '# ── 5. Run PyDESeq2 ──────────────────────────────────────────',
            'dds = DeseqDataSet(',
            '    counts=count_matrix,',
            '    metadata=metadata,',
            '    design_factors="condition",',
            '    ref_level=["condition", "control"],',
            ')',
            'dds.deseq2()',
            '',
            'stat_res = DeseqStats(dds, contrast=["condition","disease","control"])',
            'stat_res.summary()',
            'results = stat_res.results_df',
            '',
            '# ── 6. EIF2AK Spotlight ──────────────────────────────────────',
            'eif2ak = ["EIF2AK1","EIF2AK2","EIF2AK3","EIF2AK4"]',
            'spotlight = results[results.index.isin(eif2ak)]',
            'print(spotlight)',
            '',
            'results.to_csv("deseq2_results.csv")',
        ]
        return "\n".join(lines)

    def syntax_highlight(code: str) -> str:
        """Very lightweight HTML syntax colouring for the code block."""
        import html as hlib
        lines = []
        kws = {"import","from","as","def","class","return","for","in","if","else",
               "elif","with","try","except","True","False","None","and","or","not"}
        for raw in code.split("\n"):
            safe = hlib.escape(raw)
            # comments
            if re.match(r'^\s*#', raw):
                lines.append(f'<span class="cmt">{safe}</span>')
                continue
            # strings
            safe = re.sub(r'(&quot;[^&]*&quot;|&#x27;[^&]*&#x27;)',
                          r'<span class="str">\1</span>', safe)
            # numbers
            safe = re.sub(r'\b(\d+\.?\d*)\b', r'<span class="num">\1</span>', safe)
            # keywords
            for kw in kws:
                safe = re.sub(rf'\b({kw})\b',
                              r'<span class="kw">\1</span>', safe)
            lines.append(safe)
        return "<br>".join(lines)

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
                if info["supp_files"]:
                    st.markdown("**Supplementary Files:**")
                    for sf in info["supp_files"]:
                        st.markdown(f"- `{sf}`")

            with st.expander("🧬 Sample Groups (AI-guessed)"):
                sample_data = [
                    {"Sample ID": sid, "Title": m["title"],
                     "Detected Group": groups.get(sid,"unknown"),
                     "Characteristics": "; ".join(m["chars"][:3])}
                    for sid, m in info["samples"].items()
                ]
                st.dataframe(pd.DataFrame(sample_data), use_container_width=True)

            r_script  = build_r_script(info, groups)
            py_script = build_python_script(info, groups)

            st.markdown('<div class="section-header">📜 Generated Geo2R R Script (Reference)</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="code-block">{syntax_highlight(r_script)}</div>',
                        unsafe_allow_html=True)
            st.download_button("⬇ Download R Script", r_script,
                               file_name=f"{acc}_geo2r.R", mime="text/plain")

            st.markdown('<div class="section-header">🐍 Auto-translated Python / PyDESeq2 Script</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="code-block">{syntax_highlight(py_script)}</div>',
                        unsafe_allow_html=True)
            st.download_button("⬇ Download Python Script", py_script,
                               file_name=f"{acc}_pydeseq2.py", mime="text/plain")

            st.session_state["geo_meta"]   = info
            st.session_state["geo_groups"] = groups

    # ── 2-C  CLEANING SANDBOX ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🧹 Cleaning Sandbox</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <div class="card-title">Upload Count Matrix</div>
      Rows = genes, Columns = samples. Accepts CSV, TSV, TXT, Excel, and other tabular formats.
      First column / row index will be used as gene names.
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload expression matrix",
                                type=None)

    if uploaded:
        fname = uploaded.name.lower()
        if fname.endswith((".tsv", ".txt")):
            sep = "\t"
        else:
            sep = ","  # default CSV for .csv and all other types
        try:
            raw_df = pd.read_csv(uploaded, sep=sep, index_col=0)
        except Exception:
            try:
                uploaded.seek(0)
                raw_df = pd.read_csv(uploaded, sep="\t", index_col=0)
            except Exception as e:
                st.error(f"Could not read file: {e}")
                st.stop()

        st.markdown(f"**Raw shape:** {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")
        st.dataframe(raw_df.head(5), use_container_width=True)

        # Gene names taken from row index by default (Gene Symbol Column removed)
        gene_col = "(use row index as gene names)"

        st.markdown("#### 2️⃣ Sample Columns")
        all_cols = [c for c in raw_df.columns if c != gene_col] \
                   if gene_col != "(use row index as gene names)" \
                   else list(raw_df.columns)
        sample_cols = st.multiselect("Sample columns", all_cols,
                                     default=all_cols[:min(len(all_cols), 12)])

        st.markdown("#### 3️⃣ Cleaning Options")
        force_int  = st.toggle("Force Integer Conversion (required for DESeq2)", value=True)
        drop_zero  = st.toggle("Drop genes with zero total counts", value=True)
        drop_lowex = st.toggle("Drop genes with < 10 total counts", value=False)

        if st.button("✅ Apply Cleaning & Save to Session"):
            if not sample_cols:
                st.error("Select at least one sample column.")
                st.stop()
            if gene_col != "(use row index as gene names)":
                df_clean = raw_df[[gene_col] + sample_cols].set_index(gene_col)
            else:
                df_clean = raw_df[sample_cols].copy()

            if force_int:
                df_clean = df_clean.apply(pd.to_numeric, errors="coerce").fillna(0)
                df_clean = df_clean.round(0).astype(np.int64)
            if drop_zero:
                df_clean = df_clean[df_clean.sum(axis=1) > 0]
            if drop_lowex:
                df_clean = df_clean[df_clean.sum(axis=1) >= 10]

            st.session_state["clean_counts"]      = df_clean
            st.session_state["counts_transposed"] = df_clean.T
            st.session_state["sample_names"]      = list(df_clean.columns)

            st.success(
                f"✅ Saved!  {df_clean.shape[0]} genes × {df_clean.shape[1]} samples  "
                f"→ navigate to 📊 Differential Expression."
            )
            st.dataframe(df_clean.head(8), use_container_width=True)

            buf = io.StringIO(); df_clean.to_csv(buf)
            st.download_button("⬇ Download Cleaned Matrix (CSV)", buf.getvalue(),
                               file_name="cleaned_counts.csv", mime="text/csv",
                               key="dl_clean")
            buf2 = io.StringIO(); df_clean.T.to_csv(buf2)
            st.download_button("⬇ Download Transposed Matrix (CSV)", buf2.getvalue(),
                               file_name="transposed_counts.csv", mime="text/csv",
                               key="dl_trans")

    elif "clean_counts" in st.session_state:
        dc = st.session_state["clean_counts"]
        st.success(f"✅ Already loaded: {dc.shape[0]} genes × {dc.shape[1]} samples.")
        st.dataframe(dc.head(5), use_container_width=True)
        buf = io.StringIO(); dc.to_csv(buf)
        st.download_button("⬇ Re-download Cleaned Matrix", buf.getvalue(),
                           file_name="cleaned_counts.csv", mime="text/csv",
                           key="redown_clean")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: DIFFERENTIAL EXPRESSION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Differential Expression":

    st.markdown("""
    <div class="hero-banner">
      <div class="hero-title">Differential Expression Analysis</div>
      <div class="hero-subtitle">
        Upload cleaned CSV → label samples → run PyDESeq2 →
        mygene ID mapping → EIF2AK spotlight
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 3-A  Load data: session OR fresh CSV upload ───────────────────────────
    st.markdown("""
    <div class="step-indicator">
      <div class="step-num">0</div>
      <div class="step-label">Load cleaned count matrix</div>
    </div>
    """, unsafe_allow_html=True)

    if "clean_counts" in st.session_state:
        counts_df   = st.session_state["clean_counts"]
        sample_names = list(counts_df.columns)
        st.success(
            f"✅ Using session data: {counts_df.shape[0]} genes × "
            f"{counts_df.shape[1]} samples"
        )
    else:
        st.info(
            "No data in session. You can either:\n"
            "1. Go to **🔬 GEO AI-Agent & Explorer** → upload & clean data there, OR\n"
            "2. Upload your cleaned CSV directly below."
        )
        up_csv = st.file_uploader(
            "Upload cleaned count matrix (genes × samples, CSV)",
            type=["csv"],
            key="deseq_upload"
        )
        if up_csv:
            try:
                counts_df    = pd.read_csv(up_csv, index_col=0)
                counts_df    = counts_df.apply(pd.to_numeric, errors="coerce").fillna(0)
                counts_df    = counts_df.round(0).astype(np.int64)
                sample_names = list(counts_df.columns)
                st.session_state["clean_counts"]  = counts_df
                st.session_state["sample_names"]  = sample_names
                st.success(
                    f"✅ Loaded {counts_df.shape[0]} genes × {counts_df.shape[1]} samples"
                )
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                st.stop()
        else:
            st.stop()

    # ── 3-B  Sample grouping ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div class="step-indicator">
      <div class="step-num">1</div>
      <div class="step-label">Label each sample as Control, Disease, or Exclude</div>
    </div>
    """, unsafe_allow_html=True)

    # Auto-guess groups from sample names
    def auto_guess_label(name: str) -> str:
        n = name.lower()
        ctrl_kw = ["ctrl","control","normal","healthy","wt","vehicle","untreated","mock"]
        dis_kw  = ["disease","treated","tumor","cancer","patient","kd","ko","mut","stim"]
        if any(k in n for k in ctrl_kw):
            return "Control"
        if any(k in n for k in dis_kw):
            return "Disease"
        return "Control"

    if ("group_df" not in st.session_state or
            set(st.session_state["group_df"]["Sample"].tolist()) != set(sample_names)):
        st.session_state["group_df"] = pd.DataFrame({
            "Sample": sample_names,
            "Group":  [auto_guess_label(s) for s in sample_names]
        })

    edited_groups = st.data_editor(
        st.session_state["group_df"],
        column_config={
            "Sample": st.column_config.TextColumn("Sample", disabled=True),
            "Group":  st.column_config.SelectboxColumn(
                "Group", options=["Control","Disease","Exclude"], required=True
            )
        },
        use_container_width=True,
        num_rows="fixed",
        key="group_editor"
    )
    st.session_state["group_df"] = edited_groups

    valid        = edited_groups[edited_groups["Group"] != "Exclude"]
    ctrl_samples = valid[valid["Group"] == "Control"]["Sample"].tolist()
    dis_samples  = valid[valid["Group"] == "Disease"]["Sample"].tolist()

    c1, c2, c3 = st.columns(3)
    c1.metric("Control Samples", len(ctrl_samples))
    c2.metric("Disease Samples", len(dis_samples))
    c3.metric("Excluded",        len(sample_names) - len(ctrl_samples) - len(dis_samples))

    # ── 3-C  Gene ID Mapping ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div class="step-indicator">
      <div class="step-num">2</div>
      <div class="step-label">Gene ID detection &amp; mapping (Ensembl / Entrez → Symbol)</div>
    </div>
    """, unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def map_gene_ids(gene_tuple: tuple):
        """Map Ensembl/Entrez IDs to symbols via mygene."""
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
                return None  # Already symbols
            results = mg.querymany(
                list(gene_tuple), scopes=scope,
                fields="symbol", species="human", returnall=False
            )
            return {r["query"]: r["symbol"] for r in results if "symbol" in r}
        except Exception as e:
            st.warning(f"mygene mapping: {e}")
            return None

    map_toggle   = st.toggle("Auto-detect & map gene IDs to symbols", value=True)
    gene_mapping = None
    if map_toggle:
        with st.spinner("Mapping gene IDs…"):
            gene_mapping = map_gene_ids(tuple(counts_df.index.tolist()))
        if gene_mapping:
            st.success(f"✅ Mapped {len(gene_mapping)} IDs to gene symbols.")
        else:
            st.info("Gene IDs appear to be symbols already (or mapping skipped).")

    # ── 3-D  Run DESeq2 ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div class="step-indicator">
      <div class="step-num">3</div>
      <div class="step-label">Run PyDESeq2 statistical analysis</div>
    </div>
    """, unsafe_allow_html=True)

    run_deseq = st.button("🚀 Run Differential Expression Analysis")

    if run_deseq:
        if len(ctrl_samples) < 2 or len(dis_samples) < 2:
            st.error("Need ≥ 2 Control and ≥ 2 Disease samples to run DESeq2.")
            st.stop()

        try:
            from pydeseq2.dds import DeseqDataSet
            from pydeseq2.ds  import DeseqStats
        except ImportError:
            st.error("PyDESeq2 is not installed. Add `pydeseq2` to requirements.txt.")
            st.stop()

        keep_cols    = ctrl_samples + dis_samples
        count_matrix = counts_df[keep_cols].T.copy()   # samples × genes
        metadata     = pd.DataFrame({
            "condition": (
                ["control"] * len(ctrl_samples) +
                ["disease"] * len(dis_samples)
            )
        }, index=keep_cols)

        if gene_mapping:
            count_matrix.columns = [
                gene_mapping.get(g, g) for g in count_matrix.columns
            ]
        count_matrix = count_matrix.loc[:, ~count_matrix.columns.duplicated()]

        with st.spinner("🔬 Running PyDESeq2… (may take 1–3 minutes)"):
            try:
                dds = DeseqDataSet(
                    counts=count_matrix,
                    metadata=metadata,
                    design_factors="condition",
                    ref_level=["condition", "control"],
                    refit_cooks=True,
                    inference=None,
                )
                dds.deseq2()
                stat_res = DeseqStats(
                    dds, contrast=["condition", "disease", "control"]
                )
                stat_res.summary()
                res_df = stat_res.results_df.copy()
                res_df.index.name = "Gene"
                res_df = res_df.reset_index()
                st.session_state["deseq_results"] = res_df
                st.success("✅ Analysis complete!")
            except Exception as e:
                st.error(f"PyDESeq2 error: {e}")
                st.stop()

    # ── 3-E  Display results ───────────────────────────────────────────────────
    if "deseq_results" in st.session_state:
        res_df = st.session_state["deseq_results"]

        # ── Full Results ──────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📋 Full Results Table</div>',
                    unsafe_allow_html=True)
        sorted_res = res_df.sort_values("padj", na_position="last")
        sig_n = int((sorted_res["padj"] < 0.05).sum()) if "padj" in sorted_res else 0
        up_n  = int(
            ((sorted_res["padj"] < 0.05) & (sorted_res.get("log2FoldChange", pd.Series([0])) > 0)).sum()
        ) if "padj" in sorted_res and "log2FoldChange" in sorted_res else 0
        dn_n  = sig_n - up_n

        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-box"><div class="metric-val">{len(res_df)}</div>
            <div class="metric-lbl">Total Genes</div></div>
          <div class="metric-box"><div class="metric-val">{sig_n}</div>
            <div class="metric-lbl">Significant (padj&lt;0.05)</div></div>
          <div class="metric-box"><div class="metric-val">{up_n}</div>
            <div class="metric-lbl">Up-regulated</div></div>
          <div class="metric-box"><div class="metric-val">{dn_n}</div>
            <div class="metric-lbl">Down-regulated</div></div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            lfc_t  = st.slider("log₂ FC threshold", 0.0, 3.0, 0.0, 0.25)
        with col2:
            padj_t = st.slider("Adjusted p-value threshold", 0.001, 0.5, 0.05, 0.001)

        filtered = sorted_res.copy()
        if "log2FoldChange" in filtered and lfc_t > 0:
            filtered = filtered[filtered["log2FoldChange"].abs() >= lfc_t]
        if "padj" in filtered:
            filtered = filtered[filtered["padj"] <= padj_t]

        st.dataframe(filtered, use_container_width=True, height=380)

        buf1 = io.StringIO(); res_df.to_csv(buf1, index=False)
        st.download_button("⬇ Download Full Results CSV", buf1.getvalue(),
                           file_name="eif2ak_deseq2_full.csv", mime="text/csv",
                           key="dl_full")

        # ── EIF2AK Spotlight ───────────────────────────────────────────────────
        st.markdown('<div class="section-header">🌟 EIF2AK Gene Spotlight</div>',
                    unsafe_allow_html=True)

        spotlight = res_df[res_df["Gene"].isin(EIF2AK_GENES)].copy()
        if spotlight.empty:
            spotlight = res_df[
                res_df["Gene"].str.contains("EIF2AK", na=False)
            ].copy()

        if spotlight.empty:
            st.info(
                "No EIF2AK genes found. Ensure gene ID mapping was applied "
                "and that the dataset contains EIF2AK family genes."
            )
        else:
            st.markdown("""
            <div class="card">
              <div class="card-title">🟢 Green rows = padj &lt; 0.05 (statistically significant)</div>
            </div>
            """, unsafe_allow_html=True)

            def highlight_sig(row):
                try:
                    if float(row.get("padj", 1)) < 0.05:
                        return (["background-color: rgba(16,185,129,0.18); "
                                 "color: #10b981; font-weight:600"] * len(row))
                except (TypeError, ValueError):
                    pass
                return [""] * len(row)

            styled = spotlight.style.apply(highlight_sig, axis=1)
            st.dataframe(styled, use_container_width=True)

            buf2 = io.StringIO(); spotlight.to_csv(buf2, index=False)
            st.download_button("⬇ Download EIF2AK Spotlight CSV", buf2.getvalue(),
                               file_name="eif2ak_spotlight.csv", mime="text/csv",
                               key="dl_spot")

            # ── Per-gene mini-cards ────────────────────────────────────────────
            st.markdown('<div class="section-header">📊 Gene-level Summary</div>',
                        unsafe_allow_html=True)
            gene_cols = st.columns(min(len(spotlight), 4))
            for i, (_, row) in enumerate(spotlight.iterrows()):
                lfc  = row.get("log2FoldChange", "N/A")
                padj = row.get("padj", "N/A")
                try:
                    sig_label = "🟢 Significant" if float(padj) < 0.05 else "⚪ Not Sig."
                    padj_str  = f"{float(padj):.2e}"
                except (TypeError, ValueError):
                    sig_label = "❓ Unknown"
                    padj_str  = str(padj)
                try:
                    lfc_val   = float(lfc)
                    lfc_str   = f"{lfc_val:+.3f}"
                    lfc_color = "#10b981" if lfc_val > 0 else "#ef4444"
                except (TypeError, ValueError):
                    lfc_str   = str(lfc)
                    lfc_color = "#00ffcc"

                with gene_cols[i % 4]:
                    st.markdown(f"""
                    <div class="gene-spotlight">
                      <div class="gene-name">{row['Gene']}</div>
                      <div class="gene-lfc" style="color:{lfc_color};">{lfc_str}</div>
                      <div style="font-size:0.7rem; color:#5a6a80; margin:0.15rem 0 0.5rem 0;">
                        log₂ Fold Change
                      </div>
                      <div style="font-size:0.78rem; color:#94a3b8;">padj: {padj_str}</div>
                      <div style="margin-top:0.35rem; font-size:0.8rem;">{sig_label}</div>
                    </div>
                    """, unsafe_allow_html=True)
