"""
app.py  -  Tweet Sentiment Analysis Dashboard
----------------------------------------------
ML-powered Streamlit dashboard.
Uses TF-IDF + Naive Bayes / Logistic Regression trained on Sentiment140.
No TextBlob. Pure scikit-learn NLP pipeline.

Tabs:
  1. Overview      - KPI cards, sentiment pie, trend, hourly chart
  2. Words         - Word cloud, top keywords
  3. Users         - Top users, sentiment & confidence per user
  4. Search        - Keyword search with sentiment breakdown
  5. Classifier    - Real-time ML sentiment classifier (NB vs LR)
  6. Model Report  - Accuracy comparison, confusion matrix, metrics
  7. Viral Tweets  - Most confident positive/negative tweets

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re, json
from pathlib import Path
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tweet Sentiment Analysis",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark Premium Theme ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0d1117; color: #e6edf3; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#161b22,#0d1117);
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

[data-testid="stMetricValue"] {
    font-size:1.9rem !important; font-weight:800 !important; color:#58a6ff !important;
}
[data-testid="stMetricLabel"] { color:#8b949e !important; font-size:.85rem !important; }
[data-testid="metric-container"] {
    background: linear-gradient(135deg,rgba(22,27,34,.95),rgba(13,17,23,.95));
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 1.1rem 1.3rem !important;
}

.stButton > button {
    background: linear-gradient(135deg,#1f6feb,#58a6ff);
    color:#fff; border:none; border-radius:9px;
    font-weight:700; padding:.55rem 1.6rem;
    transition:all .18s ease;
}
.stButton > button:hover {
    transform:translateY(-2px);
    box-shadow:0 8px 28px rgba(31,111,235,.5);
}

.stTextArea textarea, .stTextInput input {
    background:#161b22 !important; color:#e6edf3 !important;
    border:1px solid #30363d !important; border-radius:9px !important;
}
.stSelectbox div[data-baseweb="select"] {
    background:#161b22; border:1px solid #30363d; border-radius:9px;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8b949e;
    font-weight: 600;
    border-radius: 8px;
    padding: .4rem .9rem;
}
.stTabs [aria-selected="true"] {
    background: #21262d !important;
    color: #58a6ff !important;
}

hr { border-color: #21262d; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
PROCESSED_CSV = BASE_DIR / "processed_tweets.csv"
DEPLOYMENT_SAMPLE_CSV = BASE_DIR / "dashboard_sample.csv"
MODEL_DIR     = BASE_DIR / "model"
POS_COLOR     = "#2ea043"
NEG_COLOR     = "#f85149"
CHART_BG      = "rgba(0,0,0,0)"
GRIDCOLOR     = "#21262d"
TEXT_COLOR    = "#e6edf3"

PLOTLY_BASE = dict(
    paper_bgcolor=CHART_BG,
    plot_bgcolor=CHART_BG,
    font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
    margin=dict(l=20, r=20, t=45, b=20),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_COLOR)),
)

def styled(fig, **layout_kwargs):
    """Apply PLOTLY_BASE + default grid styles, then any extra layout kwargs."""
    fig.update_layout(**PLOTLY_BASE, **layout_kwargs)
    fig.update_xaxes(gridcolor=GRIDCOLOR, zerolinecolor=GRIDCOLOR)
    fig.update_yaxes(gridcolor=GRIDCOLOR, zerolinecolor=GRIDCOLOR)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# NLP PIPELINE  (mirrors preprocess.py — for real-time classifier)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_nlp():
    try:
        sw = set(stopwords.words("english")) - {"no", "nor", "not"}
    except LookupError:
        try:
            nltk.download("stopwords", quiet=True, raise_on_error=True)
            sw = set(stopwords.words("english")) - {"no", "nor", "not"}
        except Exception as exc:
            st.error(f"Could not install the required NLTK stopwords corpus: {exc}")
            st.stop()
    stem = PorterStemmer()
    return sw, stem

STOP_WORDS, STEMMER = load_nlp()

def clean_tweet(text: str) -> str:
    text = str(text).replace("’", "'")
    text = re.sub(r"\bwon't\b", "will not", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcan't\b", "can not", text, flags=re.IGNORECASE)
    text = re.sub(r"n't\b", " not", text, flags=re.IGNORECASE)
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = text.lower()
    text = re.sub(r"([a-z])\1{2,}", r"\1\1", text)
    tokens = [STEMMER.stem(t) for t in text.split()
              if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA + MODELS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading tweet data ...")
def load_data():
    data_path = PROCESSED_CSV if PROCESSED_CSV.exists() else DEPLOYMENT_SAMPLE_CSV
    if not data_path.exists():
        return None
    return pd.read_csv(data_path)

@st.cache_resource(show_spinner="Loading ML models ...")
def load_models():
    paths = {
        "vectorizer": MODEL_DIR / "tfidf_vectorizer.pkl",
        "nb":         MODEL_DIR / "nb_model.pkl",
        "lr":         MODEL_DIR / "lr_model.pkl",
        "comparison": MODEL_DIR / "comparison.json",
    }
    if not all(p.exists() for p in paths.values()):
        return None, None, None, None
    tfidf = joblib.load(paths["vectorizer"])
    nb    = joblib.load(paths["nb"])
    lr    = joblib.load(paths["lr"])
    with open(paths["comparison"]) as f:
        comp = json.load(f)
    return tfidf, nb, lr, comp

df = load_data()
tfidf, nb_model, lr_model, comp_data = load_models()

# ── Guard ─────────────────────────────────────────────────────────────────────
if df is None or tfidf is None:
    st.error(
        "**Models / data not found.**\n\n"
        "Please run the preprocessing script first:\n"
        "```\npython preprocess.py\n```"
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🐦 Tweet Sentiment")
    st.markdown("---")

    selected_sentiments = st.multiselect(
        "Sentiment Filter",
        options=["positive", "negative"],
        default=["positive", "negative"],
    )

    years = sorted(df["year"].dropna().unique().astype(int))
    if len(years) > 1:
        year_range = st.slider(
            "Year Range",
            min_value=int(years[0]),
            max_value=int(years[-1]),
            value=(int(years[0]), int(years[-1])),
        )
    else:
        year_range = (int(years[0]), int(years[0]))
        st.info(f"Year: **{years[0]}**")

    st.markdown("---")
    best_name = comp_data.get("best_model", "N/A") if comp_data else "N/A"
    best_acc  = comp_data.get(best_name, {}).get("accuracy", 0) if comp_data else 0
    st.markdown("### 🤖 Model Info")
    st.success(f"**Best Model:** {best_name}")
    st.info(f"**Test Accuracy:** {best_acc:.2f}%")
    if PROCESSED_CSV.exists():
        st.caption(f"Dashboard data: {len(df):,} tweets | Full Sentiment140 export")
    else:
        st.caption(
            f"Dashboard data: {len(df):,} stratified deployment sample | "
            "Models trained on all 1.6M tweets"
        )

# ─────────────────────────────────────────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────────────────────────────────────────
mask = (
    df["sentiment_label"].isin(selected_sentiments) &
    (df["year"] >= year_range[0]) &
    (df["year"] <= year_range[1])
)
filtered = df[mask].copy()

if filtered.empty:
    st.error("No data matches the current filters. Adjust the sidebar.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:1.5rem 0 .5rem'>
  <h1 style='font-size:2.7rem;font-weight:800;
     background:linear-gradient(135deg,#58a6ff 0%,#bc8cff 100%);
     -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.2rem'>
    🐦 Tweet Sentiment Analysis
  </h1>
  <p style='color:#8b949e;font-size:1rem'>
    TF-IDF · Naive Bayes · Logistic Regression · Sentiment140 Dataset
  </p>
</div><hr>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────
total     = len(filtered)
pos_count = (filtered["sentiment_label"] == "positive").sum()
neg_count = (filtered["sentiment_label"] == "negative").sum()
pos_pct   = pos_count / total * 100
avg_pos_probability = filtered["confidence"].mean() * 100

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("📊 Total Tweets",    f"{total:,}")
k2.metric("😊 Positive",        f"{pos_count:,}")
k3.metric("😠 Negative",        f"{neg_count:,}")
k4.metric("✅ Positive %",      f"{pos_pct:.1f}%")
k5.metric("🤖 Avg P(positive)", f"{avg_pos_probability:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Overview",
    "☁️ Words",
    "👤 Users",
    "🔍 Search",
    "💬 Classifier",
    "📊 Model Report",
    "🚀 Viral Tweets",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    col_a, col_b = st.columns([1, 2])

    with col_a:
        # Sentiment Donut
        st.subheader("😊 Sentiment Split")
        pie = filtered["sentiment_label"].value_counts().reset_index()
        pie.columns = ["Sentiment", "Count"]
        fig_pie = px.pie(
            pie, names="Sentiment", values="Count",
            color="Sentiment",
            color_discrete_map={"positive": POS_COLOR, "negative": NEG_COLOR},
            hole=0.55,
        )
        fig_pie.update_traces(
            textinfo="percent+label", textfont_size=13,
            marker=dict(line=dict(color="#0d1117", width=2)),
        )
        styled(fig_pie, showlegend=False, height=310)
        st.plotly_chart(fig_pie, width="stretch")

        # Confidence histogram
        st.subheader("📐 Positive-Probability Distribution")
        fig_conf = px.histogram(
            filtered, x="confidence", color="sentiment_label", nbins=50,
            color_discrete_map={"positive": POS_COLOR, "negative": NEG_COLOR},
            labels={"confidence": "P(positive)", "sentiment_label": "Sentiment"},
            opacity=0.8, barmode="overlay",
        )
        fig_conf.add_vline(x=0.5, line_dash="dot", line_color="#8b949e",
                           annotation_text="Decision boundary")
        styled(fig_conf, height=280)
        st.plotly_chart(fig_conf, width="stretch")

    with col_b:
        # Tweet trend over time
        st.subheader("📈 Tweet Trend Over Time")
        trend = (
            filtered.groupby(["year", "sentiment_label"])
            .size().reset_index(name="count")
        )
        fig_trend = px.line(
            trend, x="year", y="count", color="sentiment_label",
            markers=True,
            color_discrete_map={"positive": POS_COLOR, "negative": NEG_COLOR},
            labels={"count": "Tweets", "year": "Year", "sentiment_label": "Sentiment"},
        )
        fig_trend.update_traces(line=dict(width=3))
        styled(fig_trend, height=310)
        st.plotly_chart(fig_trend, width="stretch")

        # Hourly activity
        st.subheader("🕐 Hourly Tweet Activity")
        hourly = (
            filtered.groupby(["hour", "sentiment_label"])
            .size().reset_index(name="count")
        )
        fig_hour = px.bar(
            hourly, x="hour", y="count", color="sentiment_label",
            barmode="group",
            color_discrete_map={"positive": POS_COLOR, "negative": NEG_COLOR},
            labels={"hour": "Hour (UTC)", "count": "Tweets"},
        )
        styled(fig_hour, height=280)
        st.plotly_chart(fig_hour, width="stretch")

    # Monthly volume
    st.subheader("📅 Monthly Tweet Volume")
    monthly = (
        filtered.groupby(["year", "month", "sentiment_label"])
        .size().reset_index(name="count")
    )
    monthly["period"] = (
        monthly["year"].astype(str) + "-" +
        monthly["month"].astype(str).str.zfill(2)
    )
    monthly = monthly.sort_values("period")
    fig_month = px.bar(
        monthly, x="period", y="count", color="sentiment_label",
        barmode="stack",
        color_discrete_map={"positive": POS_COLOR, "negative": NEG_COLOR},
        labels={"period": "Month", "count": "Tweets"},
    )
    styled(fig_month, height=300, xaxis_tickangle=-45)
    st.plotly_chart(fig_month, width="stretch")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — WORDS
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    col_wc, col_kw = st.columns([1, 1])

    with col_wc:
        st.subheader("☁️ Word Cloud")
        wc_sent = st.selectbox("Show words for", ["positive", "negative", "both"], key="wc")

        @st.cache_data(show_spinner="Generating word cloud ...")
        def make_wc(sent, y0, y1):
            sub = df[(df["year"] >= y0) & (df["year"] <= y1)]
            if sent != "both":
                sub = sub[sub["sentiment_label"] == sent]
            text = " ".join(sub["clean_text"].dropna().astype(str))
            if not text.strip():
                return None
            cmap = "YlGn" if sent == "positive" else ("Reds" if sent == "negative" else "cool")
            wc = WordCloud(
                width=800, height=400, background_color="#0d1117",
                colormap=cmap, max_words=150, collocations=False,
            ).generate(text)
            return wc

        wc_obj = make_wc(wc_sent, year_range[0], year_range[1])
        if wc_obj:
            fig_wc, ax_wc = plt.subplots(figsize=(8, 4))
            fig_wc.patch.set_facecolor("#0d1117")
            ax_wc.set_facecolor("#0d1117")
            ax_wc.imshow(wc_obj, interpolation="bilinear")
            ax_wc.axis("off")
            st.pyplot(fig_wc)
        else:
            st.info("No text for this selection.")

    with col_kw:
        st.subheader("🔥 Top Keywords")
        kw_sent = st.selectbox("Sentiment", ["positive", "negative", "both"], key="kw")
        kw_n    = st.slider("# of keywords", 5, 30, 15)

        @st.cache_data
        def get_top_kw(sent, y0, y1, n):
            EXTRA_STOP = {
                "the","a","an","is","it","in","on","at","to","of","for",
                "and","or","but","i","my","me","we","be","was","are",
                "this","that","with","has","have","had","so","if","as",
                "by","do","not","no","get","got","just","im","its",
                "they","he","she","you","your","from","up","out","can",
                "will","all","one","more","some","what","about","who",
                "like","time","day","know","want","see","think","good",
                "go","today","now","would","when","than","how","them",
                "make","made","much","back","still","well","new","only",
                "way","could","over","been","there","use","into","here",
                "take","am","also","rt","via","via","http","com","www",
            }
            sub = df[(df["year"] >= y0) & (df["year"] <= y1)]
            if sent != "both":
                sub = sub[sub["sentiment_label"] == sent]
            words = " ".join(sub["clean_text"].dropna().astype(str)).split()
            words = [w for w in words if len(w) > 2 and w not in EXTRA_STOP]
            top   = Counter(words).most_common(n)
            return pd.DataFrame(top, columns=["word", "count"])

        kw_df = get_top_kw(kw_sent, year_range[0], year_range[1], kw_n)
        bcolor = POS_COLOR if kw_sent == "positive" else (NEG_COLOR if kw_sent == "negative" else "#58a6ff")
        fig_kw = px.bar(
            kw_df, x="count", y="word", orientation="h",
            color_discrete_sequence=[bcolor],
            labels={"count": "Frequency", "word": ""},
        )
        styled(fig_kw, height=460)
        fig_kw.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_kw, width="stretch")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — USERS
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("👤 Top 20 Users by Tweet Volume")
    col_u1, col_u2 = st.columns(2)

    top_users_df = (
        filtered.groupby("user").size()
        .reset_index(name="tweets")
        .sort_values("tweets", ascending=False).head(20)
    )
    top20 = top_users_df["user"].tolist()

    with col_u1:
        fig_u = px.bar(
            top_users_df, x="tweets", y="user", orientation="h",
            color="tweets", color_continuous_scale="Blues",
            labels={"tweets": "Tweet Count", "user": ""},
        )
        styled(fig_u, height=520, coloraxis_showscale=False)
        fig_u.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_u, width="stretch")

    with col_u2:
        st.subheader("Sentiment per Top User")
        usent = (
            filtered[filtered["user"].isin(top20)]
            .groupby(["user", "sentiment_label"]).size()
            .reset_index(name="count")
        )
        fig_us = px.bar(
            usent, x="user", y="count", color="sentiment_label",
            barmode="group",
            color_discrete_map={"positive": POS_COLOR, "negative": NEG_COLOR},
        )
        styled(fig_us, height=320, xaxis_tickangle=-45)
        st.plotly_chart(fig_us, width="stretch")

        # Average predicted positive probability per user
        st.subheader("Average Predicted Positive Probability")
        uconf = (
            filtered[filtered["user"].isin(top20)]
            .groupby("user")["confidence"].mean()
            .reset_index().sort_values("confidence", ascending=False)
        )
        uconf["confidence_pct"] = uconf["confidence"] * 100
        fig_uc = px.bar(
            uconf, x="user", y="confidence_pct",
            color="confidence_pct",
            color_continuous_scale=[[0, NEG_COLOR], [0.5, "#8b949e"], [1, POS_COLOR]],
            labels={"confidence_pct": "Avg P(positive) %", "user": ""},
        )
        fig_uc.add_hline(y=50, line_dash="dot", line_color="#8b949e",
                         annotation_text="50% threshold")
        styled(fig_uc, height=280, coloraxis_showscale=False, xaxis_tickangle=-45)
        st.plotly_chart(fig_uc, width="stretch")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — SEARCH
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🔍 Search Tweets")

    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        keyword = st.text_input(
            "Keyword / phrase", placeholder="e.g. happy, python, love ...", key="kw_search"
        )
    with col_s2:
        filter_sent = st.selectbox("In", ["All", "Positive only", "Negative only"])

    if keyword:
        base = filtered
        if filter_sent == "Positive only":
            base = filtered[filtered["sentiment_label"] == "positive"]
        elif filter_sent == "Negative only":
            base = filtered[filtered["sentiment_label"] == "negative"]

        hits = base[
            base["text"].str.contains(re.escape(keyword), case=False, na=False, regex=True)
        ]
        st.success(f"**{len(hits):,}** results for `{keyword}`")

        if len(hits):
            hpos = (hits["sentiment_label"] == "positive").sum()
            hneg = len(hits) - hpos
            hc1, hc2, hc3, hc4 = st.columns(4)
            hc1.metric("Total", len(hits))
            hc2.metric("😊 Positive", hpos)
            hc3.metric("😠 Negative", hneg)
            hc4.metric("Avg P(positive)", f"{hits['confidence'].mean()*100:.1f}%")

            st.dataframe(
                hits[["user", "text", "sentiment_label", "confidence", "year"]]
                .rename(columns={
                    "user": "User", "text": "Tweet",
                    "sentiment_label": "Sentiment",
                    "confidence": "P(pos)", "year": "Year",
                })
                .assign(**{"P(pos)": lambda d: d["P(pos)"].map("{:.3f}".format)})
                .head(50),
                width="stretch", hide_index=True,
            )
    else:
        st.info("Type a keyword above to search through tweets.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — REAL-TIME CLASSIFIER
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("💬 Real-Time Sentiment Classifier")
    st.markdown(
        "Type any tweet below. **Both models** (Naive Bayes + Logistic Regression) "
        "will classify it and show their individual confidence scores."
    )
    st.caption(
        "This is a binary statistical classifier. 'Uncertain' is a confidence band, "
        "not a neutral class. Sarcasm, irony, mixed sentiment, and missing conversation "
        "context can still produce incorrect predictions."
    )

    user_tweet = st.text_area(
        "Enter a tweet or sentence",
        height=130,
        placeholder="e.g. I absolutely love this! Best day ever.",
        key="rt_input",
    )

    col_btn, _ = st.columns([1, 5])
    with col_btn:
        go_btn = st.button("🔬 Classify", key="classify_btn")

    if go_btn:
        if user_tweet.strip():
            cleaned = clean_tweet(user_tweet)
            vec     = tfidf.transform([cleaned])

            if not cleaned:
                st.warning("No usable words remain after preprocessing. Enter some alphabetic text.")
                st.stop()

            nb_proba = nb_model.predict_proba(vec)[0]    # [P(neg), P(pos)]
            lr_proba = lr_model.predict_proba(vec)[0]

            nb_pos = nb_proba[1]
            lr_pos = lr_proba[1]
            selected_pos = lr_pos if best_name == "Logistic Regression" else nb_pos

            def prediction_display(p):
                """Map binary P(positive) to a label and prediction confidence."""
                if p >= 0.6:
                    return "Positive 😊", POS_COLOR, p
                if p <= 0.4:
                    return "Negative 😠", NEG_COLOR, 1 - p
                return "Uncertain", "#8b949e", max(p, 1 - p)

            nb_lbl, nb_col, nb_conf = prediction_display(nb_pos)
            lr_lbl, lr_col, lr_conf = prediction_display(lr_pos)
            selected_lbl, selected_col, selected_conf = prediction_display(selected_pos)

            # ── Result cards ─────────────────────────────────────────────────
            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                st.markdown(f"""
                <div style='background:rgba(22,27,34,.9);border:1px solid {nb_col};
                    border-radius:12px;padding:1.1rem 1.3rem;'>
                  <p style='color:#8b949e;margin:0;font-size:.8rem;font-weight:600'>NAIVE BAYES</p>
                  <h3 style='color:{nb_col};margin:.3rem 0'>{nb_lbl}</h3>
                  <p style='color:#e6edf3;margin:0'>Prediction confidence: <b>{nb_conf*100:.1f}%</b></p>
                  <p style='color:#8b949e;margin:.2rem 0 0'>P(positive): {nb_pos*100:.1f}%</p>
                </div>""", unsafe_allow_html=True)

            with rc2:
                st.markdown(f"""
                <div style='background:rgba(22,27,34,.9);border:1px solid {lr_col};
                    border-radius:12px;padding:1.1rem 1.3rem;'>
                  <p style='color:#8b949e;margin:0;font-size:.8rem;font-weight:600'>LOGISTIC REGRESSION</p>
                  <h3 style='color:{lr_col};margin:.3rem 0'>{lr_lbl}</h3>
                  <p style='color:#e6edf3;margin:0'>Prediction confidence: <b>{lr_conf*100:.1f}%</b></p>
                  <p style='color:#8b949e;margin:.2rem 0 0'>P(positive): {lr_pos*100:.1f}%</p>
                </div>""", unsafe_allow_html=True)

            with rc3:
                st.markdown(f"""
                <div style='background:rgba(22,27,34,.9);border:2px solid {selected_col};
                    border-radius:12px;padding:1.1rem 1.3rem;'>
                  <p style='color:#8b949e;margin:0;font-size:.8rem;font-weight:600'>SELECTED: {best_name.upper()}</p>
                  <h3 style='color:{selected_col};margin:.3rem 0'>{selected_lbl}</h3>
                  <p style='color:#e6edf3;margin:0'>Prediction confidence: <b>{selected_conf*100:.1f}%</b></p>
                  <p style='color:#8b949e;margin:.2rem 0 0'>P(positive): {selected_pos*100:.1f}%</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Gauge — Ensemble ─────────────────────────────────────────────
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=selected_pos * 100,
                number={"suffix": "%", "font": {"color": selected_col, "size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                    "bar": {"color": selected_col, "thickness": 0.28},
                    "bgcolor": "#161b22",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 40],   "color": "#2d1117"},
                        {"range": [40, 60],  "color": "#1c2030"},
                        {"range": [60, 100], "color": "#122116"},
                    ],
                    "threshold": {
                        "line": {"color": "#e6edf3", "width": 3},
                        "thickness": 0.8,
                        "value": selected_pos * 100,
                    },
                },
                title={"text": f"{best_name} P(Positive)", "font": {"color": TEXT_COLOR}},
            ))
            fig_gauge.update_layout(
                paper_bgcolor=CHART_BG, font=dict(color=TEXT_COLOR),
                height=300, margin=dict(l=40, r=40, t=60, b=20),
            )
            st.plotly_chart(fig_gauge, width="stretch")

            # ── Side-by-side bar: NB vs LR ───────────────────────────────────
            comp_bar = pd.DataFrame({
                "Model":     ["Naive Bayes", "Logistic Regression"],
                "P(pos)":    [nb_pos, lr_pos],
                "P(neg)":    [nb_proba[0], lr_proba[0]],
            })
            comp_bar_long = comp_bar.melt(id_vars="Model", var_name="Class", value_name="Probability")
            fig_comp = px.bar(
                comp_bar_long, x="Model", y="Probability", color="Class",
                barmode="group",
                color_discrete_map={"P(pos)": POS_COLOR, "P(neg)": NEG_COLOR},
                labels={"Probability": "Probability", "Model": ""},
                title="Model Probability Breakdown",
            )
            fig_comp.add_hline(y=0.5, line_dash="dot", line_color="#8b949e",
                               annotation_text="Decision boundary")
            styled(fig_comp, height=320)
            st.plotly_chart(fig_comp, width="stretch")

            # ── Cleaned text info ─────────────────────────────────────────────
            with st.expander("🔍 NLP Preprocessing Result"):
                st.markdown(f"**Original:** `{user_tweet}`")
                st.markdown(f"**After NLP pipeline:**  `{cleaned}`")
                st.caption("(URLs removed → mentions removed → lowercase → stopwords removed → stemmed)")
        else:
            st.warning("Please enter some text.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — MODEL REPORT
# ═════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("📊 Model Performance Report")

    if comp_data:
        best = comp_data["best_model"]

        # ── Accuracy comparison bars ──────────────────────────────────────────
        models_list = [k for k in comp_data if k != "best_model"]
        acc_df = pd.DataFrame([
            {
                "Model":    m,
                "Accuracy": comp_data[m]["accuracy"],
                "Train Time (s)": comp_data[m]["train_time"],
            }
            for m in models_list
        ])

        col_r1, col_r2 = st.columns(2)

        with col_r1:
            st.markdown("#### Accuracy")
            fig_acc = px.bar(
                acc_df, x="Model", y="Accuracy",
                color="Model",
                color_discrete_sequence=["#58a6ff", "#2ea043"],
                text_auto=".2f",
                labels={"Accuracy": "Accuracy (%)"},
            )
            fig_acc.update_traces(textposition="outside", textfont_color=TEXT_COLOR)
            styled(fig_acc, height=320, showlegend=False)
            fig_acc.update_yaxes(range=[70, 100])
            st.plotly_chart(fig_acc, width="stretch")

        with col_r2:
            st.markdown("#### Training Time")
            fig_time = px.bar(
                acc_df, x="Model", y="Train Time (s)",
                color="Model",
                color_discrete_sequence=["#58a6ff", "#2ea043"],
                text_auto=".2f",
            )
            fig_time.update_traces(textposition="outside", textfont_color=TEXT_COLOR)
            styled(fig_time, height=320, showlegend=False)
            st.plotly_chart(fig_time, width="stretch")

        # ── Detailed metrics table ────────────────────────────────────────────
        st.markdown("#### Detailed Metrics")
        metrics_rows = []
        for m in models_list:
            d = comp_data[m]
            metrics_rows.append({
                "Model":              m,
                "Accuracy (%)":       d["accuracy"],
                "Pos Precision (%)":  d["precision_pos"],
                "Pos Recall (%)":     d["recall_pos"],
                "Pos F1 (%)":         d["f1_pos"],
                "Neg Precision (%)":  d["precision_neg"],
                "Neg Recall (%)":     d["recall_neg"],
                "Neg F1 (%)":         d["f1_neg"],
                "Train Time (s)":     d["train_time"],
            })
        metrics_df = pd.DataFrame(metrics_rows)
        st.dataframe(metrics_df, width="stretch", hide_index=True)

        # ── Radar chart ────────────────────────────────────────────────────────
        st.markdown("#### Metric Radar – Model Comparison")
        categories = ["Accuracy", "Pos Precision", "Pos Recall", "Pos F1",
                      "Neg Precision", "Neg Recall", "Neg F1"]
        fig_radar = go.Figure()
        colors = ["#58a6ff", "#2ea043"]
        fill_colors = ["rgba(88,166,255,0.15)", "rgba(46,160,67,0.15)"]
        for i, m in enumerate(models_list):
            d = comp_data[m]
            vals = [
                d["accuracy"],
                d["precision_pos"], d["recall_pos"], d["f1_pos"],
                d["precision_neg"], d["recall_neg"], d["f1_neg"],
            ]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                name=m,
                fill="toself",
                line=dict(color=colors[i], width=2),
                fillcolor=fill_colors[i % len(fill_colors)],
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#161b22",
                radialaxis=dict(visible=True, range=[75, 100],
                                gridcolor=GRIDCOLOR, color="#8b949e"),
                angularaxis=dict(gridcolor=GRIDCOLOR, color=TEXT_COLOR),
            ),
            paper_bgcolor=CHART_BG,
            font=dict(color=TEXT_COLOR),
            height=420,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_radar, width="stretch")

        st.success(f"🏆 **Best Model: {best}**  —  Accuracy: {comp_data[best]['accuracy']:.2f}%")

        # ── NLP Pipeline explanation ───────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🔤 NLP Pipeline Used")
        pipeline_df = pd.DataFrame({
            "Step": ["1. Regex Cleaning", "2. Lowercase", "3. Stopword Removal",
                     "4. Porter Stemming", "5. TF-IDF Vectorization", "6. ML Model"],
            "What it does": [
                "Remove URLs, @mentions, #hashtags, numbers, special characters",
                "Normalize text (Hello → hello)",
                "Remove common English words (the, is, and ...) using NLTK",
                "Reduce words to root form (running → run, loved → love)",
                "Convert text to numeric features (unigrams + bigrams, max 60k features)",
                "Naive Bayes & Logistic Regression trained on the complete labeled dataset",
            ],
        })
        st.table(pipeline_df)
    else:
        st.warning("Run `python preprocess.py` to generate model comparison data.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 7 — VIRAL TWEETS
# ═════════════════════════════════════════════════════════════════════════════
with tab7:
    st.subheader("🚀 Most Confident Tweets")

    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.markdown("### 😊 Most Confidently Positive")
        top_pos = (
            filtered[filtered["sentiment_label"] == "positive"]
            .sort_values("confidence", ascending=False)
            .head(10)[["user", "text", "confidence", "year"]]
            .rename(columns={"user": "User", "text": "Tweet",
                             "confidence": "P(pos)", "year": "Year"})
        )
        st.dataframe(top_pos, width="stretch", hide_index=True)

    with col_v2:
        st.markdown("### 😠 Most Confidently Negative")
        top_neg = (
            filtered[filtered["sentiment_label"] == "negative"]
            .sort_values("confidence")
            .head(10)[["user", "text", "confidence", "year"]]
            .rename(columns={"user": "User", "text": "Tweet",
                             "confidence": "P(pos)", "year": "Year"})
        )
        st.dataframe(top_neg, width="stretch", hide_index=True)

    # Confidence scatter
    st.subheader("📊 Confidence Score Distribution — Sample")
    sample = filtered.sample(min(4000, len(filtered)), random_state=42)
    fig_sc = px.scatter(
        sample, x="year", y="confidence",
        color="sentiment_label",
        color_discrete_map={"positive": POS_COLOR, "negative": NEG_COLOR},
        opacity=0.4,
        hover_data={"user": True, "confidence": ":.3f"},
        labels={"confidence": "P(positive)", "year": "Year"},
    )
    fig_sc.add_hline(y=0.5, line_dash="dot", line_color="#8b949e",
                     annotation_text="Decision boundary (0.5)")
    styled(fig_sc, height=380)
    st.plotly_chart(fig_sc, width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#484f58;font-size:.82rem;padding:.5rem 0 1rem'>
  🐦 Tweet Sentiment Analysis &nbsp;·&nbsp;
  TF-IDF + Naive Bayes + Logistic Regression &nbsp;·&nbsp;
  Sentiment140 Dataset (1.6M tweets) &nbsp;·&nbsp;
  Built with Streamlit &amp; Plotly
</div>
""", unsafe_allow_html=True)
