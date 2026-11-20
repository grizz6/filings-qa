"""Compare: one question, answered from each chosen company's filing, side by side."""

import altair as alt
import pandas as pd
import streamlit as st
from client import (
    ApiError,
    call,
    load_config,
    safe_markdown,
    search_settings_sidebar,
    series_color,
)

MIN_COMPANIES, MAX_COMPANIES = 2, 4
EXAMPLES = [
    ("How do they describe supply chain risk?", ["TSLA", "NKE", "WMT"]),
    ("How could cybersecurity incidents affect them?", ["JPM", "MSFT", "NFLX"]),
    ("What do they say about climate change?", ["XOM", "DAL", "PFE"]),
]

st.set_page_config(page_title="Compare · Filings Q&A", page_icon="⚖️", layout="wide")
cfg = load_config()
names = cfg["companies"]
settings = search_settings_sidebar(cfg, choose_companies=False)

st.title("Compare companies")
st.write(
    "Ask one question and read how each company answers it in its own **Risk Factors** "
    "section. Each answer comes only from that company's 10-K and cites it."
)
st.caption(
    "Write the question without a company name, e.g. *How do they describe supply chain "
    "risk?* Each company counts as one question toward the app's daily limit."
)

st.session_state.setdefault("compare_question", "")
st.session_state.setdefault("compare_companies", ["TSLA", "NKE"])
cols = st.columns(len(EXAMPLES))
for i, (question, tickers) in enumerate(EXAMPLES):
    help_text = "Compares " + ", ".join(names[t] for t in tickers)
    if cols[i].button(question, key=f"cmp{i}", help=help_text, width="stretch"):
        st.session_state.compare_question = question
        st.session_state.compare_companies = tickers

with st.form("compare"):
    question = st.text_input(
        "Question",
        key="compare_question",
        max_chars=cfg["app"]["max_question_chars"],
        placeholder="e.g. How do they describe supply chain risk?",
    )
    chosen = st.multiselect(
        f"Companies ({MIN_COMPANIES} to {MAX_COMPANIES})",
        list(names),
        key="compare_companies",
        format_func=lambda t: f"{names[t]} ({t})",
        max_selections=MAX_COMPANIES,
    )
    submitted = st.form_submit_button("Compare", type="primary")

if submitted:
    if not question.strip():
        st.warning("Write a question first.")
        st.stop()
    if len(chosen) < MIN_COMPANIES:
        st.warning(f"Pick at least {MIN_COMPANIES} companies.")
        st.stop()
    results = {}
    progress = st.progress(0.0, text="Asking each filing…")
    for i, ticker in enumerate(chosen):
        progress.progress(i / len(chosen), text=f"Reading {names[ticker]}'s filing…")
        try:
            results[ticker] = call(
                "POST",
                "/ask",
                json={"question": question, "settings": {**settings, "companies": [ticker]}},
            )
        except ApiError as exc:
            results[ticker] = {"error": str(exc)}
    progress.empty()
    st.session_state.compare_results = {"question": question, "answers": results}

done = st.session_state.get("compare_results")
if not done:
    st.stop()

st.subheader(done["question"])
answers = done["answers"]
fit = pd.DataFrame(
    [
        {"company": f"{names[t]} ({t})", "similarity": max(s["score"] for s in r["sources"])}
        for t, r in answers.items()
        if r.get("sources")
    ]
)
if not fit.empty:
    st.caption(
        "How directly each filing addresses the question: the similarity of its best passage."
    )
    chart = (
        alt.Chart(fit)
        .mark_bar(color=series_color(), cornerRadiusEnd=4)
        .encode(
            x=alt.X(
                "similarity:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(tickCount=5, format=".1f"),
                title="best passage similarity",
            ),
            y=alt.Y(
                "company:N",
                sort="-x",
                title=None,
                scale=alt.Scale(paddingInner=0.35),
                axis=alt.Axis(labelLimit=220, labelOverlap=False),
            ),
            tooltip=["company", alt.Tooltip("similarity:Q", format=".2f")],
        )
        .properties(height=34 * len(fit) + 30)
    )
    st.altair_chart(chart, width="stretch")

for col, (ticker, result) in zip(st.columns(len(answers)), answers.items(), strict=True):
    with col, st.container(border=True):
        st.markdown(f"#### {names[ticker]} ({ticker})")
        if "error" in result:
            st.warning(result["error"])
            continue
        if result["refused"]:
            st.info("Not found in this filing.")
        else:
            st.markdown(safe_markdown(result["answer"]))
            for c in result["citations"]:
                st.markdown(f"[{c['n']}] `{c['id']}` · [open the 10-K]({c['source_url']})")
        st.caption(f"Answered in {result['latency_ms'] / 1000:.1f} s")
