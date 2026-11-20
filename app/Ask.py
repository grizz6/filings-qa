"""Ask page: question in, cited answer out."""

import altair as alt
import pandas as pd
import streamlit as st
from client import (
    ApiError,
    answer_markdown,
    call,
    highlighted_passage,
    load_config,
    safe_markdown,
    search_settings_sidebar,
    series_color,
    setup_problems,
    start_warmup,
)

from src.highlight import supporting_sentence

MUTED_BAR = "#a3a8b8"  # passages the answer did not cite

EXAMPLES = [
    "What does Tesla say about supply chain risk?",
    "How could cybersecurity incidents affect Walmart?",
    "What risks does Delta Air Lines see from fuel prices?",
    "How does Pfizer describe risks from drug pricing pressure?",
]

st.set_page_config(page_title="Filings Q&A", page_icon="📄", layout="centered")
start_warmup()  # load the search models while the visitor reads the page
cfg = load_config()
settings = search_settings_sidebar(cfg)

st.title("Filings Q&A")
st.write(
    "Ask about the **Risk Factors** section (Item 1A) of the latest annual report (10-K) of "
    "these companies. Answers come only from the filings and cite them."
)
st.caption(" · ".join(f"{name} ({t})" for t, name in cfg["companies"].items()))

try:
    missing = setup_problems()
except ApiError as exc:
    st.warning(str(exc))
    missing = []
if missing:
    st.error(
        f"**Setup incomplete:** missing {' and '.join(f'`{m}`' for m in missing)}. "
        "App owner: add them in Streamlit Cloud under **Manage app → Settings → Secrets** "
        "(or as environment variables), then reboot the app."
    )

if "question" not in st.session_state:
    st.session_state.question = ""

cols = st.columns(2)
for i, example in enumerate(EXAMPLES):
    if cols[i % 2].button(example, key=f"ex{i}", width="stretch"):
        st.session_state.question = example

with st.form("ask"):
    question = st.text_input(
        "Your question",
        key="question",
        max_chars=cfg["app"]["max_question_chars"],
        placeholder="e.g. What does Netflix say about competition?",
    )
    submitted = st.form_submit_button("Ask", type="primary")

if submitted and question.strip():
    with st.spinner("Searching the filings and writing a cited answer…"):
        try:
            result = call("POST", "/ask", json={"question": question, "settings": settings})
        except ApiError as exc:
            st.warning(str(exc))
            st.stop()
    # Kept so the answer stays on screen when the visitor uses its controls.
    st.session_state.last_answer = {"question": question, "result": result, "settings": settings}

last = st.session_state.get("last_answer")
if last:
    result = last["result"]
    if result["refused"]:
        st.info(
            "**Not found in the filings.** The Risk Factors sections do not answer this "
            "(they cover risks, not figures like revenue or names of executives)."
        )
    else:
        st.markdown(safe_markdown(result["answer"]))
        st.markdown("**Sources**")
        for c in result["citations"]:
            st.markdown(
                f"[{c['n']}] {c['ticker']} · `{c['id']}` · [open the 10-K]({c['source_url']})"
            )

    sources = result["sources"]
    by_id = {s["id"]: s for s in sources}
    cited = [c for c in result["citations"] if c["id"] in by_id]
    if cited:
        st.subheader("Where the answer comes from")
        st.caption(
            "Each cited passage, with the sentence that best matches what the answer says "
            "highlighted. The highlight is a reading aid found by shared words; the whole "
            "passage is what the answer cites."
        )
        for i, c in enumerate(cited):
            passage = by_id[c["id"]]
            text = passage.get("text") or passage["snippet"]
            sentence = supporting_sentence(result["answer"], c["n"], text)
            with st.expander(f"[{c['n']}] {c['ticker']} · {c['id']}", expanded=i == 0):
                st.markdown(highlighted_passage(text, sentence))
                st.markdown(f"[Open the 10-K]({c['source_url']})")

    if sources:
        st.subheader("How relevant each retrieved passage was")
        cited_ids = {c["id"] for c in cited}
        relevance = pd.DataFrame(
            [
                {
                    "passage": f"{i}. {s['id']}",
                    "similarity": s["score"],
                    "in the answer": "cited" if s["id"] in cited_ids else "not cited",
                    "rerank score": s.get("rerank_score"),
                }
                for i, s in enumerate(sources, 1)
            ]
        )
        chart = (
            alt.Chart(relevance)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X(
                    "similarity:Q",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(tickCount=5, format=".1f"),
                    title="similarity to the question",
                ),
                y=alt.Y(
                    "passage:N",
                    sort=None,
                    title=None,
                    scale=alt.Scale(paddingInner=0.3),
                    axis=alt.Axis(labelLimit=220, labelOverlap=False),
                ),
                color=alt.Color(
                    "in the answer:N",
                    scale=alt.Scale(
                        domain=["cited", "not cited"], range=[series_color(), MUTED_BAR]
                    ),
                    legend=alt.Legend(orient="top", title=None),
                ),
                tooltip=[
                    "passage",
                    "in the answer",
                    alt.Tooltip("similarity:Q", format=".2f"),
                    alt.Tooltip("rerank score:Q", format=".2f"),
                ],
            )
            .properties(height=32 * len(relevance) + 40)
        )
        st.altair_chart(chart, width="stretch")
        st.caption("Passages in the order the answer received them (after re-ranking, if on).")

    with st.expander(f"Retrieved excerpts ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            st.markdown(f"**[{i}] `{s['id']}`** · similarity {s['score']:.2f}")
            st.caption(s["snippet"])

    left, right = st.columns([3, 2])
    left.caption(f"Answered in {result['latency_ms'] / 1000:.1f} s · model {cfg['llm']['model']}")
    right.download_button(
        "Download answer (Markdown)",
        answer_markdown(last["question"], result, last["settings"], cfg["llm"]["model"]),
        file_name="filings-qa-answer.md",
        mime="text/markdown",
        on_click="ignore",
        width="stretch",
    )

st.divider()
st.caption(
    "Questions are logged (text, timing, scores) to monitor the app; see the Monitoring page. "
    "Not investment advice."
)
