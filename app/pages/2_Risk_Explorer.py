"""Risk Explorer: which risks each company's filing discusses, as a topic x company heatmap."""

import altair as alt
import pandas as pd
import streamlit as st
from client import SEQUENTIAL, ApiError, call, load_config, series_color

MAX_CUSTOM_TOPICS = 3

st.set_page_config(page_title="Risk Explorer · Filings Q&A", page_icon="🗺️", layout="wide")
cfg = load_config()

st.title("Risk Explorer")
st.write(
    "How closely each company's **Risk Factors** section speaks to each risk topic. A cell's "
    "score is the similarity (0 to 1) between the topic and that company's best-matching "
    "passage: higher means the filing has a passage squarely about it. No AI model writes "
    "anything here; the scores come straight from the search index."
)

st.session_state.setdefault("custom_topics", {})
custom = st.session_state.custom_topics

with st.expander("Add your own topic"):
    with st.form("topic", clear_on_submit=True):
        c1, c2 = st.columns([1, 3])
        name = c1.text_input("Name", max_chars=30, placeholder="e.g. Water")
        text = c2.text_input(
            "What to look for", max_chars=200, placeholder="e.g. water scarcity and drought"
        )
        if st.form_submit_button("Add topic"):
            if not name.strip() or len(text.strip()) < 3:
                st.warning("Give the topic a name and a short description.")
            elif len(custom) >= MAX_CUSTOM_TOPICS and name.strip() not in custom:
                st.warning(f"Up to {MAX_CUSTOM_TOPICS} topics of your own; remove one first.")
            else:
                custom[name.strip()] = text.strip()
    if custom:
        st.caption("Your topics: " + " · ".join(custom))
        if st.button("Remove my topics"):
            custom.clear()
            st.rerun()


@st.cache_data(ttl=3600, show_spinner=False)
def load_map(extra: tuple) -> dict:
    if extra:
        return call("POST", "/risk-map", json={"topics": dict(extra)})
    return call("GET", "/risk-map")


try:
    with st.spinner("Scoring every topic against every filing…"):
        data = load_map(tuple(sorted(custom.items())))
except ApiError as exc:
    st.warning(str(exc))
    st.stop()

companies = data["companies"]
cells = pd.DataFrame(data["cells"])
if cells.empty:
    st.info("The search index is empty, so there is nothing to map yet.")
    st.stop()
cells["company"] = cells["ticker"].map(lambda t: f"{companies[t]} ({t})")

controls = st.columns([3, 2])
shown = controls[0].multiselect("Topics", data["topics"], default=data["topics"])
color_by = controls[1].radio(
    "Colour by",
    ["Similarity", "Relative within topic"],
    horizontal=True,
    help="Relative: 1 is the company that discusses the topic most, 0 the one that does least.",
)
if not shown:
    st.info("Pick at least one topic.")
    st.stop()

view = cells[cells["topic"].isin(shown)].copy()
low = view.groupby("topic")["score"].transform("min")
high = view.groupby("topic")["score"].transform("max")
view["relative"] = ((view["score"] - low) / (high - low).where(high > low, 1)).round(3)
field, title = ("score", "similarity") if color_by == "Similarity" else ("relative", "relative")

order = [f"{companies[t]} ({t})" for t in companies]
base = alt.Chart(view).encode(
    x=alt.X(
        "ticker:N",
        sort=list(companies),
        title=None,
        scale=alt.Scale(paddingInner=0.04),  # gaps between cells show the page itself
        axis=alt.Axis(orient="top", labelAngle=0, labelOverlap=False),
    ),
    y=alt.Y("topic:N", sort=shown, title=None, scale=alt.Scale(paddingInner=0.08)),
)
heat = base.mark_rect(cornerRadius=4).encode(
    color=alt.Color(f"{field}:Q", title=title, scale=alt.Scale(range=SEQUENTIAL)),
    tooltip=[
        "topic",
        "company",
        alt.Tooltip("score:Q", title="similarity", format=".2f"),
        alt.Tooltip("relative:Q", title="relative within topic", format=".2f"),
        alt.Tooltip("chunk_id:N", title="best passage"),
    ],
)
# White text from the mid-blue steps up; dark text on the light steps.
midpoint = view[field].min() + 0.4 * (view[field].max() - view[field].min())
labels = base.mark_text(fontSize=11).encode(
    text=alt.Text("score:Q", format=".2f"),
    color=alt.condition(alt.datum[field] > midpoint, alt.value("#ffffff"), alt.value("#31333f")),
)
st.altair_chart((heat + labels).properties(height=34 * len(shown) + 60), width="stretch")

st.subheader("Look closer")
left, right = st.columns([2, 3])
with left:
    topic = st.selectbox("Topic", shown)
    by_topic = view[view["topic"] == topic].sort_values("score", ascending=False)
    bars = (
        alt.Chart(by_topic)
        .mark_bar(color=series_color(), cornerRadiusEnd=4)
        .encode(
            x=alt.X(
                "score:Q",
                title="similarity",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(tickCount=5, format=".1f"),
            ),
            y=alt.Y(
                "company:N",
                sort=None,
                title=None,
                scale=alt.Scale(paddingInner=0.35),
                axis=alt.Axis(labelLimit=220, labelOverlap=False),
            ),
            tooltip=["company", alt.Tooltip("score:Q", title="similarity", format=".2f")],
        )
        .properties(height=28 * len(by_topic) + 30)
    )
    st.altair_chart(bars, width="stretch")
with right:
    pick = st.selectbox("Company", by_topic["company"].tolist())
    cell = by_topic[by_topic["company"] == pick].iloc[0]
    st.markdown(
        f"**{pick}** on **{topic}** · similarity {cell['score']:.2f} · `{cell['chunk_id']}`"
    )
    st.caption(cell["snippet"])
    st.markdown(f"[Open the 10-K]({cell['source_url']})")
    if st.button("Ask about this", type="primary"):
        name = companies[cell["ticker"]]
        description = data["descriptions"].get(topic, topic.lower())
        st.session_state.question = f"What does {name} say about {description}?"
        st.switch_page("Ask.py")

with st.expander("Table view"):
    table = view.pivot(index="topic", columns="company", values="score").reindex(shown)
    st.dataframe(table[[c for c in order if c in table.columns]], width="stretch")
