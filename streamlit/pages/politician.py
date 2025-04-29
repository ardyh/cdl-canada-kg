import streamlit as st
import pandas as pd
import altair as alt
from utils.politician_summaries import politician_summaries
from utils.st_utils import get_kg_querier, fallback_pol

kg = get_kg_querier()

st.set_page_config(
    page_title="Narrative Tracker",
)

if "pol" not in st.session_state:
    st.session_state.pol = fallback_pol
else:
    st.session_state.pol_mentions = kg.get_politician_mentions(st.session_state.pol)

# Title
def capitalize_name(name):
    """
    Capitalizes the first letter of each word in a name.
    Example: "john doe" -> "John Doe"
    """
    # Split the name by spaces
    words = name.split(" ")
    
    # Capitalize the first letter of each word
    capitalized_words = [word.capitalize() for word in words]
    
    # Join the words back together with spaces
    return " ".join(capitalized_words)

st.write(f"# {capitalize_name(st.session_state.pol)}")
if st.button("Back to Dashboard"):
    st.switch_page("app.py")

summary = politician_summaries.get(st.session_state.pol, None)
if summary:
    st.write(f"*{summary}*")
    st.caption(f"~ Wikipedia, summarized by Perplexity")


st.subheader("Mentions", divider="rainbow")
mentions_col = st.columns(2)
with mentions_col[0]:
    st.write("### All Platforms")
    st.write(f"# :blue[{st.session_state.pol_mentions['post_counts'].values[0]}]")

with mentions_col[1]:
    # Add platform breakdown
    st.markdown("### Platform Breakdown")
    platforms = ["Bluesky", "X", "Reddit"]
    platform_counts = [
        st.session_state.pol_mentions['count_bluesky'].values[0],
        st.session_state.pol_mentions['count_x'].values[0],
        st.session_state.pol_mentions['count_reddit'].values[0]
    ]

    platform_df = pd.DataFrame({
        "Platform": platforms,
        "Mentions": platform_counts
    })

    platform_chart = alt.Chart(platform_df).mark_bar().encode(
        x=alt.X("Mentions:Q", title="Mentions"),
        y=alt.Y("Platform:N", sort='-x', title=None),
        color=alt.Color("Platform:N", legend=None)
    ).properties(height=200)

    st.altair_chart(platform_chart, use_container_width=True)

LIMIT_CLUSTERS = 5
LIMIT_SAMPLES = 2
st.subheader(f"Top-{LIMIT_CLUSTERS} Narratives", divider="rainbow")
narratives_data = kg.get_politician_narratives(st.session_state.pol, limit=LIMIT_CLUSTERS)

# Function to navigate to politician detail page
def view_narrative_details(narrative):
    st.session_state.selected_narrative = narrative
    st.switch_page("pages/narrative.py")

narr_header_cols = st.columns([2, 2, 1])

narr_header_cols[0].write("**Cluster**")
# narr_header_cols[1].write("**Count**")
narr_header_cols[1].write("**Samples**")
narr_header_cols[2].write("")

for i, row in narratives_data.iterrows():
    # Create a clickable container
    with st.container():
        # Make the entire container clickable by using a callback when clicked
        
        # Display politician information
        narr_cols = st.columns([2, 2, 1])
        
        with narr_cols[0]:
            st.write(f"*\"{row['narrative_cluster']}\"*")
            st.metric(label="Mentions", value=row['count_mentions'])
        # with narr_cols[1]:
        #     st.write(row['count_mentions'])
        with narr_cols[1]:
            st.write(row['sample_narratives'][:LIMIT_SAMPLES])
        with narr_cols[2]:
            click_narr = st.button("View details", key=f"btn_narr_pairs_{i}")
        
        st.markdown("---")

        if click_narr:
            view_narrative_details({
                "narrative_cluster_id": row['narrative_cluster_id'], 
                "narrative_cluster": row['narrative_cluster']
            })

st.subheader("Frequently Mentioned With", divider="rainbow")
LIMIT_PAIRS = 5
pair_pols = kg.get_co_mentions(st.session_state.pol, limit=LIMIT_PAIRS)

# Store data in session state for access in detail page
if "pair_pols" not in st.session_state:
    st.session_state.pair_pols = pair_pols

# Function to navigate to politician detail page
def view_politician(politician_name):
    st.session_state.pol = politician_name
    st.switch_page("pages/politician.py")

freq_pair_col_spec = 2
freq_pair_header_cols = st.columns(freq_pair_col_spec)

freq_pair_header_cols[0].write("**Pair**")
freq_pair_header_cols[1].write(f"**Sample Narratives ({st.session_state.pol})**")

for i, row in pair_pols.iterrows():
    # Create a clickable container
    with st.container():
        # Make the entire container clickable by using a callback when clicked
        
        # Display politician information
        freq_pair_cols = st.columns(freq_pair_col_spec)
        
        with freq_pair_cols[0]:
            click_pol = st.button(
                f"{row['entity2']}", 
                key=f"btn_pol_pairs_{i}",
                use_container_width=True,
                )
            st.metric(label="Mentions", value=row['co_occurrences'])
        with freq_pair_cols[1]:
            st.write(row['narratives1'][:LIMIT_SAMPLES])
        
        st.markdown("---")

        if click_pol:
            view_politician(row['entity2'])

