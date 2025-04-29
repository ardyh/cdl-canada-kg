import streamlit as st
import pandas as pd
import numpy as np  # Make sure to import numpy here too
import altair as alt
import random
from datetime import datetime, timedelta
from utils.st_utils import get_kg_querier, fallback_narrative, fallback_pol

kg = get_kg_querier()

if "pol" not in st.session_state: 
    st.session_state.pol = fallback_pol
if "selected_narrative" not in st.session_state:
    st.session_state.selected_narrative = fallback_narrative
else:
    pol = st.session_state.pol

st.write(f"## {st.session_state.pol} :gray[| Narrative]")
if st.button(f"View Entity", key=f"btn_pol_det_main"):
    st.switch_page("pages/politician.py")

st.subheader(
    f"*\"{st.session_state.selected_narrative['narrative_cluster']}\"*",
    divider="rainbow"
)

narrative_data = kg.get_popular_narratives(
    st.session_state.selected_narrative['narrative_cluster_id']
)

st.subheader("Mentions", divider=None)
mentions_col = st.columns(2)
with mentions_col[0]:
    platforms = ["Bluesky", "X", "Reddit"]
    platform_counts = [
        narrative_data['count_bluesky'].values[0],
        narrative_data['count_x'].values[0],
        narrative_data['count_reddit'].values[0]
    ]

    platform_df = pd.DataFrame({
        "Platform": platforms,
        "Mentions": platform_counts
    })

    platform_chart = alt.Chart(platform_df).mark_bar().encode(
        y=alt.Y("Mentions:Q", title="Mentions"),
        x=alt.X("Platform:N", sort='-x',  axis=alt.Axis(labelAngle=0), title=None),
        color=alt.Color("Platform:N", legend=None)
    ).properties(height=200)

    st.altair_chart(platform_chart)

st.subheader("Posts", divider="rainbow")
LIMIT_POST_CHARS = 500

pair_pols = narrative_data['post_details'].values[0]

# Store data in session state for access in detail page
if "pair_pols" not in st.session_state:
    st.session_state.pair_pols = pair_pols

# Function to navigate to politician detail page
def view_post(post):
    st.session_state.selected_post = post
    st.switch_page("pages/post.py")

narr_posts_col_spec = [1, 2, 1]
narr_posts_header_cols = st.columns(narr_posts_col_spec)
narr_posts_header_texts = [
    "**Platform/ID**",
    "**Text**",
    ""
]
for i, _ in enumerate(narr_posts_header_texts):
    narr_posts_header_cols[i].write(narr_posts_header_texts[i])

for i, row in enumerate(pair_pols):
    # Create a clickable container
    with st.container():
        # Make the entire container clickable by using a callback when clicked
        
        # Display politician information
        narr_posts_cols = st.columns(narr_posts_col_spec)
        
        with narr_posts_cols[0]:
            st.write(f"**{row['platform']}**")
            st.write(f"`{row['id']}`")
        with narr_posts_cols[1]:
            st.write(f"""*{row['text'][:LIMIT_POST_CHARS]}{
                '...' if len(row['text']) > LIMIT_POST_CHARS else ''
            }*""")
        with narr_posts_cols[2]:
            click_post = st.button(
                f"View Post", 
                key=f"btn_post_{i}",
                use_container_width=True,
                )
        
        st.markdown("---")

        if click_post:
            view_post(row)

st.subheader("Narratives Included in this Cluster", divider="rainbow")
for i, row in enumerate(narrative_data['narratives'].values[0]):
    # Create a clickable container
    with st.container():
        st.write(f"- *\"{row}\"*")
