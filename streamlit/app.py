import streamlit as st
import pandas as pd
import altair as alt
from utils.st_utils import get_kg_querier
from utils.politician_summaries import politician_summaries
from datetime import datetime
from streamlit_agraph import agraph, Node, Edge, Config

kg = get_kg_querier()

st.set_page_config(
    page_title="Narrative Tracker",
)

st.write("""
# :telescope: Narascope | :gray[Narrative Tracker]

*A bird's eye view of what people are saying in social media.*
         
This system is built on top of a :rainbow[Knowledge Graph] which connects between social media posts by extracting the entities and narratives mentioned in each post.
Our end goal is to help mitigate the spread of misinformation and disinformation, 
but on this project, we focus on one of the foundational components towards that goal: **Scalable monitoring of narratives on social media**. 
For this project, we use the 2025 Canadian Election as a base context and focus on the narratives regarding the key political figures
""")

pols = [
    "mark carney",
    "pierre poilievre",
    "yves-françois blanchet",
    "jagmeet singh",
]

default_pol_idx = 0

# Input for stock pols
pol = st.selectbox(
    "Get started",
    options=pols,
    index=None,
    placeholder="Select a key figure",
)

if pol:
    st.session_state.pol = pol
    st.switch_page("pages/politician.py")

count_data = {
    "Posts": 959,
    "Entities": 1162,
    "Narratives": 1313,
    "Actions": 409,
    "Users": 160
}
data_cols = st.columns(len(count_data))

for i, (key, value) in enumerate(count_data.items()):
    data_cols[i].metric(key, value)

# Create nodes with appropriate colors
# Details on nodes, edges, and config was assisted by Claude
nodes = [
    Node(id="User", label="User", size=30, color="#b2d27a"),  # Light green
    Node(id="Post", label="Post", size=30, color="#d4a2cc"),  # Light purple
    Node(id="Narrative", label="Narrative", size=30, color="#e6d5f2"),  # Very light purple
    Node(id="Entity", label="Entity", size=30, color="#c4937d"),  # Light brown
    Node(id="Action", label="Action", size=30, color="#b3dbf2")  # Light blue
]

# Create edges with labels
edges = [
    Edge(source="User", target="User", label="INTERACTS_WITH", type="CURVE_SMOOTH"),
    Edge(source="User", target="Post", label="POSTED", type="CURVE_SMOOTH"),
    Edge(source="Post", target="Narrative", label="CONTAINS_NARRATIVE", type="CURVE_SMOOTH"),
    Edge(source="Post", target="Entity", label="MENTIONS", type="CURVE_SMOOTH"),
    Edge(source="Post", target="Action", label="MENTIONS", type="CURVE_SMOOTH"),
    Edge(source="Entity", target="Narrative", label="PART_OF_NARRATIVE", type="CURVE_SMOOTH"),
    Edge(source="Action", target="Narrative", label="PART_OF_NARRATIVE", type="CURVE_SMOOTH"),
    Edge(source="Entity", target="Action", label="DOES", type="CURVE_SMOOTH"),
    Edge(source="Action", target="Entity", label="AFFECTS", type="CURVE_SMOOTH")
]

# Configure the graph
config = Config(
    width=800,
    height=500,
    directed=True,
    physics={
        "enabled": True,
        "forceAtlas2Based": {
            "gravitationalConstant": -100,  # More negative value increases repulsion
            "centralGravity": 0.005,  # Decreased to allow nodes to spread out
            "springLength": 250,  # Increased spring length for more spacing
            "springConstant": 0.05,  # Decreased for more flexibility
            "damping": 0.9,
            "avoidOverlap": 1.0  # Maximum avoidance of node overlap
        },
        "solver": "forceAtlas2Based",
        "stabilization": {
            "enabled": True,
            "iterations": 2000,  # More iterations for better layout
            "updateInterval": 50
        },
        "minVelocity": 0.75,
        "maxVelocity": 50
    },
    hierarchical=False,
    nodeHighlightBehavior=True,
    linkHighlightBehavior=True,
    highlightColor="#F7A7A6",
    collapsible=True,
    # Customize node appearance
    node={
        "labelProperty": "label",
        "fontColor": "black",
        "fontSize": 14,
        "fontWeight": "normal"
    },
    # Customize link appearance
    link={
        "labelProperty": "label",
        "renderLabel": True,
        "fontColor": "gray",
        "fontSize": 10,
        "fontWeight": "normal"
    }
)

# Render the graph in Streamlit
agraph(nodes=nodes, edges=edges, config=config)