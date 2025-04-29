import streamlit as st
from utils.st_utils import get_kg_querier, fallback_post

kg = get_kg_querier()

if "selected_post" not in st.session_state:
    # DEBUG
    st.session_state.selected_post = fallback_post

st.write(f"## {st.session_state.selected_post['platform']} | `{st.session_state.selected_post['id']}`")
if st.button("Back to Narrative"):
    st.switch_page("pages/narrative.py")

st.write(f"{st.session_state.selected_post['text']}")

st.subheader("Entities Mentioned", divider="rainbow")
triples = kg.get_post_entities(st.session_state.selected_post['id'])

# Function to navigate to politician detail page
def view_politician(politician_name):
    st.session_state.pol = politician_name
    st.switch_page("pages/politician.py")

def smart_triple_renderer(res):
    all_triples = []
    cur_ent, cur_action, cur_obj, cur_narr = set(), set(), set(), set()
    for i, r in res.iterrows():
        if len(cur_ent) == 0 and len(cur_action) == 0:
            cur_ent.add(r["actor"])
            cur_action.add(r["action"])
            cur_obj.add(r["target"])
            cur_narr.add(r["narrative"])
        elif r["actor"] in cur_ent or r["action"] in cur_action:
            if r["actor"] in cur_ent:
                cur_action.add(r["action"])
            if r["action"] in cur_action:
                cur_ent.add(r["actor"])
            cur_obj.add(r["target"])
            cur_narr.add(r["narrative"])
        else:
            all_triples.append((list(cur_ent), list(cur_action), list(cur_obj), list(cur_narr)))
            cur_ent, cur_action, cur_obj, cur_narr = set([r["actor"]]), set([r["action"]]), set([r["target"]]), set([r["narrative"]])
    all_triples.append((list(cur_ent), list(cur_action), list(cur_obj), list(cur_narr)))
    return all_triples

triples_header_cols = st.columns(4)
triples_header_cols[0].write("**Entity**")
triples_header_cols[1].write("**Action**")
triples_header_cols[2].write(f"**Object**")
triples_header_cols[3].write("**Narrative**")

smart_triples = smart_triple_renderer(triples)
for i, row in enumerate(smart_triples):
    # Create a clickable container
    with st.container():        
        triples_cols = st.columns(4)
        
        with triples_cols[0]:
            for j, e in enumerate(row[0]):
                click_post_ent = st.button(
                    f"{e[:17]}{'...' if len(e) > 17 else ''}", 
                    key=f"btn_post_ent_{i}_{j}_{e.replace(' ', '_')}",
                    use_container_width=True,
                    )
                if click_post_ent:
                    view_politician(e)
                
        with triples_cols[1]:
            for a in row[1]:
                st.write(a)
                
        with triples_cols[2]:
            for j, o in enumerate(row[2]):
                click_post_obj = st.button(
                    f"{o[:17]}{'...' if len(o) > 17 else ''}", 
                    key=f"btn_post_obj_{i}_{j}_{o.replace(' ', '_')}",
                    use_container_width=True,
                    )
                if click_post_obj:
                    view_politician(o)
                
        with triples_cols[3]:
            for n in row[3]:
                st.write(n)
                
        
        st.markdown("---")

