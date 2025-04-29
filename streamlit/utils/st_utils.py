from utils.kg_querier import KnowledgeGraphQuerier
import streamlit as st
import os

def get_kg_querier():
    if "kg_querier" not in st.session_state:
        uri = st.secrets["neo4j"]["uri"]
        user = st.secrets["neo4j"]["username"]
        password = st.secrets["neo4j"]["password"]
        database = st.secrets["neo4j"]["database"]

        st.session_state["kg_querier"] = KnowledgeGraphQuerier(
            uri=uri,
            user=user,
            password=password,
            database=database
        )

        def _cleanup():
            if 'kg_querier' in st.session_state:
                st.session_state.kg_querier.close()
                del st.session_state.kg_querier
        
        st.session_state.on_cleanup = _cleanup
        
    return st.session_state.kg_querier

fallback_pol = "mark carney"

fallback_post = {
    "id": "1iro6yw",
    "platform": "Reddit",
    "text": """One of the most striking characteristics of Pierre Poilievre's rhetoric is anti-intellectualism. He speaks in monosyllables, wielding "Verb the Noun!" type slogans which have no real substance behind them. Even more concerning is the way he regards academia with disdain, especially those sections of it he considers "woke". He sees the struggles people are facing, and the hopelessness they feel. He takes advantage of it by weaponizing their righteous anger, directing it at the people who are suffering most under our economic system. Most importantly, he paints himself as the only solution, the only one who can fix the system by ridding it of inefficiencies and corrupt elements. Some people view this as a new, alien phenomenon, but it's not.

In the early days of fascist Italy, there was a marked shift in academia away from the humanities and towards a utilitarian approach to education. 

Basically, if you weren't at university to enlarge the economy or advance industry in some manner, your field was considered useless. This bears striking resemblance to the kind of right-wing populist rhetoric which raves about "underwater basket weavers", CRT, etc which is so commonplace today. 

Things seem hopeless because we were told (in the early years of neoliberalism) that this mechanicist approach to education would uplift us, but instead it put us into debt and never gave the rewards we were made to expect. Now most of us can't even afford it, and so who do we blame? 

We've been so atomized and propagandized that we blame each other, even the people trying to help us (protestors, teachers, unions) or especially the most vulnerable people (immigrants, the homeless, queer people) instead of the billionaire oligarchs who profit from our ever-worsening conditions... because we've been taught that they've earned their billions, that if we want to live well we should aspire to become them. This aspiration towards capital is exactly why so many of us fall for Poilievre's savior rhetoric.

If we ever want to be free of this, of the nihilism and the hatred, we need to realize from where the chains originate... the problem isn't external, and the system hasn't failed or been corrupted, because it wasn't built for us in the first place. It was built for people like Pierre Poilievre, and things will only change when we realize the solution is in our hands, through our labour and our unity. No one is going to come down from above and save us, not even Mark Carney. We have to save ourselves.
  """}

fallback_narrative = {
        "narrative_cluster": "mark carney is accused of prioritizing u.s. interests over canadian jobs and investments.",
        "narrative_cluster_id": 9245
    }

