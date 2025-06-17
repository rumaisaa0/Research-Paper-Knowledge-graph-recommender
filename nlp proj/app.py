import streamlit as st
import textwrap
import requests

import backend as backend

# --- Streamlit app starts here ---

st.set_page_config(page_title="Research Paper Recommender + RAG Assistant", layout="wide")

st.title("📚 Research Paper Recommender + AI Summarizer")

query = st.text_input("Enter your query:", placeholder="e.g., Explain dynamic backtracking...")

if st.button("Get Recommendations and Summary"):
    if query.strip() == "":
        st.warning("Please enter a query!")
    else:
        with st.spinner('🔄 Generating recommendations and summary...'):
            papers,answer = backend.rag_pipeline(query)

        # Display Recommended Papers
        st.subheader("🔎 Recommended Papers")
        for idx, paper in enumerate(papers, 1):
            st.markdown(f"**{idx}. [{paper.get('title', 'No Title')}]({paper.get('url', '#')})**")
            st.markdown(f"**Abstract:** {paper.get('content', 'No Abstract')}")
            st.markdown("---")

        # Display RAG Answer
        st.subheader("📝 LLM Generated Answer (RAG)")
        wrapped_answer = textwrap.fill(answer, width=100)
        st.write(wrapped_answer)
