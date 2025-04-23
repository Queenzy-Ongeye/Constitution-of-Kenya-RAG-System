import streamlit as st
import requests

API_URL = "http://localhost:8000/query"

st.set_page_config(page_title="Kenyan Constitution Assistant", page_icon="📘")
st.title("📘 Ask the Kenyan Constitution")
st.markdown("Type your question below and get answers from the Constitution using AI.")

# Input field for the user question
question = st.text_input("🔍 Ask a question:", placeholder="e.g., What are the rights of citizens in Kenya?")
top_k = st.slider("Number of answers to return:", min_value=1, max_value=10, value=3)

# On submit
if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Querying the Constitution..."):
            try:
                response = requests.post(API_URL, json={"question": question})
                response.raise_for_status()
                results = response.json().get("results", [])

                if not results:
                    st.info("No relevant information found.")
                else:
                    st.success(f"Top {top_k} answers found:")
                    for i, res in enumerate(results[:top_k], 1):
                        st.markdown(f"### {i}. Title: *{res.get('title', 'Unknown')}*")
                        st.markdown(f"> {res['text']}")
                        st.caption(f"🧠 Similarity distance: `{res['distance']:.4f}`")

            except requests.exceptions.RequestException as e:
                st.error(f"API Error: {str(e)}")
