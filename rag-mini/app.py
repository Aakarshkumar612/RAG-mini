import streamlit as st
import os
from dotenv import load_dotenv
from rag_pipeline import load_document, build_vectorstore, get_qa_chain

load_dotenv()

st.set_page_config(page_title="Mini RAG Q&A", page_icon="📄", layout="wide")
st.title("📄 Mini RAG Q&A System")
st.markdown("Upload a document **(TXT or PDF, up to 20 MB)** and ask questions grounded in it.")

if not os.getenv("GROQ_API_KEY"):
    st.error("⚠️ GROQ_API_KEY not set. Add it to your `.env` file.")
    st.stop()

if "chain" not in st.session_state:
    st.session_state.chain = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

uploaded_file = st.file_uploader("Upload a document (TXT or PDF)", type=["txt", "pdf"])

# Reset chain when a different file is uploaded
if uploaded_file and uploaded_file.name != st.session_state.doc_name:
    st.session_state.chain = None

if uploaded_file:
    if st.button("Process Document"):
        bar = st.progress(0, text="Splitting document into chunks…")

        def on_progress(done: int, total: int) -> None:
            bar.progress(int(done / total * 100), text=f"Embedding chunks… {done}/{total}")

        docs = load_document(uploaded_file)
        vectorstore = build_vectorstore(docs, progress_callback=on_progress)
        bar.progress(100, text="Finalising QA chain…")
        st.session_state.chain = get_qa_chain(vectorstore)
        st.session_state.doc_name = uploaded_file.name
        bar.empty()
        st.success(f"✅ '{uploaded_file.name}' processed — ask your questions below.")

# -----------------------------------------------
# Q&A Interface
# -----------------------------------------------
if st.session_state.chain:
    st.divider()
    st.subheader("Ask a Question")
    query = st.text_input("Your question", placeholder="e.g. How many paid leaves are allowed?")

    if st.button("Get Answer") and query:
        with st.spinner("Thinking…"):
            result = st.session_state.chain.invoke({"query": query})

        st.markdown("### Answer")
        st.info(result["result"])

        with st.expander("📚 Retrieved Context (top 3 chunks)"):
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**Chunk {i + 1}:**")
                st.write(doc.page_content)
                st.divider()
