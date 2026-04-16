from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import Callable, Optional
import os
import tempfile

load_dotenv()

# -----------------------------------------------
# Singleton — embedding model loads once per process
# -----------------------------------------------
_embeddings: Optional[HuggingFaceEmbeddings] = None

def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
        )
    return _embeddings


# -----------------------------------------------
# 1. Document Loading
# -----------------------------------------------
def load_document(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path) if suffix.lower() == ".pdf" else TextLoader(tmp_path, encoding="utf-8")
    return loader.load()


# -----------------------------------------------
# 2. Chunking + Batched Embedding + FAISS
# -----------------------------------------------
def build_vectorstore(
    docs,
    progress_callback: Optional[Callable[[int, int], None]] = None,
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,          # 400 words — mid-range of the 200–500 word spec
        chunk_overlap=50,        # 50-word overlap for context continuity
        length_function=lambda text: len(text.split()),  # count words, not chars
    )
    chunks = splitter.split_documents(docs)
    total = len(chunks)

    embeddings = _get_embeddings()
    BATCH = 100  # embed 100 chunks at a time → avoids RAM spikes

    # First batch bootstraps the FAISS index
    vectorstore = FAISS.from_documents(chunks[:BATCH], embeddings)
    if progress_callback:
        progress_callback(min(BATCH, total), total)

    # Stream remaining batches directly into the index
    for start in range(BATCH, total, BATCH):
        vectorstore.add_documents(chunks[start : start + BATCH])
        if progress_callback:
            progress_callback(min(start + BATCH, total), total)

    return vectorstore


# -----------------------------------------------
# 3. QA Chain (Groq LLaMA3.1 + strict RAG prompt)
# -----------------------------------------------
def get_qa_chain(vectorstore, api_key: str):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt_template = """You are a helpful assistant.
Answer ONLY using the provided context. Do NOT use outside knowledge.
If the answer is not in the context, say: "I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
    )
