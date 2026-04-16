from rag_pipeline import load_document, build_vectorstore, get_qa_chain
from dotenv import load_dotenv
import sys
import os

load_dotenv()

def main():
    print("\n=== Mini RAG — Document Q&A ===\n")

    # Get document path
    doc_path = input("Enter document path (PDF or TXT): ").strip()

    if not os.path.exists(doc_path):
        print(f"Error: File not found — {doc_path}")
        sys.exit(1)

    # Create a fake file object that mimics Streamlit's UploadedFile
    class LocalFile:
        def __init__(self, path):
            self.name = os.path.basename(path)
            self._path = path
        def read(self):
            with open(self._path, "rb") as f:
                return f.read()

    print("\nLoading document...")
    docs = load_document(LocalFile(doc_path))

    print("Building vector store...")
    vectorstore = build_vectorstore(docs)

    print("Setting up QA chain...")
    chain = get_qa_chain(vectorstore)

    print("\nDone! You can now ask questions. Type 'exit' to quit.\n")

    while True:
        query = input("Your question: ").strip()

        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not query:
            continue

        print("\nThinking...\n")
        result = chain.invoke({"query": query})

        print(f"Answer: {result['result']}")
        print("\n--- Retrieved Chunks ---")
        for i, doc in enumerate(result["source_documents"]):
            print(f"\nChunk {i+1}: {doc.page_content[:200]}...")
        print("------------------------\n")

if __name__ == "__main__":
    main()