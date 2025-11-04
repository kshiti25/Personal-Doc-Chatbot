import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# we’ll reuse your Docling loader
from docling_loader import DoclingLoader

DB_DIR = "chroma_db_openai"
DOCS_DIR = "data/docs"


def get_vectorstore():
    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model="text-embedding-3-small",
    )
    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )
    return vectordb, embeddings


def build_rag_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-3.5-turbo",
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Use ONLY the provided context to answer. "
                "If the answer is not in the context, say you don't know."
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}"
            ),
        ]
    )

    def format_docs(docs):
        return "\n\n".join(
            f"Source: {d.metadata.get('source', 'unknown')}\n{d.page_content}"
            for d in docs
        )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return rag_chain, retriever


def ingest_uploaded_file(uploaded_file, vectordb):
    """
    Save the uploaded file to disk, read it with Docling, chunk, and add to Chroma.
    """
    os.makedirs(DOCS_DIR, exist_ok=True)

    # 1. save the file
    save_path = os.path.join(DOCS_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 2. parse with docling
    loader = DoclingLoader(save_path)
    docs = loader.load()

    # 3. split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        add_start_index=True,
    )
    splits = splitter.split_documents(docs)

    # 4. add to existing vectorstore
    # we can call .add_documents on the Chroma instance we already loaded
    vectordb.add_documents(splits)

    return len(splits)


def main():
    st.set_page_config(page_title="Personal AI Chatbot", page_icon="🤖")
    st.title("🤖 Personal Docs Chatbot")
    st.caption("Upload a doc, then ask questions from it.")

    # make sure DB exists (if not, create empty one)
    os.makedirs(DOCS_DIR, exist_ok=True)

    vectordb, _ = get_vectorstore()
    rag_chain, retriever = build_rag_chain(vectordb)

    # 📁 uploader
    st.subheader("Upload documents")
    uploaded_file = st.file_uploader(
        "Drop a PDF/DOCX/PPTX here", type=["pdf", "docx", "pptx", "txt"], accept_multiple_files=False
    )
    if uploaded_file is not None:
        with st.spinner("Reading and indexing your file..."):
            num_chunks = ingest_uploaded_file(uploaded_file, vectordb)
        st.success(f"Indexed {uploaded_file.name} into the database ({num_chunks} chunks).")

    st.divider()

    # 💬 chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Ask your question here..."):
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing your documents..."):
                answer = rag_chain.invoke(question)
                st.markdown(answer.content)

                docs = retriever.invoke(question)
                with st.expander("📂 Sources used"):
                    for d in docs:
                        st.write(f"**{d.metadata.get('source', 'unknown')}**")
                        st.write(d.page_content[:400] + "...")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer.content}
        )


if __name__ == "__main__":
    main()
