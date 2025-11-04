import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

DB_DIR="chroma_db_openai"

def get_vectorstore():
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return vectordb

def build_rag_chain(vectordb):
    retriever=vectordb.as_retriever(search_kwargs={"k": 4})
    llm=ChatOpenAI(api_key=OPENAI_API_KEY, model='gpt-3.5-turbo', temperature=0)


    prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful assistant. Use ONLY the provided context to answer. "
                "If the answer is not in the context, say you don't know."
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}"
            ),
        ])

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

def main():
    st.set_page_config(page_title="Personal AI Chatbot", page_icon="🤖")
    st.title("🤖 Personal Docs Chatbot")
    st.caption("Ask questions from your uploaded documents")

    if not os.path.exists(DB_DIR):
        st.error("⚠️ No database found! Run `python ingest.py` first to build your Chroma DB.")
        st.stop()
    vectordb = get_vectorstore()
    rag_chain, retriever = build_rag_chain(vectordb)

    # Store chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
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

        st.session_state.messages.append({"role": "assistant", "content": answer.content})

if __name__ == "__main__":
    main()