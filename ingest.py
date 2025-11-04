import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docling_loader import DoclingLoader

DATA_DIR = "data/docs"
DB_DIR = "chroma_db_openai"

def load_all_docs(data_dir=DATA_DIR):
    docs=[]
    for filename in os.listdir(data_dir):
        filepath=os.path.join(data_dir,filename)
        if os.path.isfile(filepath):
            loader=DoclingLoader(filepath)
            docs.extend(loader.load())
    return docs

def main():
    print("🔄 Loading documents...")
    docs = load_all_docs()
    print(f"✅ Loaded {len(docs)} docs")

    splitter=RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        add_start_index=True,
    )
    splits=splitter.split_documents(docs)
    print(f"✂️ Split into {len(splits)} chunks")

    embeddings=OpenAIEmbeddings(api_key=OPENAI_API_KEY,model="text-embedding-3-small")

    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    print("💾 Chroma database saved successfully!")

if __name__ == "__main__":
    main()
    
