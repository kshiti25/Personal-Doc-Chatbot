🤖 Personal AI Chatbot (RAG-Based Document Assistant)

Your Personal AI Chatbot is a Retrieval-Augmented Generation (RAG) system that can read your personal documents (PDFs, DOCXs, etc.) and answer questions about them intelligently using OpenAI’s GPT models.

Built with LangChain, Docling, ChromaDB, and Streamlit, it gives you your own private AI assistant — one that knows your data.

🧠 Key Features

🗂️ Reads and understands PDFs/DOCX using Docling

🔍 Retrieval-Augmented Generation (RAG) pipeline for precise answers

🧭 Semantic Search using ChromaDB

💬 Chat interface built with Streamlit

🔐 Private and local — your data never leaves your system

⚙️ Embeddings powered by OpenAI (text-embedding-3-small)

🧩 Modular design (Docling → Embeddings → Chroma → GPT → Streamlit)

🏗️ Architecture Overview
graph TD

    A[📁 Documents (PDF/DOCX)] --> B[DoclingLoader
    B --> C[LangChain Text Splitter]
    C --> D[OpenAI Embeddings]
    D --> E[Chroma Vector DB]
    F[User Question] --> G[OpenAI Embeddings (query)]
    G --> H[Retriever from Chroma]
    H --> I[Relevant Chunks]
    I --> J[Prompt Template + GPT-3.5-turbo]
    J --> K[🧠 Final Answer Shown in Streamlit Chat]

⚙️ Tech Stack
| Component        | Purpose                                      |
| ---------------- | -------------------------------------------- |
| **Python 3.10+** | Core programming language                    |
| **Streamlit**    | Chat UI                                      |
| **LangChain**    | Framework to chain LLM + retrieval           |
| **OpenAI API**   | LLM (GPT) + embeddings                       |
| **Docling**      | Extracts text from PDFs/DOCXs                |
| **ChromaDB**     | Local vector database for document retrieval |
| **dotenv**       | Securely loads API keys                      |

📁 Folder Structure
personal-ai-chatbot/

│

├── data/

│   └── docs/                ← place your PDFs/DOCXs here

│

├── chroma_db_openai/        ← auto-generated local vector DB (after running ingest.py)

│

├── .env                     ← stores your OpenAI API key (DO NOT COMMIT)

├── requirements.txt          ← dependencies

├── docling_loader.py         ← handles text extraction

├── ingest.py                 ← builds embeddings + vector DB

├── app.py                    ← Streamlit chat interface

└── README.md


🔑 Environment Setup
1️⃣ Clone the repository
git clone https://github.com/yourusername/personal-ai-chatbot.git
cd personal-ai-chatbot

2️⃣ Create a virtual environment
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Window

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Add your OpenAI API key
Create a file named .env in the root directory:
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx

5️⃣ Add your documents
Place your .pdf or .docx files in:
data/docs/

6️⃣ Build your local vector database
Run the ingestion script (this reads, chunks, embeds, and stores your docs):
python3 ingest.py

If successful, you’ll see a new folder chroma_db_openai/.

7️⃣ Launch the chatbot
python3 -m streamlit run app.py
Then open the local URL shown in your terminal (usually http://localhost:8501).

💬 How It Works

Document Processing
docling_loader.py uses Docling to extract clean text from your PDFs/DOCXs.

Chunking & Embeddings
ingest.py splits long text into overlapping chunks.
Each chunk is embedded (converted to a vector) using OpenAI’s embedding model.

Vector Storage
The embeddings are stored locally in ChromaDB.

Retrieval + Generation
When you ask a question in the Streamlit UI, LangChain retrieves the most relevant chunks from ChromaDB.
GPT-3.5 reads the context and generates a grounded answer.

Display + Sources
The chatbot responds with an answer and shows which files it used as sources.
