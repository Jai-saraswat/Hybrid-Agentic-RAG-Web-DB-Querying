# 🤖 Hybrid Agentic RAG: Web & Database Querying System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.6-green.svg)](https://langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1.19-orange.svg)](https://github.com/langchain-ai/langgraph)

A sophisticated **Agentic RAG (Retrieval-Augmented Generation)** pipeline built with **LangGraph** and **LangChain** that intelligently routes user queries to appropriate data sources using AI-powered decision making. The system combines PDF-based document retrieval, Web-based retrieval with conversational AI capabilities, leveraging FAISS vector search and HuggingFace embeddings for efficient semantic search.

> **⚠️ Current Status:** Both the Initial Agentic RAG and Web Querying pipelines have been made. Integration is in progress.

---

## 📋 Table of Contents

- [Architecture Overview](#-architecture-overview)
- [System Flow](#-system-flow)
- [Tech Stack](#-tech-stack)
- [Key Features](#-key-features)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🏗️ Architecture Overview

The system implements an **Agentic RAG architecture** using LangGraph's state management and routing capabilities. The agent intelligently decides whether to:
- Retrieve information from indexed PDF documents (RAG mode)
- Respond using general conversational AI (Chat mode)
- **(Coming Soon)** Fetch real-time information from the web (Web Search mode)

### Architecture Diagram for Web Querying Pipeline
```
┌──────────────────────────────────────────────────────────────┐
│                          USER QUERY                           │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                  QUERY GENERATION LAYER                      │
│  - spaCy Semantic Analysis (NER, Noun Chunks, POS)          │
│  - Tokenization & Phrase Protection                          │
│  - LLM Reformulation (Groq GPT-OSS-120B → 5 Queries)        │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                     WEB SEARCH LAYER                         │
│  - Bing Search Execution                                     │
│  - URL Extraction & Decoding                                 │
│  - Duplicate Removal                                         │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                 CONTENT RETRIEVAL LAYER                      │
│  - HTTP Fetch (Headers, Timeout)                             │
│  - Main Text Extraction (Readability + Fallback HTML)       │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                VALIDATION & SCORING LAYER                    │
│  - English Check                                             │
│  - Keyword Match                                             │
│  - Trusted Domain Boost                                      │
│  - Length Heuristics → Page Score Filter                     │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                DOCUMENT & CHUNKING LAYER                     │
│  - JSON Document Assembly                                    │
│  - Text Cleaning & Word Chunking (120 / min 15)             │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│              EMBEDDING & VECTOR DATABASE LAYER               │
│  - MiniLM-L6-v2 Embeddings (384-Dim)                         │
│  - FAISS Index Build & Save                                  │
│  - FAISS Load & Similarity Search (Top-K=3)                  │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                        FINAL OUTPUT                          │
│  - Retrieved Chunks                                          │
│  - Document Count                                            │
│  - Chunk Count                                               │
│  - RAG-Ready Context                                         │
└──────────────────────────────────────────────────────────────┘
```

### Architecture Diagram For Agentic Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUERY                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH WORKFLOW                           │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  ROUTER NODE                              │  │
│  │         (ChatGroq - GPT-OSS-120B)                        │  │
│  │   Analyzes query intent & routes to appropriate node     │  │
│  └────────────┬──────────────────────────┬───────────────────┘  │
│               │                          │                       │
│               ▼                          ▼                       │
│  ┌────────────────────────┐  ┌─────────────────────────┐       │
│  │   VECTORSTORE NODE     │  │    LLM CHAT NODE        │       │
│  │  (RAG Pipeline)        │  │  (Conversational AI)    │       │
│  │                        │  │                         │       │
│  │  1. Semantic Search    │  │  Direct LLM Response    │       │
│  │     via FAISS          │  │  for casual queries     │       │
│  │  2. Retrieve Top-K     │  │                         │       │
│  │     Chunks (k=3)       │  │  Model: Llama-4-Scout   │       │
│  │  3. Context Building   │  │  Temperature: 0.7       │       │
│  │  4. Generate Answer    │  │                         │       │
│  │                        │  │                         │       │
│  │  Model: Llama-4-Scout  │  └─────────────────────────┘       │
│  └────────────────────────┘                                     │
│               │                          │                       │
│               └──────────┬───────────────┘                       │
│                          ▼                                       │
│              ┌─────────────────────┐                            │
│              │   FINAL RESPONSE    │                            │
│              └─────────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    SUPPORTING SYSTEMS                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────┐    ┌─────────────────────────┐      │
│  │   DOCUMENT PIPELINE  │    │   VECTOR DATABASE       │      │
│  │                      │    │                         │      │
│  │  1. PDF Loading      │───▶│  FAISS Index            │      │
│  │     (PyMuPDF)        │    │                         │      │
│  │  2. Text Cleaning    │    │  - 384 Dimensions       │      │
│  │  3. Page Merging     │    │  - Cosine Similarity    │      │
│  │  4. Text Chunking    │    │  - 275 Chunks Stored    │      │
│  │     (1200/200)       │    │                         │      │
│  │  5. Embedding        │    └─────────────────────────┘      │
│  │     (MiniLM-L6-v2)   │                                      │
│  └──────────────────────┘                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Current Main System Flow

### 1. Document Ingestion Pipeline
```
PDF Document → PyMuPDF Loader → Text Cleaning → Page Consolidation 
    → Recursive Text Splitting (chunk_size=1200, overlap=200)
    → HuggingFace Embeddings (all-MiniLM-L6-v2, 384-dim)
    → FAISS Vector Database
```

### 2. Query Processing Pipeline
```
User Query → LangGraph Entry Point → Router Node (LLM-based Intent Classification)
    ├─→ [Technical/ML Query] → Vectorstore Node → Semantic Search (FAISS)
    │                         → Retrieve Top-3 Chunks → Context Building
    │                         → LLM Generation with Context → Response
    │
    └─→ [Casual/General Query] → LLM Chat Node → Direct LLM Response
```

### 3. LangGraph State Management
The system uses a shared state graph with:
- **Messages**: Conversation history (annotated with `add_messages`)
- **Route Decision**: Current routing decision (`vectorstore` or `llm_chat`)
- **Conditional Edges**: Dynamic routing based on LLM classification

---

## 🛠️ Tech Stack

### Core Frameworks
| Technology | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.8+ | Primary language |
| **LangChain** | 0.2.6 | RAG orchestration & chains |
| **LangGraph** | 0.1.19 | Agentic workflow & state management |
| **LangChain Community** | 0.2.6 | Vector stores & document loaders |

### Machine Learning & Embeddings
| Technology | Version | Purpose |
|-----------|---------|---------|
| **PyTorch** | 2.3.1 | Deep learning backend |
| **Transformers** | 4.41.2 | HuggingFace models |
| **Sentence Transformers** | 2.6.1 | Embedding generation |
| **HuggingFace Hub** | 0.23.4 | Model management |

### Vector Database
| Technology | Version | Purpose |
|-----------|---------|---------|
| **FAISS** | (via LangChain) | Efficient similarity search |
| **ChromaDB** | 0.5.5 | Alternative vector store (available) |

### LLM Providers
| Technology | Version | Purpose |
|-----------|---------|---------|
| **Groq** | (langchain-groq 0.1.4) | Fast inference API |
| **Ollama** | (langchain-ollama 0.1.2) | Local LLM deployment |

**Models Used:**
- **Router LLM**: `openai/gpt-oss-120b` (temperature=0, deterministic routing)
- **Responder LLM**: `meta-llama/llama-4-scout-17b-16e-instruct` (temperature=0.7, creative responses)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)

### Document Processing
| Technology | Version | Purpose |
|-----------|---------|---------|
| **PyMuPDF** | 1.26.7 | Fast PDF parsing |
| **pypdf** | 4.2.0 | PDF text extraction |
| **Unstructured** | 0.14.6 | Document structure parsing |
| **pytesseract** | 0.3.10 | OCR capabilities |
| **pdf2image** | 1.17.0 | PDF to image conversion |

### Utilities
| Technology | Version | Purpose |
|-----------|---------|---------|
| **NumPy** | 1.26.4 | Numerical operations |
| **Pydantic** | 2.7.1 | Data validation |
| **python-dotenv** | 1.0.1 | Environment management |
| **Rich** | 13.7.1 | Terminal formatting |

---

## ✨ Key Features

### Current Implementation
- ✅ **Intelligent Query Routing** - LLM-powered decision engine classifies query intent  
- ✅ **Semantic Search** - FAISS-based vector similarity search (384-dimensional embeddings)  
- ✅ **Context-Aware RAG** - Retrieves top-3 relevant chunks with metadata  
- ✅ **Conversational AI** - Fallback to general LLM for casual queries  
- ✅ **GPU Acceleration** - CUDA support for embeddings (if available)  
- ✅ **Robust Document Processing** - Text cleaning, deduplication, and smart chunking  
- ✅ **LangGraph Workflow** - State-based agentic execution with conditional edges  
- ✅ **Visualization** - Mermaid diagram generation for workflow debugging  

### Coming Soon 🚀
- 🔲 **Web Search Integration** - Real-time web context retrieval for current events  
- 🔲 **Multi-Source Fusion** - Combine PDF, web, and database results  
- 🔲 **Advanced Routing** - Tri-modal routing (PDF/Web/Chat)  
- 🔲 **Citation Tracking** - Source attribution for all retrieved information  
- 🔲 **Conversational Memory** - Multi-turn dialogue with context retention  

---

## 🔍 How It Works

### Step 1: Document Processing
1. **PDF Loading**: PyMuPDF extracts text from PDF files page by page
2. **Text Cleaning**: Regex-based cleaning removes excessive whitespace, page numbers, and artifacts
3. **Page Consolidation**: Fragments are merged into coherent page-level documents
4. **Chunking**: RecursiveCharacterTextSplitter creates overlapping chunks (1200 chars, 200 overlap)
5. **Embedding**: HuggingFace's MiniLM model generates 384-dimensional vectors
6. **Indexing**: FAISS builds an efficient similarity search index

### Step 2: Query Processing
1. **User Input**: Query enters the LangGraph workflow at the START node
2. **Router Node**: GPT-OSS-120B analyzes intent and classifies as:
   - `vectorstore`: Technical/factual queries requiring document retrieval
   - `llm_chat`: Casual/general queries for direct LLM response
3. **Conditional Routing**: Graph edges direct flow based on classification

### Step 3: Response Generation

#### For Technical Queries (vectorstore route):
1. Query embedding generated using same MiniLM model
2. FAISS performs cosine similarity search (k=3 chunks)
3. Retrieved chunks assembled into context
4. Llama-4-Scout generates grounded answer using context
5. Response returned to user

#### For Casual Queries (llm_chat route):
1. Query passed directly to Llama-4-Scout
2. LLM generates conversational response
3. No retrieval or context injection

---

## 📁 Project Structure

```
Hybrid-Agentic-RAG-Web-DB-Querying/
│
├── Notebook/
│   └── Agentic_RAG.ipynb          # Main implementation notebook
│
├── Data/                          # PDF documents for indexing
│   └── Machine_learning.pdf       # Example ML textbook
│
├── faiss_db/                      # FAISS vector database storage
│   ├── index.faiss                # FAISS index file
│   └── index.pkl                  # Metadata pickle file
│
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
├── LICENSE                        # MIT License
└── README.md                      # This file
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for acceleration)
- Tesseract OCR (for PDF OCR capabilities)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Jai-saraswat/Hybrid-Agentic-RAG-Web-DB-Querying.git
cd Hybrid-Agentic-RAG-Web-DB-Querying
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACE_TOKEN=your_hf_token_here  # Optional
```

### Step 5: Install Tesseract (Optional)
**Windows:**
```bash
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Update path in notebook: pytesseract.pytesseract.tesseract_cmd
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

---

## 💻 Usage

### Running the Notebook
1. Launch Jupyter:
```bash
jupyter notebook
```

2. Open `Notebook/Agentic_RAG.ipynb`

3. Run cells sequentially:
   - Check GPU availability
   - Load environment variables
   - Process PDF documents
   - Build FAISS index
   - Initialize LangGraph workflow
   - Test queries

### Example Queries

#### Technical Query (routes to vectorstore):
```python
query = "What are neural networks?"
result = app.invoke({"messages": [HumanMessage(content=query)]})
print(result["messages"][-1].content)
```

#### Casual Query (routes to llm_chat):
```python
query = "Hello! How are you?"
result = app.invoke({"messages": [HumanMessage(content=query)]})
print(result["messages"][-1].content)
```

### Visualizing the Workflow
```python
from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))
```

---

## 📊 Configuration

### Chunking Parameters
```python
CHUNK_SIZE = 1200      # Maximum characters per chunk
CHUNK_OVERLAP = 200    # Overlapping context between chunks
```

### Embedding Model
```python
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

### LLM Settings
```python
# Router (deterministic)
router_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# Responder (creative)
responder_llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.7)
```

### Retrieval Settings
```python
k = 3  # Number of chunks to retrieve
```

---

## 🗺️ Roadmap

### Phase 1: Current Implementation ✅
- [x] PDF document processing pipeline
- [x] FAISS vector database integration
- [x] LangGraph agentic workflow
- [x] Bi-modal routing (RAG vs Chat)
- [x] GPU-accelerated embeddings

### Phase 2: Web Search Integration 🚧
- [ ] Add DuckDuckGo/Google Search API
- [ ] Implement web scraping node
- [ ] Tri-modal routing (PDF/Web/Chat)
- [ ] Result fusion and ranking

### Phase 3: Advanced Features 📅
- [ ] Multi-document RAG
- [ ] Conversational memory
- [ ] Citation tracking
- [ ] Custom evaluation metrics
- [ ] Streamlit/Gradio UI

### Phase 4: Database Integration 📅
- [ ] SQL database querying
- [ ] Graph database support
- [ ] Hybrid retrieval strategies

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **LangChain** team for the incredible framework
- **LangGraph** for agentic workflow capabilities
- **HuggingFace** for open-source embeddings
- **FAISS** (Facebook AI) for efficient vector search
- **Groq** for fast LLM inference

---

## 📧 Contact

**Jai Saraswat**  
GitHub: [@Jai-saraswat](https://github.com/Jai-saraswat)  
Project Link: [https://github.com/Jai-saraswat/Hybrid-Agentic-RAG-Web-DB-Querying](https://github.com/Jai-saraswat/Hybrid-Agentic-RAG-Web-DB-Querying)

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ using LangChain, LangGraph, and FAISS

</div>
