# 🛡️ Project Aegis — Agentic Document Extraction (ADE) Platform

> **AI for Bharat Hackathon | Theme 3 — Dual-Cover Bidding Compliance Automation**

Project Aegis is a state-of-the-art **Agentic Document Extraction** platform that replaces error-prone manual cross-referencing in government procurement with a visually grounded, explainable AI pipeline. It leverages **Vision-Language Models (VLMs)** to reason directly over document images, anchoring every compliance decision to specific spatial coordinates on the source evidence.

---

## 📑 Table of Contents

- [Architecture Overview](#-architecture-overview)
- [Tech Stack](#-tech-stack)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Running the Application](#-running-the-application)
- [Usage Guide](#-usage-guide)
- [Project Structure](#-project-structure)
- [Pipeline Phases](#-pipeline-phases)
- [API Endpoints](#-api-endpoints)

---

## 🏗 Architecture Overview

```
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Phase 1     │───▶│  Phase 2         │───▶│  Phase 3         │───▶│  Phase 4         │───▶│  Phase 5         │
│  Sanitizer   │    │  Deconstruction  │    │  VLM Extraction  │    │  Decision Engine │    │  Benchmarker     │
│  (Presidio + │    │  Agent (LLaMA    │    │  (Nemotron VL    │    │  (Spatial        │    │  (Accuracy,      │
│   GLiNER)    │    │   3.2 11B)       │    │   8B + Instructor│    │   Grounding)     │    │   Precision,     │
│              │    │                  │    │                  │    │                  │    │   Recall)        │
└──────────────┘    └──────────────────┘    └──────────────────┘    └──────────────────┘    └──────────────────┘
```

The pipeline is orchestrated via **LangGraph** as a stateful, directed acyclic graph with 4 sequential nodes.

---

## 🧰 Tech Stack

| Layer              | Technology                                                   |
|--------------------|--------------------------------------------------------------|
| **Orchestration**  | LangGraph (StateGraph)                                       |
| **VLM Inference**  | NVIDIA NIM API (`llama-3.1-nemotron-nano-vl-8b-v1`)          |
| **Rule Extraction**| NVIDIA NIM API (`meta/llama-3.2-11b-vision-instruct`)        |
| **Schema Enforce** | Instructor (Pydantic-enforced structured outputs)            |
| **PII Sanitizer**  | Microsoft Presidio + GLiNER (zero-shot NER)                  |
| **RAG Engine**     | Qdrant (vector DB) + Sentence-Transformers + BM25            |
| **PDF Processing** | PyMuPDF (fitz)                                               |
| **API Server**     | FastAPI + Uvicorn                                            |
| **Frontend**       | React 18 (CDN) + Tailwind CSS                                |
| **Language**       | Python 3.10+                                                 |

---

## ✅ Prerequisites

Before setting up the project, ensure you have the following installed:

1. **Python 3.10+** — [Download Python](https://www.python.org/downloads/)
2. **pip** — Comes bundled with Python
3. **Git** — [Download Git](https://git-scm.com/downloads)
4. **NVIDIA NIM API Key** — Required for VLM inference. Obtain one from the [NVIDIA NIM Portal](https://build.nvidia.com/)

---

## 🚀 Installation & Setup

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd THeme3\ -\ Copy
```

### Step 2: Create a Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

If a `requirements.txt` is not present, install the core dependencies manually:

```bash
pip install fastapi uvicorn python-dotenv python-multipart
pip install langgraph langchain-core
pip install PyMuPDF Pillow pandas openpyxl xlrd
pip install instructor openai
pip install presidio-analyzer presidio-anonymizer gliner
pip install sentence-transformers qdrant-client rank-bm25
pip install evaluate rouge-score nltk
pip install pydantic torch transformers
pip install spacy
python -m spacy download en_core_web_lg
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root (if not already present):

```env
NVIDIA_NIM_API_KEY="your-nvidia-nim-api-key-here"
```

| Variable              | Description                                      | Required |
|-----------------------|--------------------------------------------------|----------|
| `NVIDIA_NIM_API_KEY`  | API key for NVIDIA NIM hosted VLM endpoints      | ✅ Yes   |
| `LOAD_VLM`            | Enable/disable live VLM API calls (default: `True`) | Optional |
| `USE_LLM_FOR_RULES`   | Use LLM for rule extraction (default: `True`)    | Optional |

---

## ▶️ Running the Application

### Start the API Server

```bash
python api.py
```

This will:
1. Load the **GLiNER** NER model and **Sentence-Transformer** embeddings at startup (one-time, ~30s)
2. Initialize all pipeline engines (Sanitizer, RAG, VLM, Decision Engine)
3. Compile the **LangGraph** orchestration workflow
4. Start the FastAPI server on **`http://localhost:8001`**

### Access the Dashboard

Open your browser and navigate to:

```
http://localhost:8001
```

The React-based dashboard will load automatically from `static/index.html`.

---

## 📖 Usage Guide

### Using the Web Dashboard

1. **Open** `http://localhost:8001` in your browser
2. **Upload Documents** — Click the upload area and select your files:
   - **Tender PDF** (e.g., `CRPF_Tender_TE_406_Master.pdf`) — The master tender document
   - **Bidder PDF** (e.g., `Bidder_Alpha_Tech_Portfolio.pdf`) — A bidder's portfolio
3. **Click "Start AI Evaluation"** — The pipeline will process the documents through all 5 phases
4. **View Results** — The dashboard displays:
   - **Metrics**: Accuracy, Precision, Recall, and Status
   - **Compliance Audit Ledger**: Per-requirement breakdown with extracted values, confidence scores, and compliance status
   - **Visual Grounding Evidence**: Annotated PDF with red translucent highlights on source evidence regions

### Using the CLI (Direct Orchestrator)

```bash
python orchestrator.py
```

This runs the pipeline directly on the `work_953482` directory (or modify the `target` variable in `__main__`).

### Testing Individual Components

```bash
# Test PII Sanitizer
python sanitizer.py

# Test Tender Deconstruction Agent
python deconstruction_agent.py

# Test VLM Engine
python vlm_engine.py

# Test Decision Engine
python decision_engine.py

# Test Benchmarker
python benchmarker.py

# Test RAG Engine
python rag_engine.py

# Test Data Ingestion
python data_ingestion.py
```

---

## 📁 Project Structure

```
THeme3 - Copy/
│
├── api.py                      # FastAPI server — main entrypoint
├── orchestrator.py             # LangGraph orchestrator (4-node pipeline)
├── models.py                   # Pydantic data models (GoldenRule, CriterionEvaluation)
│
├── sanitizer.py                # Phase 1: PII detection & sanitization (Presidio + GLiNER)
├── deconstruction_agent.py     # Phase 2: Tender deconstruction (LLM + heuristic fallback)
├── vlm_engine.py               # Phase 3: VLM-based value extraction (NVIDIA NIM)
├── decision_engine.py          # Phase 4: Numerical reasoning & spatial grounding
├── benchmarker.py              # Phase 5: Metrics computation (Accuracy, Precision, Recall)
│
├── rag_engine.py               # Hybrid RAG (Qdrant + BM25 + Sentence-Transformers)
├── data_ingestion.py           # Multi-format document parser (PDF, Excel, CSV)
│
├── static/
│   └── index.html              # React dashboard (single-file, CDN-based)
│
├── results/                    # Output directory for visually grounded PDFs
├── qdrant_data/                # Local Qdrant vector database storage
├── .env                        # Environment variables (NVIDIA_NIM_API_KEY)
│
├── CRPF_Tender_TE_406_Master.pdf       # Sample tender document
├── Bidder_Alpha_Tech_Portfolio.pdf      # Sample bidder portfolio
├── Bidder_Beta_Innovations_Portfolio.pdf
├── Bidder_Gamma_Defense_Portfolio.pdf
│
├── GEMINI.md                   # Project mandates & engineering rules
└── T3_Pipeline.pdf             # Pipeline architecture diagram
```

---

## 🔄 Pipeline Phases

| Phase | Component                  | Description                                                                                          |
|-------|----------------------------|------------------------------------------------------------------------------------------------------|
| **1** | `sanitizer.py`             | Detects and replaces PII (PAN, Aadhaar, GSTIN, names) using length-preserving synthetic replacement  |
| **2** | `deconstruction_agent.py`  | Extracts "Golden Rules" from the tender via LLM, with VLM visual discovery fallback for scanned PDFs |
| **3** | `vlm_engine.py`            | Sends document page images to NVIDIA NIM VLM for raw value extraction with tight bounding boxes      |
| **4** | `decision_engine.py`       | Performs numerical threshold validation, coordinate translation, and deterministic text grounding     |
| **5** | `benchmarker.py`           | Computes Accuracy, Precision, and Recall against ground truth for compliance classification          |

---

## 🌐 API Endpoints

| Method | Endpoint                  | Description                              |
|--------|---------------------------|------------------------------------------|
| `GET`  | `/`                       | Serves the React dashboard               |
| `POST` | `/api/analyze`            | Upload documents and run the ADE pipeline|
| `GET`  | `/api/results/{filename}` | Download visually grounded PDF results   |

### Example: Upload via cURL

```bash
curl -X POST http://localhost:8001/api/analyze \
  -F "files=@CRPF_Tender_TE_406_Master.pdf" \
  -F "files=@Bidder_Alpha_Tech_Portfolio.pdf"
```

---

## 🛑 Troubleshooting

| Issue                                    | Solution                                                                                       |
|------------------------------------------|------------------------------------------------------------------------------------------------|
| `Qdrant database is locked`              | Another instance is running. Stop it or the system will auto-fallback to in-memory mode.       |
| `NVIDIA_NIM_API_KEY not set`             | Add your key to the `.env` file. The VLM engine will fall back to local mode without it.       |
| Models take long to load at startup      | First run downloads ~400MB of models (GLiNER + Sentence-Transformers). Subsequent runs are cached. |
| `ModuleNotFoundError`                    | Ensure your virtual environment is activated and all dependencies are installed.                |
| Port 8001 already in use                 | Another process is using port 8001. Stop it or modify the port in `api.py` line 126.           |

---

## 📜 License

Built for the **AI for Bharat Hackathon** — Agentic Document Extraction Platform.
