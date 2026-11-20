# 📑 Regulatory & Tax Policy RAG Assistant

An end-to-end Retrieval-Augmented Generation (RAG) system designed to answer complex policy and tax queries strictly from official PDF documentation, complete with inline page citations and verifiable source snippets.

## 🎯 Problem Statement
General-purpose LLMs often hallucinate specific statutory thresholds, exemptions, and procedural rules when asked about financial or regulatory compliance. This project solves that by decoupling knowledge retrieval from language generation—forcing the model to answer exclusively from indexed official PDFs and cite exact page numbers.

## 🏗️ System Architecture

```text
[Official PDFs] 
       │
       ▼ (PyPDFDirectoryLoader)
[Raw Document Pages + Metadata]
       │
       ▼ (RecursiveCharacterTextSplitter: 1000 chars, 200 overlap)
[Contextual Text Chunks]
       │
       ▼ (BAAI/bge-small-en-v1.5)
[Local ChromaDB Vector Store]
       │
       ◄── User Query ──► [Top-K Similarity Search]
                                  │
                                  ▼
                    [Grounded Prompt + Context]
                                  │
                                  ▼ (Llama-3.3-70B @ Temp 0.0)
                    [Cited Answer + Evidence UI]
```

## 🛠️ Tech Stack & Design Decisions
* **Embedding Model (`BAAI/bge-small-en-v1.5`):** Selected for strong retrieval accuracy on MTEB benchmarks while running fast locally on CPU without API costs.
* **Vector Database (`ChromaDB`):** Provides local, persistent disk storage with metadata filtering (`source` filename and `page` index).
* **Chunking Strategy (`1000` size / `200` overlap):** Preserves multi-sentence regulatory rules and bulleted conditions without truncating boundary clauses.
* **Inference Engine (`Llama-3.3-70b-versatile` via Groq):** Configured at `temperature=0.0` with a strict negative-constraint system prompt so the model refuses to answer when context is missing.
* **Frontend (`Streamlit`):** Displays conversational history alongside expandable evidence drawers showing relevance scores and raw retrieved text.

## 🚀 Quickstart

1. **Clone & Install Dependencies**
   ```bash
   git clone <your-repo-url>
   cd rag-policy-assistant
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key
   ```

3. **Ingest Documents**
   Place your PDF documents inside the `data/` folder and run:
   ```bash
   python ingest.py
   ```

4. **Launch Application**
   ```bash
   streamlit run app.py
   ```

## 📈 Future Roadmap
* **Hybrid Search:** Combine dense vector search with sparse keyword matching (`BM25`) to better capture exact section numbers (e.g., *Section 80C*, *Section 194J*).
* **Cross-Encoder Reranking:** Add a `bge-reranker` step to re-score top-10 retrieved chunks down to the top-3 before LLM generation.
* **Automated Evaluation:** Integrate `RAGAS` to benchmark Faithfulness and Context Precision across a golden test set of 30 Q&A pairs.