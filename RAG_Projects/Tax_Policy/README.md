# Tax Policy RAG Assistant

## Setup

From the project folder, create and activate a virtual environment, then install
the declared dependencies:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Set `GROQ_API_KEY` in a `.env` file in the project folder. The app defaults to
Groq's `openai/gpt-oss-120b` model. To choose another model available to your
Groq account, optionally set `GROQ_MODEL` in the same `.env` file. Add policy
PDF files to `Data\` (the repository currently includes `Data\RBI_01.pdf`).

## Run

Build the local ChromaDB index, then launch the Streamlit app from the project
folder:

```powershell
python code\ingest.py
python -m streamlit run code\streamlit_app.py
```