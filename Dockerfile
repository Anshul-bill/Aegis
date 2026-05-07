FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_lg

# Copy project files
COPY api.py orchestrator.py models.py sanitizer.py deconstruction_agent.py \
     vlm_engine.py decision_engine.py benchmarker.py rag_engine.py \
     data_ingestion.py tender_agent.py block-secrets.py bulk_tester.py \
     golden_rule_set.json audit_ledger.json ./

# Copy static frontend
COPY static/ static/

# Copy sample PDFs
COPY *.pdf ./

# HF Spaces requires port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["python", "api.py"]
