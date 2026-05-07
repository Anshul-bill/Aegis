import fitz
import pandas as pd
import os
from typing import List, Dict, Any

class AegisIngestionEngine:
    """
    SOTA Ingestion Engine: Handes PDF, Excel (BOQ), and text files generically.
    Ensures structural integrity is preserved for RAG.
    """
    def extract_text(self, file_path: str) -> str:
        # Prevent hanging on massive files (Limit: 200MB)
        if os.path.getsize(file_path) > 200 * 1024 * 1024:
            print(f"  [Skipping] File too large: {os.path.basename(file_path)}")
            return ""

        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".pdf":
            return self._parse_pdf(file_path)
        elif ext in [".xls", ".xlsx"]:
            return self._parse_excel(file_path)
        elif ext == ".csv":
            return self._parse_csv(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()

    def _parse_pdf(self, file_path: str) -> str:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    def _parse_excel(self, file_path: str) -> str:
        # For BOQ files, we often want to preserve the table structure as text
        try:
            df = pd.read_excel(file_path)
            return df.to_string() # Simple string representation of the table
        except Exception as e:
            return f"Error parsing Excel {file_path}: {str(e)}"

    def _parse_csv(self, file_path: str) -> str:
        try:
            df = pd.read_csv(file_path)
            return df.to_string()
        except Exception as e:
            return f"Error parsing CSV {file_path}: {str(e)}"

if __name__ == "__main__":
    engine = AegisIngestionEngine()
    # Test on the BOQ file
    boq_text = engine.extract_text("work_953482/BOQ_953482.xls")
    print("--- BOQ Extraction Test ---")
    print(boq_text[:500])
