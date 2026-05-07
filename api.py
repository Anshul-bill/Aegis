import os
import shutil
import tempfile
from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from sanitizer import AegisSOTASanitizer
from deconstruction_agent import AegisDeconstructionAgent
from vlm_engine import AegisVLMEngine
from decision_engine import AegisDecisionEngine
from benchmarker import AegisBenchmarker
from rag_engine import AegisRAGEngine
from orchestrator import AegisOrchestrator

from gliner import GLiNER
from sentence_transformers import SentenceTransformer

# --- [SOTA Memory Optimization] Global Singletons ---
# Load models once at startup to prevent OOM errors on 7.5GB RAM
print("--- [SOTA] Loading Global AI Models ---")
_gliner_model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
_st_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Engines with Shared Models
sanitizer = AegisSOTASanitizer(preloaded_gliner=_gliner_model)
rag_engine = AegisRAGEngine(preloaded_model=_st_model)
vlm_engine = AegisVLMEngine()
decision_engine = AegisDecisionEngine()
decon_agent = AegisDeconstructionAgent(preloaded_rag=rag_engine)

# Initialize Orchestrator with Singletons
orchestrator = AegisOrchestrator(
    preloaded_sanitizer=sanitizer,
    preloaded_decon=decon_agent,
    preloaded_vlm=vlm_engine,
    preloaded_decision=decision_engine
)

app = FastAPI(title="Project Aegis - ADE Pipeline API")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure results directory exists for grounded PDFs
os.makedirs("results", exist_ok=True)

# Mount static directory for the React frontend
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/api/analyze")
async def analyze_documents(files: List[UploadFile] = File(...)):
    """
    Accepts uploaded tender documents (PDFs, Excel BOQs), saves them,
    runs the AegisOrchestrator, and returns the evaluation metrics.
    """
    # Create a temporary directory for this evaluation run
    temp_dir = tempfile.mkdtemp(prefix="aegis_")
    
    try:
        # Save uploaded files to the temp directory
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                
        # Use the global orchestrator instance
        print(f"Starting analysis on {len(files)} files in {temp_dir}")
        result = orchestrator.run(temp_dir)
        
        # Convert Pydantic objects to dicts for JSON response
        golden_rules = [r.model_dump() for r in result.get("golden_rules", [])]
        evaluations = [e.model_dump() for e in result.get("evaluations", [])]
        
        # Check if grounded PDF was generated
        grounded_pdf_url = None
        bidder_path = result.get("bidder_pdf_path")
        if bidder_path:
            filename = os.path.basename(bidder_path)
            if os.path.exists(os.path.join("results", f"grounded_{filename}")):
                grounded_pdf_url = f"/api/results/grounded_{filename}"
                
        return JSONResponse({
            "status": result.get("status"),
            "metrics": result.get("metrics", {}),
            "golden_rules": golden_rules,
            "evaluations": evaluations,
            "grounded_pdf_url": grounded_pdf_url
        })
        
    except Exception as e:
        print(f"API Error: {str(e)}")
        return JSONResponse({"status": "ERROR", "error": str(e)}, status_code=500)
    finally:
        # Cleanup temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.get("/api/results/{filename}")
async def get_result_pdf(filename: str):
    """Serve the visually grounded PDFs."""
    file_path = os.path.join("results", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/pdf")
    return JSONResponse({"error": "File not found"}, status_code=404)

@app.get("/")
async def root():
    """Serve the main React dashboard."""
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"error": "Frontend not found. Please create static/index.html"}, status_code=404)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    print(f"Starting Project Aegis API on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
