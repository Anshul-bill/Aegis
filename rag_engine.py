import os
import hashlib
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np

class AegisRAGEngine:
    def __init__(self, collection_name: str = "tender_rules", preloaded_model=None):
        self.collection_name = collection_name
        # Using a fast, local vector DB (file-based)
        try:
            self.client = QdrantClient(path="./qdrant_data")
        except RuntimeError as e:
            if "already accessed by another instance" in str(e):
                print("WARNING: Qdrant database is locked. Falling back to in-memory instance.")
                self.client = QdrantClient(location=":memory:")
            else:
                raise e
        # Memory Optimization: Use preloaded model if available
        self.model = preloaded_model or SentenceTransformer('all-MiniLM-L6-v2')
        
        # SOTA: ColPali / Visual RAG infrastructure 
        # Multi-vector support is required for document image embeddings
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "text": models.VectorParams(size=384, distance=models.Distance.COSINE),
                    "visual": models.VectorParams(size=128, distance=models.Distance.COSINE, multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM))
                }
            )

    def close(self):
        """Explicitly release the Qdrant storage lock."""
        try:
            if hasattr(self, 'client'):
                self.client.close()
        except (ImportError, TypeError):
            # Suppress errors during interpreter shutdown
            pass

    def __del__(self):
        self.close()

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        # Semantic chunking placeholder - simple windowing for now
        # In Phase 2, this should be improved to respect section boundaries
        chunks = []
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def index_tender(self, text: str, metadata: Dict[str, Any] = None):
        chunks = self.chunk_text(text)
        if not chunks:
            print("WARNING: No text chunks found for indexing.")
            self.bm25 = None
            self.chunks = []
            return
            
        embeddings = self.model.encode(chunks)
        
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Use MD5 hash as a deterministic ID for chunks to avoid duplicates
            point_id = hashlib.md5(chunk.encode()).hexdigest()
            points.append(models.PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "text": chunk,
                    "metadata": metadata or {}
                }
            ))
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        # Initialize BM25 for sparse search
        self.bm25 = BM25Okapi([c.split() for c in chunks])
        self.chunks = chunks

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # 1. Dense Search
        query_vector = self.model.encode(query).tolist()
        dense_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k * 2
        ).points
        
        # 2. Sparse Search (Manual BM25 merging as fastembed failed)
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Merge Results (Reciprocal Rank Fusion or simple score addition)
        # Here we just use the dense results and augment with BM25 if needed
        
        combined_results = []
        for res in dense_results:
            combined_results.append({
                "text": res.payload["text"],
                "score": res.score,
                "type": "dense"
            })
            
        # Add top BM25 if not already in dense (basic overlap check)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
        for idx in top_bm25_indices:
            chunk = self.chunks[idx]
            if not any(r["text"] == chunk for r in combined_results):
                combined_results.append({
                    "text": chunk,
                    "score": float(bm25_scores[idx]),
                    "type": "sparse"
                })
                
        return sorted(combined_results, key=lambda x: x["score"], reverse=True)[:top_k]

if __name__ == "__main__":
    engine = AegisRAGEngine()
    sample_tender = """
    TENDER REQUIREMENT:
    The bidder must have a minimum annual turnover of 5 Crore INR in the last 3 fiscal years.
    Mandatory Certification: ISO 9001:2015 is required.
    Equipment: 3.8 GHz or higher processor, 5 KVA ONLINE UPS.
    """
    engine.index_tender(sample_tender)
    
    query = "What is the mandatory turnover and UPS requirement?"
    results = engine.hybrid_search(query)
    
    print(f"Query: {query}")
    for res in results:
        print(f"[{res['type']}] Score: {res['score']:.4f}\nContent: {res['text']}\n")
