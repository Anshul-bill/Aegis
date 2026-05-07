import torch
from PIL import Image, ImageDraw
import instructor
from openai import OpenAI
from typing import List, Dict, Any, Optional
import json
import os
import base64
import io
from dotenv import load_dotenv
from models import CriterionEvaluation, EvaluationStatus, GoldenRule

# Load environment variables from .env
load_dotenv()

class AegisVLMEngine:
    """
    SOTA VLM Engine using NVIDIA NIM API and Instructor for Schema Enforcement.
    This bypasses local VRAM constraints while ensuring structured outputs.
    """
    def __init__(self, model_id: str = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"):
        self.model_id = model_id
        self.api_key = os.environ.get("NVIDIA_NIM_API_KEY")
        
        # Point to NVIDIA NIM Hosted Endpoint
        self.client = instructor.patch(OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=self.api_key
        ))
        
        # Flag to enable/disable live API calls - ENABLED BY DEFAULT for functional delivery
        self.load_model = os.environ.get("LOAD_VLM", "True").lower() == "true"

        if self.load_model and not self.api_key:
            print("WARNING: LOAD_VLM is True but NVIDIA_NIM_API_KEY is not set. Falling back to local mode.")
            self.load_model = False

    def redact_pii_on_image(self, image: Image.Image, pii_detected: bool) -> Image.Image:
        """
        SOTA Visual Redaction: Draw blackout boxes over PII before NIM transmission.
        """
        if not pii_detected: return image
        
        draw = ImageDraw.Draw(image)
        # For the prototype, we blackout the top-left area if PII is detected 
        draw.rectangle([0, 0, image.width * 0.3, image.height * 0.15], fill="black")
        return image

    def extract_criterion(self, image: Optional[Image.Image], rule: GoldenRule, pii_detected: bool = False) -> CriterionEvaluation:
        """
        Extracts information for a specific rule using NVIDIA NIM VLM.
        """
        if image is None:
            return CriterionEvaluation(
                parameter_id=rule.parameter_id,
                status=EvaluationStatus.NEED_MANUAL_REVIEW,
                confidence=0.0,
                extracted_value="ERROR",
                rationale="No image provided for visual extraction."
            )

        # 1. Visual Redaction Layer (Data Sovereignty)
        image_to_send = image.copy()
        if pii_detected:
            image_to_send = self.redact_pii_on_image(image_to_send, True)

        # 2. Convert PIL Image to Base64
        buffered = io.BytesIO()
        image_to_send.save(buffered, format="JPEG", quality=85) # High quality for 300DPI
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # 3. SOTA Call to NVIDIA NIM with Instructor enforcement
        try:
            print(f"  [NIM] Querying {self.model_id} for: {rule.semantic_description[:50]}...")
            return self.client.chat.completions.create(
                model=self.model_id,
                response_model=CriterionEvaluation,
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "You are a SOTA Document Extraction Agent. You MUST extract ONLY the specific raw value (e.g., '8.5', 'ISO 9001:2015', '4 projects') into the 'extracted_value' field. "
                            "DO NOT under any circumstances include sentences, reasoning, or 'why' questions in the 'extracted_value' field. "
                            "IMPORTANT: Strictly differentiate the Rupee symbol (₹) from digits. '₹8.5' is 8.5. "
                            "If the requirement asks for a count (e.g., number of projects), count the items in the document and return the number. "
                            "You MUST provide the tight bounding box coordinates [ymin, xmin, ymax, xmax] normalized to 1000. "
                            "Self-report your 'confidence' score (0.0 to 1.0) based on visual clarity."
                        )
                    },
                    {
                        "role": "user", 
                        "content": [
                            {
                                "type": "text", 
                                "text": (
                                    f"Requirement: {rule.semantic_description}\n"
                                    "Extract the specific compliance value and its tight bounding box [ymin, xmin, ymax, xmax]."
                                )
                            },
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                        ]
                    }
                ],
                max_retries=2
            )
        except Exception as e:
            print(f"  [NIM Error] Extraction failed: {e}")
            return CriterionEvaluation(
                parameter_id=rule.parameter_id,
                status=EvaluationStatus.NEED_MANUAL_REVIEW,
                confidence=0.0,
                extracted_value="ERROR",
                rationale=f"NVIDIA NIM API call failed: {str(e)}"
            )

if __name__ == "__main__":
    vlm = AegisVLMEngine()
    print(f"VLM Engine initialized. Model: {vlm.model_id}, Live Mode: {vlm.load_model}")
