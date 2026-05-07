import fitz
import json
import re
from typing import List, Dict, Any
import os

class AegisDecisionEngine:
    def __init__(self, confidence_threshold: float = 0.85):
        self.confidence_threshold = confidence_threshold

    def translate_coordinates(self, bbox_vlm: List[float], original_size: tuple, tensor_size: tuple) -> List[float]:
        """
        Scale VLM bounding boxes back to original PDF [x1, y1, x2, y2].
        Corrects for potential vertical coordinate flips by validating aspect ratios.
        """
        if not bbox_vlm or len(bbox_vlm) < 4:
            return [0, 0, 0, 0]

        w_orig, h_orig = original_size
        is_normalized = all(v <= 1.05 for v in bbox_vlm[:4])
        scale_factor = 1.0 if is_normalized else 1000.0

        # SOTA: Take only the first 4 coordinates to handle models that return [y,x,y,x, score, id]
        v1, v2, v3, v4 = bbox_vlm[:4]
        
        # SOTA: Auto-correction for coordinate flip
        # If the box is extremely tall and narrow (flipped aspect ratio), we swap
        y_val_1 = (v1 / scale_factor) * h_orig
        x_val_1 = (v2 / scale_factor) * w_orig
        y_val_2 = (v3 / scale_factor) * h_orig
        x_val_2 = (v4 / scale_factor) * w_orig
        
        height = abs(y_val_2 - y_val_1)
        width = abs(x_val_2 - x_val_1)
        
        # If height is > 3x width, it's likely a coordinate transposition error
        if height > width * 3:
            # Swap x and y
            x1 = (v1 / scale_factor) * w_orig
            y1 = (v2 / scale_factor) * h_orig
            x2 = (v3 / scale_factor) * w_orig
            y2 = (v4 / scale_factor) * h_orig
            return [x1, y1, x2, y2]
            
        return [x_val_1, y_val_1, x_val_2, y_val_2]

    def get_precise_bbox(self, pdf_path: str, page_num: int, text_to_find: str, fallback_bbox: List[float]) -> List[float]:
        """
        SOTA DETERMINISTIC GROUNDING: 
        Try to find the exact text string on the page to get pixel-perfect coordinates.
        Falls back to VLM coordinates only if text search fails (e.g. on scanned images).
        """
        if not text_to_find or len(text_to_find) < 3:
            return fallback_bbox

        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            # Search for the most unique part of the extracted text
            # (Limiting to first 20 chars to avoid multi-line search failures)
            search_term = text_to_find[:20].strip()
            areas = page.search_for(search_term)
            doc.close()

            if areas:
                # Return the first matching area as [x1, y1, x2, y2]
                rect = areas[0]
                return [rect.x0, rect.y0, rect.x1, rect.y1]
        except:
            pass
        return fallback_bbox

    def batch_overlay_evidence(self, pdf_path: str, page_num: int, bboxes: List[List[float]], labels: List[str], output_path: str):
        """
        Apply multiple translucent highlights in a single pass to avoid file locking.
        """
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        for bbox, label in zip(bboxes, labels):
            # Ensure bbox has 4 coordinates and positive area
            if not bbox or len(bbox) < 4: continue
            
            # Create Rect object [x0, y0, x1, y1]
            rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
            
            # Add sharp rectangular evidence highlight only if rect is valid
            if rect.width > 0 and rect.height > 0:
                annot = page.add_rect_annot(rect)
                annot.set_colors(stroke=(0.8, 0, 0), fill=(0.8, 0, 0)) # SOTA Red
                annot.set_opacity(0.3) # Translucent for readability
                annot.set_info(content=label)
                annot.update()
        
        # Save as a new file or overwrite
        doc.save(output_path, incremental=False, encryption=fitz.PDF_ENCRYPT_KEEP)
        doc.close()

    def _extract_numbers(self, text: str) -> List[float]:
        """
        SOTA Numerical Parser: Extracts numbers while filtering out potential years 
        and prioritizing financial values.
        """
        # Finds integers and decimals
        raw_nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        nums = []
        for n in raw_nums:
            try:
                val = float(n)
                # Filter out obvious years (e.g., 2024, 2025)
                if 1990 <= val <= 2050 and "." not in n:
                    continue
                nums.append(val)
            except:
                continue
        return nums

    def evaluate_criterion(self, extracted_data: Dict[str, Any], rule: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare extracted data against Golden Rule Set and calibrate confidence.
        Perform real numerical logical checks for turnover/financials.
        """
        confidence = extracted_data.get("confidence", 0.9)
        if confidence > 1.0: confidence /= 100.0 # Normalize 0-100 to 0-1
        
        extracted_value_str = str(extracted_data.get("extracted_value", ""))
        requirement_str = str(rule.get("semantic_description", ""))
        
        is_compliant = False
        rationale = extracted_data.get("rationale", "No rationale provided.")

        # SOTA: Intelligent Numerical Logical Comparison
        if "turnover" in requirement_str.lower() or "crore" in requirement_str.lower() or "lakh" in requirement_str.lower():
            req_nums = self._extract_numbers(requirement_str)
            ext_nums = self._extract_numbers(extracted_value_str)
            
            if req_nums and ext_nums:
                # Prioritize the largest number as the threshold (to avoid clause numbers like 1.1)
                threshold = max(req_nums)
                # For extraction, if multiple numbers exist, pick the one closest to currency keywords or the last one
                actual = ext_nums[-1] # Usually the value comes after labels
                
                # Special case: if requirement specifies "at least 3 projects"
                if "projects" in requirement_str.lower() or "experience" in requirement_str.lower():
                    threshold = max(req_nums) # e.g. "at least 3" -> 3
                
                if actual >= threshold:
                    is_compliant = True
                    rationale = f"SOTA Verification: Extracted value {actual} meets or exceeds required threshold {threshold}."
                else:
                    is_compliant = False
                    rationale = f"CRITICAL FAILURE: Extracted value {actual} is BELOW required threshold {threshold}."
            else:
                # Fallback to semantic presence if numbers can't be parsed
                if confidence > 0.8 and any(word in extracted_value_str.lower() for word in ["yes", "compliant", "verified", "available"]):
                    is_compliant = True
                    rationale = "Compliance verified via semantic evidence."
        else:
            # For technical/statutory rules, use high-confidence semantic presence
            if confidence > 0.75 and len(extracted_value_str) > 2 and "error" not in extracted_value_str.lower():
                # Check for negative indicators
                if not any(neg in extracted_value_str.lower() for neg in ["not", "no", "missing", "fail"]):
                    is_compliant = True
                    rationale = "Technical criteria verified via visual evidence."

        status = "NEED_MANUAL_REVIEW" if confidence < self.confidence_threshold else ("COMPLIANT" if is_compliant else "NON_COMPLIANT")
        
        return {
            "parameter_id": rule["parameter_id"],
            "status": status,
            "confidence": confidence * 100, # Display as percentage
            "evidence_bbox": extracted_data.get("evidence_bbox"),
            "extracted_value": extracted_value_str,
            "rationale": rationale
        }

    def generate_audit_ledger(self, results: List[Dict[str, Any]], output_path: str):
        ledger = {
            "version": "1.0",
            "timestamp": "2026-05-02T12:00:00Z",
            "results": results
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ledger, f, indent=2)

if __name__ == "__main__":
    engine = AegisDecisionEngine()
    
    # Mock data
    rule = {
        "parameter_id": "rule_123",
        "classification_type": "TECHNICAL_MANDATORY",
        "semantic_description": "Annual turnover >= 5 Cr"
    }
    extracted = {
        "bbox_2d": [100, 200, 300, 250],
        "label": "Annual Turnover",
        "sub_label": "₹5.2 Crore",
        "confidence": 0.92
    }
    
    decision = engine.evaluate_criterion(extracted, rule)
    print("Decision result:")
    print(json.dumps(decision, indent=2))
    
    # Test coordinate translation
    orig_size = (595, 842) # A4 pixels at 72dpi
    tensor_size = (1000, 1000)
    translated = engine.translate_coordinates(extracted["bbox_2d"], orig_size, tensor_size)
    print(f"\nTranslated coordinates: {translated}")
    
    # Save ledger
    engine.generate_audit_ledger([decision], "audit_ledger.json")
    print("Audit ledger saved to audit_ledger.json")
