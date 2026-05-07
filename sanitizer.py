import re
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern, RecognizerResult, LocalRecognizer, EntityRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from gliner import GLiNER
import random
import string
from typing import List

# Verhoeff algorithm for Aadhaar checksum validation (omitted for brevity in display, but included in implementation)
VERHOEFF_D = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1], [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
VERHOEFF_P = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1], [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]
VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]

def validate_verhoeff(number):
    c = 0
    ll = [int(x) for x in str(number)]
    for i, j in enumerate(ll[::-1]):
        c = VERHOEFF_D[c][VERHOEFF_P[i % 8][j]]
    return c == 0

class GLiNERRecognizer(EntityRecognizer):
    """
    SOTA: Using GLiNER for Zero-Shot Named Entity Recognition.
    This replaces brittle regex for complex entities like addresses or custom IDs.
    """
    def __init__(self, model_name: str = "urchade/gliner_small-v2.1", labels: List[str] = None, preloaded_model=None):
        super().__init__(supported_entities=labels or ["PERSON", "ADDRESS", "IN_PAN", "IN_GSTIN"])
        # Memory Optimization: Use preloaded model if available
        self.model = preloaded_model or GLiNER.from_pretrained(model_name)
        self.labels = labels or ["person", "address", "indian pan number", "gst number"]

    def load(self):
        pass # Already loaded in __init__

    def analyze(self, text, entities, nlp_artifacts=None):
        results = []
        gliner_results = self.model.predict_entities(text, self.labels, threshold=0.4)
        
        # Map GLiNER labels to Presidio entity types
        mapping = {
            "person": "PERSON",
            "address": "ADDRESS",
            "indian pan number": "IN_PAN",
            "gst number": "IN_GSTIN"
        }
        
        for res in gliner_results:
            entity_type = mapping.get(res["label"], res["label"].upper())
            results.append(
                RecognizerResult(
                    entity_type=entity_type,
                    start=res["start"],
                    end=res["end"],
                    score=res["score"]
                )
            )
        return results

# Standard Pattern-based Recognizer for Aadhaar (requires checksum validation)
class AadhaarRecognizer(LocalRecognizer):
    def __init__(self):
        patterns = [Pattern(name="aadhaar_pattern", regex=r"\b[2-9][0-9]{3}[ -]?[0-9]{4}[ -]?[0-9]{4}\b", score=0.5)]
        super().__init__(supported_entities=["IN_AADHAAR"])
        self.patterns = patterns
    def analyze(self, text, entities, nlp_artifacts=None):
        results = []
        for match in re.finditer(self.patterns[0].regex, text):
            if len(re.sub(r"[ -]", "", match.group())) == 12 and validate_verhoeff(re.sub(r"[ -]", "", match.group())):
                results.append(RecognizerResult(entity_type="IN_AADHAAR", start=match.start(), end=match.end(), score=0.85))
        return results

# Helper functions for synthetic data (Length preserving)
def generate_fake_str(old_value):
    result = []
    for char in old_value:
        if char.isupper(): result.append(random.choice(string.ascii_uppercase))
        elif char.islower(): result.append(random.choice(string.ascii_lowercase))
        elif char.isdigit(): result.append(random.choice(string.digits))
        else: result.append(char)
    return "".join(result)

class AegisSOTASanitizer:
    def __init__(self, preloaded_gliner=None):
        self.analyzer = AnalyzerEngine(default_score_threshold=0.4)
        # Register GLiNER as the primary SOTA engine
        self.analyzer.registry.add_recognizer(GLiNERRecognizer(preloaded_model=preloaded_gliner))
        self.analyzer.registry.add_recognizer(AadhaarRecognizer())
        self.anonymizer = AnonymizerEngine()

    def sanitize(self, text: str):
        results = self.analyzer.analyze(text=text, language="en", 
                                        entities=["PERSON", "ADDRESS", "IN_PAN", "IN_GSTIN", "IN_AADHAAR", "PHONE_NUMBER", "EMAIL_ADDRESS"])
        
        operators = {
            "PERSON": OperatorConfig("custom", {"lambda": generate_fake_str}),
            "ADDRESS": OperatorConfig("custom", {"lambda": generate_fake_str}),
            "IN_PAN": OperatorConfig("custom", {"lambda": generate_fake_str}),
            "IN_GSTIN": OperatorConfig("custom", {"lambda": generate_fake_str}),
            "IN_AADHAAR": OperatorConfig("custom", {"lambda": generate_fake_str}),
            "PHONE_NUMBER": OperatorConfig("mask", {"chars_to_mask": 10, "masking_char": "X", "from_end": True}),
            "EMAIL_ADDRESS": OperatorConfig("custom", {"lambda": generate_fake_str})
        }
        
        anonymized_result = self.anonymizer.anonymize(text=text, analyzer_results=results, operators=operators)
        return anonymized_result.text

    def get_pii_entities(self, text: str):
        """Returns the list of identified PII entities for visual redaction logic."""
        return self.analyzer.analyze(text=text, language="en", 
                                    entities=["PERSON", "ADDRESS", "IN_PAN", "IN_GSTIN", "IN_AADHAAR", "PHONE_NUMBER", "EMAIL_ADDRESS"])

if __name__ == "__main__":
    sanitizer = AegisSOTASanitizer()
    sample = """
    Contact: Amit Sharma, 123, MG Road, Bangalore. 
    PAN: ABCDE1234F, GST: 27ABCDE1234F1Z5.
    Email: amit.s@domain.com
    """
    print("Original:\n", sample)
    print("\nSOTA Sanitized:\n", sanitizer.sanitize(sample))
