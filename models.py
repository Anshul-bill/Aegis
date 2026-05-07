from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class ClassificationType(str, Enum):
    TECHNICAL_MANDATORY = "TECHNICAL_MANDATORY"
    FINANCIAL_MANDATORY = "FINANCIAL_MANDATORY"
    OPTIONAL_PREFERENCE = "OPTIONAL_PREFERENCE"

class GoldenRule(BaseModel):
    parameter_id: str = Field(..., description="Unique hash of the requirement")
    classification_type: ClassificationType = Field(..., description="The category of the requirement")
    semantic_description: str = Field(..., description="Detailed textual description of the requirement")
    documentary_evidence: List[str] = Field(default_factory=list, description="List of documents needed to satisfy this")
    regulatory_override: bool = Field(default=False, description="Flag for MSE/Make-in-India overrides")

class EvaluationStatus(str, Enum):
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    NEED_MANUAL_REVIEW = "NEED_MANUAL_REVIEW"

class CriterionEvaluation(BaseModel):
    parameter_id: str
    status: EvaluationStatus
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0 or 0 to 100)")
    extracted_value: str
    evidence_bbox: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0], description="[ymin, xmin, ymax, xmax] normalized 0-1000")
    rationale: str = Field(..., description="Explanation of why this decision was made")
    source_file: Optional[str] = Field(None, description="The name of the document where evidence was found")
    rule_description: Optional[str] = Field(None, description="The original requirement description")

    @property
    def normalized_confidence(self) -> float:
        if self.confidence > 1.0:
            return self.confidence / 100.0
        return self.confidence

class ProjectState(BaseModel):
    input_directory: str
    golden_rules: List[GoldenRule] = []
    evaluations: List[CriterionEvaluation] = []
    overall_confidence: float = 0.0
