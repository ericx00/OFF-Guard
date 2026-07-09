from pydantic import BaseModel
from typing import List


class Case(BaseModel):
    document: str
    context: str


class RiskReport(BaseModel):
    rating: str          # "low", "medium", "high"
    suggestion: str      # what to do next
    reasons: List[str]   # bullet-point justifications
