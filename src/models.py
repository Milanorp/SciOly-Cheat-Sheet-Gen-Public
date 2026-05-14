from typing import List, Optional, Dict
from pydantic import BaseModel, Field

# ==========================================
# PHASE 0 & 1: Planning Models
# ==========================================

class ExtractedConcepts(BaseModel):
    """Output from Phase 0: Test Cruncher."""
    core_concepts: List[str] = Field(description="List of technical concepts tested.")
    test_traps: List[str] = Field(description="Common tricky wordings or unit mistakes.")

class BlueprintSection(BaseModel):
    """A single section in the architect's plan."""
    section_name: str
    micro_topics: List[str]

class CheatSheetBlueprint(BaseModel):
    """The master plan generated in Phase 1."""
    event_analysis: str
    sections: List[BlueprintSection]

# ==========================================
# PHASE 2: Research Models
# ==========================================

class ResearchNote(BaseModel):
    """A single piece of research for one target topic."""
    original_target: str
    expanded_requirements: str
    content: str

class SectionResearch(BaseModel):
    """All research for a specific section."""
    section_name: str
    notes: List[ResearchNote]

# ==========================================
# PHASE 2.5: Audit Models
# ==========================================

class AuditReport(BaseModel):
    """The auditor's evaluation of a research note."""
    grade: str = Field(description="A, B, C, D, or F")
    feedback: str = Field(description="Actionable correction memo")
    is_pass: bool

# ==========================================
# SYSTEM: Pipeline State
# ==========================================

class PipelineState(BaseModel):
    """The persistent state for resume/checkpointing."""
    current_phase: float
    event_name: Optional[str] = None
    frequency_data: Optional[Dict] = None
    blueprint: Optional[Dict] = None # Map section -> list of topics
    research_notes: Optional[Dict] = None # Map section -> list of ResearchNote
    cache_info: Optional[Dict] = None
    failed_topics: List[str] = []
    retry_count: int = 0
