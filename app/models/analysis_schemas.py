# app/models/analysis_schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional

class QuestionAnalysis(BaseModel):
    """Schema for individual question-answer analysis from transcript"""
    timestamp: str = Field(..., description="Timestamp of the question")
    question: str = Field(..., description="Interview question asked")
    answer: str = Field(..., description="Candidate's answer")
    analysis: str = Field(..., description="Detailed analysis of the response")
    relevance_to_resume: str = Field(..., description="How question relates to resume")
    answer_quality: str = Field(..., description="Quality rating (Poor/Fair/Good/Excellent)")
    technical_accuracy: str = Field(..., description="Technical correctness assessment")
    score: float = Field(..., ge=0, le=10, description="Score out of 10")

    class Config:
        # Keep example if useful, or remove
        pass

class OverallAnalysis(BaseModel):
    """Schema for transcript analysis"""
    user_id: Optional[str] = Field(None, description="Candidate user ID") # Made optional if not always present
    session_id: str = Field(..., description="Interview session ID")
    interview_date: Optional[str] = Field(None, description="Interview date") # Made optional
    questions_analyzed: List[QuestionAnalysis] = Field(..., description="List of Q&A analyses")
    overall_summary: str = Field(..., description="Overall summary")
    strengths: List[str] = Field(..., description="Candidate's strengths")
    weaknesses: List[str] = Field(..., description="Areas for improvement")
    resume_alignment_score: float = Field(..., ge=0, le=10)
    communication_score: float = Field(..., ge=0, le=10)
    technical_knowledge_score: float = Field(..., ge=0, le=10)
    overall_score: float = Field(..., ge=0, le=10)
    recommendations: List[str] = Field(...)

    class Config:
         # Keep example if useful, or remove
        pass

# --- NEW: Schema for Combined Analysis ---
class CombinedAnalysisReport(BaseModel):
    """Schema for the final combined analysis report"""
    session_id: str
    candidate_id: str
    interview_date: Optional[str]

    # Scores
    behavioral_score: Optional[float] = Field(None, ge=0, le=10, description="Overall score from behavioral analysis (e.g., based on metrics)")
    transcript_overall_score: float = Field(..., ge=0, le=10, description="Overall score from transcript analysis")
    transcript_communication_score: float = Field(..., ge=0, le=10)
    transcript_technical_score: float = Field(..., ge=0, le=10)
    final_weighted_score: float = Field(..., ge=0, le=10, description="Combined weighted score")

    # Summaries & Lists
    behavioral_summary: Optional[str] = Field(None, description="Summary from behavioral analysis")
    transcript_summary: str
    combined_strengths: List[str]
    combined_weaknesses: List[str]
    combined_recommendations: List[str]

    # Raw Data (Optional, could be large)
    # behavioral_analysis_raw: Optional[Dict] = None # Example: Full behavioral result
    # transcript_analysis_raw: OverallAnalysis # Full transcript result