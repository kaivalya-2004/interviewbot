# app/services/combined_analyzer.py
import logging
from typing import Dict, Any, Optional, List
# --- Import schemas from the new location ---
from app.models.analysis_schemas import OverallAnalysis, CombinedAnalysisReport
# --- END Import ---

logger = logging.getLogger(__name__)

# Define weights for combining scores (adjust as needed)
WEIGHT_BEHAVIORAL = 0.4
WEIGHT_TRANSCRIPT = 0.6

def combine_analyses(
    behavioral_result: Optional[Dict], # Result from your existing BehavioralAnalyzer
    transcript_result: Optional[OverallAnalysis], # Result from TranscriptAnalyzer
    session_id: str,
    candidate_id: str,
    # interview_date: Optional[str] # Date now comes from transcript_result
) -> CombinedAnalysisReport:
    """
    Combines results from behavioral and transcript analysis into a final report.

    Args:
        behavioral_result: Dict containing behavioral analysis data.
                           Expected keys (adjust as needed): 'status', 'executive_summary', 'metrics'.
        transcript_result: OverallAnalysis Pydantic object from transcript analysis (can be None).
        session_id: The interview session ID.
        candidate_id: The candidate ID.

    Returns:
        A CombinedAnalysisReport Pydantic object.
    Raises:
         ValueError: If input data is invalid or report cannot be constructed.
    """
    logger.info(f"Combining analyses for session: {session_id}")

    # --- Extract data from Behavioral Analysis ---
    behavioral_summary = None
    behavioral_score = None
    behavioral_strengths = [] # Placeholder - Add if behavioral analysis provides these
    behavioral_weaknesses = [] # Placeholder

    if behavioral_result and behavioral_result.get('status') == 'success':
        behavioral_summary = behavioral_result.get('executive_summary', behavioral_result.get('summary', None))
        metrics = behavioral_result.get('metrics', {})
        # Prioritize an explicit overall score if present
        if 'overall_score' in metrics:
             behavioral_score = metrics['overall_score']
        elif 'overall_sentiment_score' in metrics: # Example alternative metric
             behavioral_score = metrics['overall_sentiment_score']
        # If no overall score, calculate an average (example, adjust based on relevant metrics)
        elif metrics:
             relevant_scores = [v for k, v in metrics.items() if isinstance(v, (int, float)) and 'score' in k.lower()]
             if relevant_scores:
                 behavioral_score = round(sum(relevant_scores) / len(relevant_scores), 1)

        # Clean score if out of range (0-10)
        if behavioral_score is not None:
             behavioral_score = max(0.0, min(10.0, float(behavioral_score))) # Ensure float

        # Placeholder: Extract strengths/weaknesses if your behavioral analysis provides them
        # behavioral_strengths = behavioral_result.get('strengths', [])
        # behavioral_weaknesses = behavioral_result.get('weaknesses', [])

        logger.info(f"Extracted Behavioral Score: {behavioral_score}, Summary: {'Present' if behavioral_summary else 'Missing'}")
    else:
        logger.warning(f"Behavioral analysis data missing or indicates failure for session {session_id}.")


    # --- Extract data from Transcript Analysis ---
    transcript_overall_score = 0.0
    transcript_comm_score = 0.0
    transcript_tech_score = 0.0
    transcript_summary = "Transcript analysis failed or was not provided."
    transcript_strengths = []
    transcript_weaknesses = []
    transcript_recs = []
    interview_date_str = None # Get date from transcript result

    if transcript_result and isinstance(transcript_result, OverallAnalysis):
        # Access attributes directly from the Pydantic model
        transcript_overall_score = transcript_result.overall_score
        transcript_comm_score = transcript_result.communication_score
        transcript_tech_score = transcript_result.technical_knowledge_score
        transcript_summary = transcript_result.overall_summary
        transcript_strengths = transcript_result.strengths
        transcript_weaknesses = transcript_result.weaknesses
        transcript_recs = transcript_result.recommendations
        interview_date_str = transcript_result.interview_date # Get date from transcript analysis

        # Clean scores ensure they are float and within bounds
        transcript_overall_score = max(0.0, min(10.0, float(transcript_overall_score)))
        transcript_comm_score = max(0.0, min(10.0, float(transcript_comm_score)))
        transcript_tech_score = max(0.0, min(10.0, float(transcript_tech_score)))

        logger.info(f"Extracted Transcript Scores: Overall={transcript_overall_score}, Comm={transcript_comm_score}, Tech={transcript_tech_score}")
    else:
        logger.warning(f"Transcript analysis data missing or invalid type for session {session_id}.")


    # --- Calculate Final Weighted Score ---
    final_score = 0.0
    # Use only available scores for weighting
    if behavioral_score is not None and transcript_result is not None:
         # Ensure scores are floats before calculation
         final_score = round((float(behavioral_score) * WEIGHT_BEHAVIORAL) + (float(transcript_overall_score) * WEIGHT_TRANSCRIPT), 1)
         logger.info(f"Calculated weighted score: {final_score}")
    elif transcript_result is not None:
         final_score = round(float(transcript_overall_score), 1)
         logger.warning("Behavioral score missing, using transcript overall score as final.")
    elif behavioral_score is not None:
         final_score = round(float(behavioral_score), 1)
         logger.warning("Transcript score missing, using behavioral overall score as final.")
    else:
         logger.error("Both analysis results are missing scores. Final score is 0.")

    # Ensure final score is within bounds
    final_score = max(0.0, min(10.0, final_score))


    # --- Combine Lists (Remove duplicates, preserve order) ---
    def combine_unique(list1, list2):
        seen = set()
        combined = []
        # Ensure inputs are lists
        l1 = list1 if isinstance(list1, list) else []
        l2 = list2 if isinstance(list2, list) else []
        for item in l1 + l2:
            # Ensure item is hashable (e.g., string) before adding to set
            if isinstance(item, (str, int, float, tuple)):
                if item not in seen:
                    seen.add(item)
                    combined.append(item)
            elif isinstance(item, list): # Handle potential list of lists (less likely)
                 item_tuple = tuple(item)
                 if item_tuple not in seen:
                      seen.add(item_tuple)
                      combined.append(item) # Append original list
            # Add handling for other types if necessary
            else:
                 logger.warning(f"Skipping non-hashable item during list combination: {type(item)}")

        return combined

    combined_strengths = combine_unique(behavioral_strengths, transcript_strengths)
    combined_weaknesses = combine_unique(behavioral_weaknesses, transcript_weaknesses)
    # Assume recommendations primarily come from transcript analysis
    combined_recommendations = combine_unique([], transcript_recs) # Modify [] if behavioral provides recs


    # --- Create Report Object ---
    try:
        report_data = {
            "session_id": session_id,
            "candidate_id": candidate_id,
            "interview_date": interview_date_str, # Use date extracted from transcript analysis
            "behavioral_score": behavioral_score,
            "transcript_overall_score": transcript_overall_score,
            "transcript_communication_score": transcript_comm_score,
            "transcript_technical_score": transcript_tech_score,
            "final_weighted_score": final_score,
            "behavioral_summary": behavioral_summary,
            "transcript_summary": transcript_summary,
            "combined_strengths": combined_strengths,
            "combined_weaknesses": combined_weaknesses,
            "combined_recommendations": combined_recommendations,
        }
        # Create Pydantic object - this also validates the data types/ranges
        report = CombinedAnalysisReport(**report_data)
        logger.info(f"Successfully created combined analysis report object for {session_id}.")
        return report
    except Exception as e:
         logger.error(f"Failed to create CombinedAnalysisReport Pydantic object: {e}", exc_info=True)
         # Raise a specific error indicating report creation failure
         raise ValueError(f"Could not construct final report due to data validation errors: {e}")