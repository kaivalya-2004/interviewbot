# app/services/transcript_analyzer.py
# app/services/transcript_analyzer.py
import google.generativeai as genai
import json
import re
import logging
from typing import Dict, Any
from fastapi import HTTPException
# --- Import schemas from the models directory ---
from app.models.analysis_schemas import OverallAnalysis, QuestionAnalysis
# --- END Import ---
import os # Import os for env var

logger = logging.getLogger(__name__)

# --- Get settings directly from env ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Use a specific model for analysis, separate from the chat model
TRANSCRIPT_ANALYSIS_MODEL_NAME = os.getenv("TRANSCRIPT_ANALYSIS_MODEL", "gemini-2.5-pro") # Default to pro for analysis
# --- END Config ---


class TranscriptAnalyzer:
    """Service to analyze interviews using Gemini AI based on transcripts"""

    def __init__(self):
        """Initialize Gemini API"""
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not configured for TranscriptAnalyzer")

        # Configure API key if not already done globally (safe to call again)
        try:
             genai.configure(api_key=GEMINI_API_KEY)
        except Exception as e:
             logger.warning(f"Potentially re-configuring Gemini API key: {e}")

        # Use the specific model defined for transcript analysis
        try:
            self.model = genai.GenerativeModel(TRANSCRIPT_ANALYSIS_MODEL_NAME)
            logger.info(f"TranscriptAnalyzer initialized with model: {TRANSCRIPT_ANALYSIS_MODEL_NAME}")
        except Exception as e:
             logger.error(f"Failed to initialize Gemini model '{TRANSCRIPT_ANALYSIS_MODEL_NAME}': {e}")
             raise ValueError(f"Could not initialize Gemini model for transcript analysis: {e}")

    # Marked as regular function, use asyncio.to_thread if called from async context
    def analyze(self, resume_content: str, transcript_data: Dict[str, Any]) -> OverallAnalysis:
        """
        Analyze interview using Gemini based on resume and parsed transcript.

        Args:
            resume_content: Candidate's resume text
            transcript_data: Parsed transcript dict {'metadata': {...}, 'qa_pairs': [...]}

        Returns:
            OverallAnalysis Pydantic object.

        Raises:
            ValueError: If analysis fails, response parsing fails, or input is invalid.
            RuntimeError: If Gemini API call fails unexpectedly.
        """
        logger.info(f"Starting transcript analysis for session {transcript_data.get('metadata', {}).get('session_id', 'N/A')}")
        if not resume_content: logger.warning("Resume content is empty for analysis.")
        if not transcript_data or not transcript_data.get('qa_pairs'):
             logger.error("No Q&A pairs provided for transcript analysis.")
             raise ValueError("Cannot analyze transcript without Q&A pairs.")

        try:
            prompt = self._create_analysis_prompt(resume_content, transcript_data)

            # Use generate_content (blocking call)
            logger.debug("Sending analysis prompt to Gemini...")
            response = self.model.generate_content(prompt)

            # --- Log Token Usage ---
            try:
                # Attempt to get metadata directly
                metadata = getattr(response, 'usage_metadata', None)
                # Fallback check via prompt_feedback if direct access fails
                if not metadata:
                    feedback = getattr(response, 'prompt_feedback', None)
                    metadata = getattr(feedback, 'usage_metadata', None) if feedback else None

                if metadata:
                    prompt_tokens = getattr(metadata, 'prompt_token_count', 'N/A')
                    # Use candidates_token_count for response tokens
                    response_tokens = getattr(metadata, 'candidates_token_count', 'N/A')
                    total_tokens = getattr(metadata, 'total_token_count', 'N/A')
                    # Make log message specific
                    logger.info(f"Gemini Transcript Analysis Tokens - Prompt: {prompt_tokens}, Response: {response_tokens}, Total: {total_tokens}")
                else:
                    logger.warning("Usage metadata not found in Gemini response for transcript analysis.")

            except AttributeError as e:
                logger.warning(f"Could not access token counts from transcript analysis metadata: {e}")
            except Exception as e:
                 logger.error(f"Unexpected error logging transcript analysis tokens: {e}")
            # --- End Log Token Usage ---


            response_text = response.text.strip()
            logger.debug(f"Received Gemini response (length: {len(response_text)}). Cleaning...")

            cleaned_json_text = self._clean_response(response_text)
            logger.debug("Cleaned Gemini response. Attempting JSON parse...")

            analysis_data = json.loads(cleaned_json_text)
            logger.info("✅ Successfully parsed Gemini analysis response.")

            # Add/ensure metadata fields required by the Pydantic model
            analysis_data['session_id'] = transcript_data.get('metadata', {}).get('session_id', 'N/A')
            analysis_data['interview_date'] = transcript_data.get('metadata', {}).get('date', 'N/A')
            analysis_data['user_id'] = transcript_data.get('metadata', {}).get('user_id', 'N/A')


            # Validate essential keys before creating object
            required_keys = ["questions_analyzed", "overall_summary", "strengths", "weaknesses",
                             "resume_alignment_score", "communication_score", "technical_knowledge_score",
                             "overall_score", "recommendations", "session_id"]
            missing = [key for key in required_keys if key not in analysis_data]
            if missing:
                 logger.error(f"Gemini response missing required keys: {missing}")
                 raise ValueError(f"Gemini analysis JSON is missing required keys: {missing}")

            # Use Pydantic model for validation and structure
            return OverallAnalysis(**analysis_data)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {e}", exc_info=True)
            logger.error(f"Problematic Text (first 1000 chars): {cleaned_json_text[:1000]}")
            # Raise ValueError instead of HTTPException
            raise ValueError(f"Failed to parse analysis from AI model: {e}")
        except ValueError as e: # Catch validation errors from missing keys or Pydantic
             logger.error(f"Validation error in Gemini response data: {e}", exc_info=True)
             raise ValueError(f"Invalid analysis data received: {e}")
        except Exception as e:
            logger.error(f"Gemini analysis error: {e}", exc_info=True)
            # Raise generic error
            raise RuntimeError(f"Error during AI transcript analysis: {type(e).__name__}")


    def _create_analysis_prompt(self, resume_content: str, transcript_data: Dict[str, Any]) -> str:
        """Create detailed prompt for Gemini analysis (using updated QA structure)"""
        qa_pairs_str = ""
        for i, pair in enumerate(transcript_data.get('qa_pairs', [])):
             q_follow = "(Follow-up)" if pair.get('is_follow_up_question') else ""
             a_follow = "(Response to Follow-up)" if pair.get('is_follow_up_answer') else ""
             qa_pairs_str += f"\n{i+1}. Question {q_follow} (Timestamp: {pair.get('timestamp', 'N/A')}):\n"
             qa_pairs_str += f"   Q: {pair.get('question', 'N/A')}\n"
             qa_pairs_str += f"   A: {a_follow} {pair.get('answer', 'N/A')}\n"

        # Extract metadata safely
        metadata = transcript_data.get('metadata', {})
        session_id = metadata.get('session_id', 'N/A')
        user_id = metadata.get('user_id', 'N/A')
        interview_date = metadata.get('date', 'N/A')


        return f"""
Analyze the following interview transcript and candidate resume. You are an expert HR analyst.

**Candidate Resume:**
{resume_content if resume_content else "Resume content not provided."}

**Interview Metadata:**
- User ID: {user_id}
- Session ID: {session_id}
- Date: {interview_date}

**Interview Transcript (Q&A Pairs):**
{qa_pairs_str if qa_pairs_str else "No Q&A pairs extracted."}

**Analysis Instructions:**

**1. Per-Question Analysis:** For EACH Q&A pair above, provide:
    * `timestamp`: The timestamp of the question.
    * `question`: The exact question text.
    * `answer`: The exact answer text.
    * `relevance_to_resume`: (String) Explain if/how the question relates to the candidate's resume. Assess alignment (e.g., "Directly related," "Loosely related," "Not related").
    * `answer_quality`: (String Enum: "Excellent", "Good", "Fair", "Poor") Rate the completeness, clarity, structure, and depth of the answer.
    * `technical_accuracy`: (String) Assess the technical correctness of the answer if applicable ("Accurate", "Partially Accurate", "Inaccurate", "Not Applicable"). Add brief justification.
    * `analysis`: (String) Provide a detailed analysis (2-3 sentences) explaining the ratings, mentioning specific strengths/weaknesses in the response (e.g., specific examples, confidence, structure, keywords missing/present).
    * `score`: (Float) A numerical score from 0.0 to 10.0 reflecting the overall quality of the answer relative to the question. Score 0-2 for "No answer provided", "I don't know", or completely unintelligible answers.

**2. Overall Analysis:** Provide:
    * `overall_summary`: (String) A concise 3-4 sentence summary of the candidate's performance, highlighting key takeaways.
    * `strengths`: (List of Strings) 3-5 key strengths demonstrated by the candidate during the interview (e.g., "Clear communication on Project X," "Solid understanding of Y concept").
    * `weaknesses`: (List of Strings) 3-5 specific areas where the candidate needs improvement (e.g., "Lacked depth on Z algorithm," "Answer structure could be clearer," "Struggled to connect experience to question").
    * `resume_alignment_score`: (Float, 0-10) How well did the *candidate's answers* align with and support the experience stated on their resume?
    * `communication_score`: (Float, 0-10) Assess clarity, articulation, confidence, and conciseness based *only* on the provided transcript text.
    * `technical_knowledge_score`: (Float, 0-10) Evaluate the depth and accuracy of technical understanding demonstrated across all relevant answers. Consider N/A questions appropriately.
    * `overall_score`: (Float, 0-10) Your final holistic score for the candidate's performance in this interview, considering all factors.
    * `recommendations`: (List of Strings) 3-5 specific, actionable recommendations for the candidate *or* the hiring manager (e.g., "Suggest candidate review core concepts of X," "Probe deeper into Y experience in next round," "Candidate seems suitable for Z role type").

**Output Format:**
Respond ONLY with a valid JSON object matching the structure below. Do NOT include ```json markdown markers or any text before or after the JSON object. Ensure all string values are properly escaped within the JSON. Include the user_id, session_id, and interview_date in the top level of the JSON.

```json
{{
    "session_id": "{session_id}",
    "user_id": "{user_id}",
    "interview_date": "{interview_date}",
    "questions_analyzed": [
        {{
            "timestamp": "YYYY-MM-DD HH:MM:SS",
            "question": "The first question text...",
            "answer": "The first answer text...",
            "relevance_to_resume": "Explanation...",
            "answer_quality": "Good",
            "technical_accuracy": "Accurate",
            "analysis": "Detailed analysis...",
            "score": 8.0
        }}
        // ... more question analysis objects
    ],
    "overall_summary": "Overall summary text...",
    "strengths": ["Strength 1...", "Strength 2..."],
    "weaknesses": ["Weakness 1...", "Weakness 2..."],
    "resume_alignment_score": 7.5,
    "communication_score": 8.0,
    "technical_knowledge_score": 6.5,
    "overall_score": 7.3,
    "recommendations": ["Recommendation 1...", "Recommendation 2..."]
}}
"""
    
    def _clean_response(self, response_text: str) -> str:
        """Clean Gemini response text to extract pure JSON"""
        # Remove markdown code blocks ```json ... ``` or ``` ... ```
        cleaned = re.sub(r'^```(?:json)?\s*', '', response_text, flags=re.MULTILINE)
        cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE)

        start_index = cleaned.find('{')
        end_index = cleaned.rfind('}')

        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_text = cleaned[start_index : end_index + 1]
            # Further clean potential escape issues if needed, but start simple
            return json_text.strip()
        else:
            logger.warning("Could not find valid JSON structure '{...}' in Gemini response.")
            # Raise ValueError instead of returning potentially bad text
            raise ValueError("AI response did not contain a valid JSON object.")