# app/services/combined_analyzer.py
import logging
import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import google.generativeai as genai
from PIL import Image

# --- Import schemas from the new location ---
from app.models.analysis_schemas import OverallAnalysis, CombinedAnalysisReport

logger = logging.getLogger(__name__)

class CombinedAnalyzer:
    """
    This service performs behavioral analysis from snapshots
    and combines it with transcript analysis into a final report.
    """

    def __init__(self):
        """Initialize the analyzer with Gemini API."""
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            
            genai.configure(api_key=api_key)
            # Use the model specified for vision analysis
            self.model = genai.GenerativeModel('models/gemini-2.5-flash') 
            logger.info("✅ CombinedAnalyzer initialized (Gemini model loaded)")
            
        except Exception as e:
            logger.error(f"Failed to initialize CombinedAnalyzer: {e}")
            raise

    # --------------------------------------------------------------------
    # --- Behavioral Analysis Methods (Moved from behavioral_analyzer.py) ---
    # --------------------------------------------------------------------

    def perform_behavioral_analysis(
        self, 
        user_id: str, 
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze interview behavior from captured snapshots.
        This is the main entry point for behavioral analysis.
        
        Args:
            user_id: Unique identifier for the candidate
            session_id: Unique identifier for the interview session
            
        Returns:
            Dictionary containing analysis results, or an error dict
        """
        try:
            logger.info(f"🔍 Starting behavioral analysis for {user_id}/{session_id}")
            
            # Locate snapshot directory
            snapshot_dir = Path("data") / user_id / session_id / "snapshots"
            
            if not snapshot_dir.exists():
                logger.error(f"Snapshot directory not found: {snapshot_dir}")
                return self._create_error_result(user_id, session_id, "No snapshot directory")
            
            # Load all snapshot images
            snapshot_files = sorted(snapshot_dir.glob("snapshot_*.jpg"))
            
            if not snapshot_files:
                logger.warning(f"No snapshots found in {snapshot_dir}")
                return self._create_error_result(user_id, session_id, "No snapshots found")
            
            logger.info(f"📸 Found {len(snapshot_files)} snapshots for analysis")
            
            # Load images
            images = []
            for snapshot_path in snapshot_files:
                try:
                    img = Image.open(snapshot_path)
                    images.append(img)
                except Exception as e:
                    logger.warning(f"Failed to load {snapshot_path}: {e}")
            
            if not images:
                logger.error("No valid images could be loaded")
                return self._create_error_result(user_id, session_id, "Failed to load images")
            
            logger.info(f"✅ Loaded {len(images)} valid images")
            
            # Construct the analysis prompt
            prompt = self._create_metrics_prompt(len(images))
            
            # Prepare content for API call (text + images)
            content = [prompt] + images
            
            logger.info("🤖 Sending behavioral request to Gemini API...")
            
            # Call Gemini API
            response = self.model.generate_content(content)
            
            if not response or not response.text:
                logger.error("Empty response from Gemini API for behavioral analysis")
                return self._create_error_result(user_id, session_id, "Empty API response")
            
            logger.info("✅ Received behavioral analysis from Gemini API")
            
            # Parse and structure the response with metrics
            analysis_result = self._parse_metrics_response(
                response.text,
                user_id,
                session_id,
                len(images)
            )
            
            # Save all behavioral reports to disk
            file_paths = self._save_all_behavioral_reports(user_id, session_id, analysis_result)
            
            logger.info(f"💾 All behavioral analysis reports saved successfully")
            
            # Return the full result for combining
            return {
                "status": "success",
                "user_id": user_id,
                "session_id": session_id,
                "analysis_timestamp": analysis_result['analysis_timestamp'],
                "total_snapshots": len(images),
                "duration_seconds": len(images) * 5,
                "files_generated": file_paths,
                "metrics": analysis_result.get('metrics', {}), 
                "summary": analysis_result.get('metrics', {}).get('summary', ''),
                "overall_confidence_score": analysis_result.get('metrics', {}).get('overall_confidence_score'),
                "hiring_recommendation": analysis_result.get('metrics', {}).get('hiring_recommendation')
            }
            
        except Exception as e:
            logger.error(f"❌ Error during behavioral analysis: {e}", exc_info=True)
            return self._create_error_result(user_id, session_id, str(e))
    
    def _create_metrics_prompt(self, num_images: int) -> str:
        """Create the analysis prompt requesting structured metrics."""
        prompt = f"""You are an expert interview analyst with extensive experience in evaluating candidate behavior during interviews. 

    I am providing you with {num_images} sequential snapshots captured throughout a virtual interview session (one snapshot every 5 seconds). 
    Your task is twofold:
    1.  **Identity Consistency Check**: First and most importantly, verify if the same person is present throughout all snapshots.
    2.  **Behavioral Assessment**: Analyze the candidate's behavior and provide a comprehensive behavioral assessment.

    **CRITICAL**: You must provide your response in the exact JSON format specified below. All scores must be numeric values.

    Please analyze and provide metrics in this EXACT JSON structure:

    ```json
    {{
    "identity_consistency": {{
        "is_consistent": <true/false>,
        "confidence_score": <1-10>,
        "observations": "Briefly state reasoning, e.g., 'Same person visible in all frames' or 'Different person appears near the end'."
    }},
    "overall_confidence_score": <1-10>,
    "emotion_analysis": {{
        "dominant_emotions": ["emotion1", "emotion2", "emotion3"],
        "emotional_stability_score": <1-10>,
        "emotional_appropriateness_score": <1-10>,
        "emotion_timeline": {{
        "beginning": "emotion",
        "middle": "emotion",
        "end": "emotion"
        }},
        "anxiety_level": <1-10>,
        "enthusiasm_level": <1-10>
    }},
    "eye_contact": {{
        "quality_score": <1-10>,
        "consistency_score": <1-10>,
        "percentage_maintained": <0-100>,
        "appropriateness_rating": "poor/fair/good/excellent"
    }},
    "gaze_behavior": {{
        "gaze_shift_frequency": "low/moderate/high/excessive",
        "average_shifts_per_minute": <number>,
        "looking_away_percentage": <0-100>,
        "shift_patterns": ["reading", "thinking", "distraction", "natural"],
        "dishonesty_indicators": <1-10>
    }},
    "body_movement": {{
        "movement_level": "minimal/moderate/active/excessive",
        "fidgeting_score": <1-10>,
        "hand_gesture_naturalness": <1-10>,
        "self_soothing_frequency": "none/rare/occasional/frequent",
        "distracting_movements_score": <1-10>
    }},
    "posture_composure": {{
        "posture_score": <1-10>,
        "engagement_level": <1-10>,
        "professional_presentation": <1-10>,
        "posture_consistency": <1-10>,
        "physical_comfort_score": <1-10>
    }},
    "communication_quality": {{
        "body_language_clarity": <1-10>,
        "non_verbal_confidence": <1-10>,
        "attentiveness_score": <1-10>
    }},
    "red_flags": {{
        "detected": true/false,
        "count": <number>,
        "descriptions": ["flag1", "flag2"]
    }},
    "positive_indicators": {{
        "count": <number>,
        "descriptions": ["indicator1", "indicator2"]
    }},
    "key_strengths": ["strength1", "strength2", "strength3"],
    "areas_for_improvement": ["area1", "area2", "area3"],
    "interview_readiness_score": <1-10>,
    "hiring_recommendation": "strong_yes/yes/maybe/no/strong_no",
    "hiring_confidence": <1-10>,
    "summary": "brief overall assessment in 2-3 sentences",
    "detailed_observations": "comprehensive narrative of behavioral patterns observed"
    }}
    ```
    """
        return prompt

    def _parse_metrics_response(
        self,
        response_text: str,
        user_id: str,
        session_id: str,
        num_snapshots: int
    ) -> Dict[str, Any]:
        """Parse the structured metrics response from Gemini API."""
        
        try:
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without code blocks
                json_str = response_text.strip()
            
            # Parse JSON
            metrics = json.loads(json_str)
            
            logger.info("✅ Successfully parsed structured metrics")
            
            # Validate and add metadata
            analysis_result = {
                "user_id": user_id,
                "session_id": session_id,
                "analysis_timestamp": datetime.now().isoformat(),
                "total_snapshots_analyzed": num_snapshots,
                "metadata": {
                    "capture_interval_seconds": 5,
                    "total_duration_seconds": num_snapshots * 5,
                    "model_used": "gemini-2.5-flash",
                    "analysis_version": "2.1-metrics-identity"
                },
                "metrics": metrics,
                "raw_response": response_text,
                "parsing_status": "success"
            }
            
            return analysis_result
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse JSON metrics: {e}")
            # Fallback to raw response
            return {
                "user_id": user_id,
                "session_id": session_id,
                "analysis_timestamp": datetime.now().isoformat(),
                "total_snapshots_analyzed": num_snapshots,
                "metadata": {
                    "capture_interval_seconds": 5,
                    "total_duration_seconds": num_snapshots * 5,
                    "model_used": "gemini-2.5-flash",
                    "analysis_version": "2.1-metrics-identity"
                },
                "metrics": None,
                "raw_response": response_text,
                "parsing_status": "failed",
                "error": str(e)
            }

    def _save_all_behavioral_reports(
        self,
        user_id: str,
        session_id: str,
        analysis_result: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Save all behavioral analysis reports to disk.
        Returns dictionary of generated file paths.
        """
        try:
            # Create report directory
            report_dir = Path("data") / user_id / session_id / "behavioral_analysis"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            file_paths = {}
            
            # 1. Save complete JSON with metrics
            json_path = report_dir / "analysis_complete.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=2, ensure_ascii=False)
            file_paths['json_complete'] = str(json_path)
            
            # 2. Save metrics-only JSON (simplified)
            if analysis_result.get('metrics'):
                metrics_json_path = report_dir / "metrics_only.json"
                with open(metrics_json_path, 'w', encoding='utf-8') as f:
                    json.dump(analysis_result['metrics'], f, indent=2, ensure_ascii=False)
                file_paths['json_metrics'] = str(metrics_json_path)
            
            # 3. Save formatted text report
            txt_path = report_dir / "report_formatted.txt"
            self._generate_formatted_report(analysis_result, txt_path)
            file_paths['text_report'] = str(txt_path)
            
            # 4. Save metrics CSV
            csv_path = report_dir / "metrics_data.csv"
            self._generate_csv_metrics(analysis_result, csv_path)
            file_paths['csv_metrics'] = str(csv_path)
            
            # 5. Save executive summary
            summary_path = report_dir / "executive_summary.txt"
            self._generate_executive_summary(analysis_result, summary_path)
            file_paths['executive_summary'] = str(summary_path)
            
            # 6. Save raw response (for debugging)
            raw_path = report_dir / "raw_response.txt"
            with open(raw_path, 'w', encoding='utf-8') as f:
                f.write(analysis_result.get('raw_response', 'No raw response available'))
            file_paths['raw_response'] = str(raw_path)
            
            # 7. Create README with file descriptions
            readme_path = report_dir / "README.txt"
            self._generate_readme(file_paths, readme_path)
            file_paths['readme'] = str(readme_path)
            
            logger.info(f"💾 All behavioral reports saved to: {report_dir}")
            return file_paths
            
        except Exception as e:
            logger.error(f"❌ Failed to save behavioral analysis reports: {e}", exc_info=True)
            return {}

    def _generate_formatted_report(self, analysis_result: Dict[str, Any], output_path: Path):
        """Generate a human-readable formatted report from metrics."""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("BEHAVIORAL ANALYSIS REPORT - COMPREHENSIVE METRICS\n")
            f.write("=" * 100 + "\n\n")
            
            f.write(f"Candidate ID: {analysis_result['user_id']}\n")
            f.write(f"Session ID: {analysis_result['session_id']}\n")
            f.write(f"Analysis Date: {analysis_result['analysis_timestamp']}\n")
            f.write(f"Total Snapshots Analyzed: {analysis_result['total_snapshots_analyzed']}\n")
            f.write(f"Interview Duration: ~{analysis_result['metadata']['total_duration_seconds']} seconds\n")
            f.write(f"Analysis Model: {analysis_result['metadata']['model_used']}\n")
            f.write(f"Analysis Version: {analysis_result['metadata'].get('analysis_version', '1.0')}\n\n")
            
            if analysis_result.get('metrics'):
                metrics = analysis_result['metrics']
                
                f.write("=" * 100 + "\n")
                f.write("IDENTITY CONSISTENCY CHECK\n")
                f.write("=" * 100 + "\n")
                identity = metrics.get('identity_consistency', {})
                is_consistent = identity.get('is_consistent', None)
                
                if is_consistent is True:
                    f.write("Status: ✅ CONSISTENT\n")
                elif is_consistent is False:
                    f.write("Status: ⚠️ INCONSISTENT (Potential Red Flag)\n")
                else:
                    f.write("Status: ❓ UNKNOWN (Check Data)\n")
                    
                f.write(f"Confidence: {identity.get('confidence_score', 'N/A')}/10\n")
                f.write(f"Observations: {identity.get('observations', 'N/A')}\n\n")

                f.write("=" * 100 + "\n")
                f.write("OVERALL ASSESSMENT\n")
                f.write("=" * 100 + "\n")
                f.write(f"Overall Confidence Score: {metrics.get('overall_confidence_score', 'N/A')}/10\n")
                f.write(f"Interview Readiness Score: {metrics.get('interview_readiness_score', 'N/A')}/10\n")
                f.write(f"Hiring Recommendation: {metrics.get('hiring_recommendation', 'N/A').upper()}\n")
                f.write(f"Hiring Confidence: {metrics.get('hiring_confidence', 'N/A')}/10\n")
                f.write(f"\n{metrics.get('summary', 'N/A')}\n\n")
                
                # (Omitted other sections for brevity, they are unchanged)
                
            else:
                f.write("\n⚠️  [ERROR] Failed to extract structured metrics\n")
                f.write("\nRaw Response:\n")
                f.write("-" * 100 + "\n")
                f.write(analysis_result.get('raw_response', 'N/A'))
                f.write("\n" + "-" * 100 + "\n")
            
            f.write("\n" + "=" * 100 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 100 + "\n")

    def _generate_csv_metrics(self, analysis_result: Dict[str, Any], output_path: Path):
        """Generate CSV file with flattened metrics for easy analysis."""
        # (This function is unchanged)
        try:
            if not analysis_result.get('metrics'): return
            metrics = analysis_result['metrics']
            flat_metrics = {
                'user_id': analysis_result['user_id'],
                'session_id': analysis_result['session_id'],
                # ... (all other metrics) ...
                'positive_indicators_count': metrics.get('positive_indicators', {}).get('count')
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(','.join(flat_metrics.keys()) + '\n')
                f.write(','.join(str(v) if v is not None else '' for v in flat_metrics.values()) + '\n')
        except Exception as e:
            logger.error(f"❌ Failed to generate CSV metrics: {e}")

    def _generate_executive_summary(self, analysis_result: Dict[str, Any], output_path: Path):
        """Generate a concise executive summary."""
        # (This function is unchanged)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("EXECUTIVE SUMMARY - BEHAVIORAL ANALYSIS\n")
            # ... (all content) ...
            f.write("=" * 80 + "\n")

    def _generate_readme(self, file_paths: Dict[str, str], output_path: Path):
        """Generate README explaining all generated files."""
        # (This function is unchanged)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("BEHAVIORAL ANALYSIS - FILE DIRECTORY\n")
            # ... (all content) ...
            f.write(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
    
    def _create_error_result(self, user_id: str, session_id: str, error_msg: str) -> Dict[str, Any]:
        """Create an error result structure."""
        return {
            "status": "error",
            "user_id": user_id,
            "session_id": session_id,
            "error_message": error_msg,
            "analysis_timestamp": datetime.now().isoformat()
        }


    # --------------------------------------------------------------------
    # --- Combined Analysis Methods (Logic from old combined_analyzer.py) ---
    # --------------------------------------------------------------------

    def combine_analyses(
        self,
        behavioral_result: Optional[Dict], # Result from self.perform_behavioral_analysis
        transcript_result: Optional[OverallAnalysis], # Result from TranscriptAnalyzer
        session_id: str,
        candidate_id: str,
    ) -> CombinedAnalysisReport:
        """
        Combines results from behavioral and transcript analysis into a final report.
        """
        logger.info(f"Combining analyses for session: {session_id}")

        behavioral_summary = None
        behavioral_score = None
        behavioral_strengths = []
        behavioral_weaknesses = []
        
        # --- FIX: Check for behavioral_result *and* its 'status' ---
        is_behavioral_success = behavioral_result and behavioral_result.get('status') == 'success'
        # --- END FIX ---

        if is_behavioral_success:
            metrics = behavioral_result.get('metrics', {})
            identity_info = metrics.get('identity_consistency', {})
            is_consistent = identity_info.get('is_consistent', None)
            identity_obs = identity_info.get('observations', 'N/A')
            
            identity_warning = ""
            if is_consistent is False:
                identity_warning = f"CRITICAL IDENTITY WARNING: The analysis detected an inconsistent person during the interview. Observations: {identity_obs} | "
                logger.error(f"Identity inconsistency detected for session {session_id}: {identity_obs}")
            elif is_consistent is True:
                identity_warning = "[Identity Check: Consistent] | "
            else:
                identity_warning = "[Identity Check: Unknown] | "

            behavioral_summary = metrics.get('summary', 'No summary provided.')
            behavioral_summary = identity_warning + behavioral_summary

            if 'overall_confidence_score' in metrics:
                behavioral_score = metrics['overall_confidence_score']
            elif metrics:
                relevant_scores = [v for k, v in metrics.items() if isinstance(v, (int, float)) and 'score' in k.lower() and k not in ['identity_confidence']]
                if relevant_scores:
                    behavioral_score = round(sum(relevant_scores) / len(relevant_scores), 1)

            if behavioral_score is not None:
                behavioral_score = max(0.0, min(10.0, float(behavioral_score))) 

            behavioral_strengths = metrics.get('key_strengths', [])
            behavioral_weaknesses = metrics.get('areas_for_improvement', [])

            logger.info(f"Extracted Behavioral Score: {behavioral_score}, Summary: {'Present' if behavioral_summary else 'Missing'}")
        else:
            logger.warning(f"Behavioral analysis data missing or indicates failure for session {session_id}.")


        transcript_overall_score = 0.0
        transcript_comm_score = 0.0
        transcript_tech_score = 0.0
        transcript_summary = "Transcript analysis failed or was not provided."
        transcript_strengths = []
        transcript_weaknesses = []
        transcript_recs = []
        interview_date_str = None 
        
        # --- FIX: Check for transcript_result (it can be None) ---
        is_transcript_success = transcript_result and isinstance(transcript_result, OverallAnalysis)
        # --- END FIX ---

        if is_transcript_success:
            transcript_overall_score = transcript_result.overall_score
            transcript_comm_score = transcript_result.communication_score
            transcript_tech_score = transcript_result.technical_knowledge_score
            transcript_summary = transcript_result.overall_summary
            transcript_strengths = transcript_result.strengths
            transcript_weaknesses = transcript_result.weaknesses
            transcript_recs = transcript_result.recommendations
            interview_date_str = transcript_result.interview_date 

            transcript_overall_score = max(0.0, min(10.0, float(transcript_overall_score)))
            transcript_comm_score = max(0.0, min(10.0, float(transcript_comm_score)))
            transcript_tech_score = max(0.0, min(10.0, float(transcript_tech_score)))

            logger.info(f"Extracted Transcript Scores: Overall={transcript_overall_score}, Comm={transcript_comm_score}, Tech={transcript_tech_score}")
        else:
            logger.warning(f"Transcript analysis data missing or invalid type for session {session_id}.")

        
        # --- FIX: Check for *any* success before proceeding ---
        if not is_behavioral_success and not is_transcript_success:
            logger.error("Both analysis results are missing or failed. Final report cannot be generated.")
            raise ValueError("No valid analysis data was provided to combine.")
        # --- END FIX ---


        final_score = 0.0
        scores_found = 0
        total_score = 0.0

        if behavioral_score is not None:
            total_score += float(behavioral_score)
            scores_found += 1
        
        if is_transcript_success: # Check if transcript analysis was successful
            total_score += float(transcript_overall_score)
            scores_found += 1

        if scores_found > 0:
            final_score = round(total_score / scores_found, 1)
            logger.info(f"Calculated average score: {final_score} from {scores_found} sources.")
        else:
            # This should now be unreachable due to the check above
            logger.error("Both analysis results are missing scores. Final score is 0.")

        final_score = max(0.0, min(10.0, final_score))


        def combine_unique(list1, list2):
            seen = set()
            combined = []
            l1 = list1 if isinstance(list1, list) else []
            l2 = list2 if isinstance(list2, list) else []
            for item in l1 + l2:
                if isinstance(item, (str, int, float, tuple)):
                    if item not in seen:
                        seen.add(item)
                        combined.append(item)
                elif isinstance(item, list):
                    item_tuple = tuple(item)
                    if item_tuple not in seen:
                        seen.add(item_tuple)
                        combined.append(item) 
                else:
                    logger.warning(f"Skipping non-hashable item during list combination: {type(item)}")
            return combined

        combined_strengths = combine_unique(behavioral_strengths, transcript_strengths)
        combined_weaknesses = combine_unique(behavioral_weaknesses, transcript_weaknesses)
        combined_recommendations = combine_unique([], transcript_recs) 


        try:
            report_data = {
                "session_id": session_id,
                "candidate_id": candidate_id,
                "interview_date": interview_date_str, 
                "behavioral_score": behavioral_score,
                "transcript_overall_score": transcript_overall_score,
                "transcript_communication_score": transcript_comm_score,
                "transcript_technical_score": transcript_tech_score,
                "final_weighted_score": final_score, # This is now an average
                "behavioral_summary": behavioral_summary,
                "transcript_summary": transcript_summary,
                "combined_strengths": combined_strengths,
                "combined_weaknesses": combined_weaknesses,
                "combined_recommendations": combined_recommendations,
            }
            report = CombinedAnalysisReport(**report_data)
            logger.info(f"Successfully created combined analysis report object for {session_id}.")
            return report
        except Exception as e:
            logger.error(f"Failed to create CombinedAnalysisReport Pydantic object: {e}", exc_info=True)
            raise ValueError(f"Could not construct final report due to data validation errors: {e}")