# app/services/behavioral_analyzer.py - UPDATED VERSION
# This version performs analysis in background and stores everything to files
# No frontend display logic - pure backend processing

import logging
import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import google.generativeai as genai
from PIL import Image
logger = logging.getLogger(__name__)


class BehavioralAnalyzer:
    """
    Backend-only behavioral analyzer.
    Performs analysis and stores all results to files.
    No frontend display logic included.
    """
    
    def __init__(self):
        """Initialize the behavioral analyzer with Gemini API."""
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('models/gemini-2.5-flash')
            logger.info("✅ Behavioral analyzer initialized (Backend mode)")
            
        except Exception as e:
            logger.error(f"Failed to initialize behavioral analyzer: {e}")
            raise
    
    def analyze_interview_snapshots(
        self, 
        user_id: str, 
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze interview behavior from captured snapshots.
        All results are stored to files automatically.
        
        Args:
            user_id: Unique identifier for the candidate
            session_id: Unique identifier for the interview session
            
        Returns:
            Dictionary containing file paths and basic status, or None if failed
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
            
            logger.info("🤖 Sending request to Gemini API...")
            
            # Call Gemini API
            response = self.model.generate_content(content)
            
            if not response or not response.text:
                logger.error("Empty response from Gemini API")
                return self._create_error_result(user_id, session_id, "Empty API response")
            
            logger.info("✅ Received analysis from Gemini API")
            
            # Parse and structure the response with metrics
            analysis_result = self._parse_metrics_response(
                response.text,
                user_id,
                session_id,
                len(images)
            )
            
            # Save all reports to disk
            file_paths = self._save_all_reports(user_id, session_id, analysis_result)
            
            logger.info(f"💾 All analysis reports saved successfully")
            
            # Return minimal result with file paths
            return {
                "status": "success",
                "user_id": user_id,
                "session_id": session_id,
                "analysis_timestamp": analysis_result['analysis_timestamp'],
                "total_snapshots": len(images),
                "duration_seconds": len(images) * 5,
                "files_generated": file_paths,
                "overall_confidence_score": analysis_result.get('metrics', {}).get('overall_confidence_score'),
                "hiring_recommendation": analysis_result.get('metrics', {}).get('hiring_recommendation')
            }
            
        except Exception as e:
            logger.error(f"❌ Error during behavioral analysis: {e}", exc_info=True)
            return self._create_error_result(user_id, session_id, str(e))
    
    def _create_metrics_prompt(self, num_images: int) -> str:
        """Create the analysis prompt requesting structured metrics."""
        prompt = f"""You are an expert interview analyst with extensive experience in evaluating candidate behavior during interviews. 

I am providing you with {num_images} sequential snapshots captured throughout a virtual interview session (one snapshot every 5 seconds). Your task is to analyze the candidate's behavior and provide a comprehensive behavioral assessment in a STRUCTURED METRICS FORMAT.

**CRITICAL**: You must provide your response in the exact JSON format specified below. All scores must be numeric values.

Please analyze and provide metrics in this EXACT JSON structure:

```json
{{
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

**Scoring Guidelines:**
- 1-3: Poor/Concerning
- 4-6: Fair/Needs Improvement
- 7-8: Good/Satisfactory
- 9-10: Excellent/Outstanding

**For Anxiety, Fidgeting, Dishonesty Indicators (lower is better):**
- 1-3: Low/Minimal concern
- 4-6: Moderate concern
- 7-10: High concern

Provide ONLY the JSON output, no additional text before or after."""

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
                    "model_used": "gemini-2.5-pro",
                    "analysis_version": "2.0-metrics"
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
                    "model_used": "gemini-2.5-pro",
                    "analysis_version": "2.0-metrics"
                },
                "metrics": None,
                "raw_response": response_text,
                "parsing_status": "failed",
                "error": str(e)
            }
    
    def _save_all_reports(
        self,
        user_id: str,
        session_id: str,
        analysis_result: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Save all analysis reports to disk.
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
            logger.info(f"✅ Saved complete JSON: {json_path}")
            
            # 2. Save metrics-only JSON (simplified)
            if analysis_result.get('metrics'):
                metrics_json_path = report_dir / "metrics_only.json"
                with open(metrics_json_path, 'w', encoding='utf-8') as f:
                    json.dump(analysis_result['metrics'], f, indent=2, ensure_ascii=False)
                file_paths['json_metrics'] = str(metrics_json_path)
                logger.info(f"✅ Saved metrics JSON: {metrics_json_path}")
            
            # 3. Save formatted text report
            txt_path = report_dir / "report_formatted.txt"
            self._generate_formatted_report(analysis_result, txt_path)
            file_paths['text_report'] = str(txt_path)
            logger.info(f"✅ Saved text report: {txt_path}")
            
            # 4. Save metrics CSV
            csv_path = report_dir / "metrics_data.csv"
            self._generate_csv_metrics(analysis_result, csv_path)
            file_paths['csv_metrics'] = str(csv_path)
            logger.info(f"✅ Saved CSV metrics: {csv_path}")
            
            # 5. Save executive summary
            summary_path = report_dir / "executive_summary.txt"
            self._generate_executive_summary(analysis_result, summary_path)
            file_paths['executive_summary'] = str(summary_path)
            logger.info(f"✅ Saved executive summary: {summary_path}")
            
            # 6. Save raw response (for debugging)
            raw_path = report_dir / "raw_response.txt"
            with open(raw_path, 'w', encoding='utf-8') as f:
                f.write(analysis_result.get('raw_response', 'No raw response available'))
            file_paths['raw_response'] = str(raw_path)
            logger.info(f"✅ Saved raw response: {raw_path}")
            
            # 7. Create README with file descriptions
            readme_path = report_dir / "README.txt"
            self._generate_readme(file_paths, readme_path)
            file_paths['readme'] = str(readme_path)
            logger.info(f"✅ Saved README: {readme_path}")
            
            logger.info(f"💾 All reports saved to: {report_dir}")
            return file_paths
            
        except Exception as e:
            logger.error(f"❌ Failed to save analysis reports: {e}", exc_info=True)
            return {}
    
    def _generate_formatted_report(self, analysis_result: Dict[str, Any], output_path: Path):
        """Generate a human-readable formatted report from metrics."""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("BEHAVIORAL ANALYSIS REPORT - COMPREHENSIVE METRICS\n")
            f.write("=" * 100 + "\n\n")
            
            # Header info
            f.write(f"Candidate ID: {analysis_result['user_id']}\n")
            f.write(f"Session ID: {analysis_result['session_id']}\n")
            f.write(f"Analysis Date: {analysis_result['analysis_timestamp']}\n")
            f.write(f"Total Snapshots Analyzed: {analysis_result['total_snapshots_analyzed']}\n")
            f.write(f"Interview Duration: ~{analysis_result['metadata']['total_duration_seconds']} seconds\n")
            f.write(f"Analysis Model: {analysis_result['metadata']['model_used']}\n")
            f.write(f"Analysis Version: {analysis_result['metadata'].get('analysis_version', '1.0')}\n\n")
            
            if analysis_result.get('metrics'):
                metrics = analysis_result['metrics']
                
                # Overall Scores
                f.write("=" * 100 + "\n")
                f.write("OVERALL ASSESSMENT\n")
                f.write("=" * 100 + "\n")
                f.write(f"Overall Confidence Score: {metrics.get('overall_confidence_score', 'N/A')}/10\n")
                f.write(f"Interview Readiness Score: {metrics.get('interview_readiness_score', 'N/A')}/10\n")
                f.write(f"Hiring Recommendation: {metrics.get('hiring_recommendation', 'N/A').upper()}\n")
                f.write(f"Hiring Confidence: {metrics.get('hiring_confidence', 'N/A')}/10\n")
                f.write(f"\n{metrics.get('summary', 'N/A')}\n\n")
                
                # Emotion Analysis
                f.write("=" * 100 + "\n")
                f.write("EMOTION ANALYSIS\n")
                f.write("=" * 100 + "\n")
                emotion = metrics.get('emotion_analysis', {})
                f.write(f"Dominant Emotions: {', '.join(emotion.get('dominant_emotions', []))}\n")
                f.write(f"Emotional Stability: {emotion.get('emotional_stability_score', 'N/A')}/10\n")
                f.write(f"Emotional Appropriateness: {emotion.get('emotional_appropriateness_score', 'N/A')}/10\n")
                f.write(f"Anxiety Level: {emotion.get('anxiety_level', 'N/A')}/10 (lower is better)\n")
                f.write(f"Enthusiasm Level: {emotion.get('enthusiasm_level', 'N/A')}/10\n")
                
                timeline = emotion.get('emotion_timeline', {})
                if timeline:
                    f.write(f"\nEmotion Timeline:\n")
                    f.write(f"  Beginning: {timeline.get('beginning', 'N/A')}\n")
                    f.write(f"  Middle: {timeline.get('middle', 'N/A')}\n")
                    f.write(f"  End: {timeline.get('end', 'N/A')}\n")
                f.write("\n")
                
                # Eye Contact
                f.write("=" * 100 + "\n")
                f.write("EYE CONTACT\n")
                f.write("=" * 100 + "\n")
                eye = metrics.get('eye_contact', {})
                f.write(f"Quality Score: {eye.get('quality_score', 'N/A')}/10\n")
                f.write(f"Consistency Score: {eye.get('consistency_score', 'N/A')}/10\n")
                f.write(f"Percentage Maintained: {eye.get('percentage_maintained', 'N/A')}%\n")
                f.write(f"Appropriateness Rating: {eye.get('appropriateness_rating', 'N/A').upper()}\n\n")
                
                # Gaze Behavior
                f.write("=" * 100 + "\n")
                f.write("GAZE BEHAVIOR\n")
                f.write("=" * 100 + "\n")
                gaze = metrics.get('gaze_behavior', {})
                f.write(f"Shift Frequency: {gaze.get('gaze_shift_frequency', 'N/A').upper()}\n")
                f.write(f"Average Shifts Per Minute: {gaze.get('average_shifts_per_minute', 'N/A')}\n")
                f.write(f"Looking Away Percentage: {gaze.get('looking_away_percentage', 'N/A')}%\n")
                f.write(f"Shift Patterns Detected: {', '.join(gaze.get('shift_patterns', []))}\n")
                f.write(f"Dishonesty Indicators: {gaze.get('dishonesty_indicators', 'N/A')}/10 (lower is better)\n\n")
                
                # Body Movement
                f.write("=" * 100 + "\n")
                f.write("BODY MOVEMENT\n")
                f.write("=" * 100 + "\n")
                body = metrics.get('body_movement', {})
                f.write(f"Movement Level: {body.get('movement_level', 'N/A').upper()}\n")
                f.write(f"Fidgeting Score: {body.get('fidgeting_score', 'N/A')}/10 (lower is better)\n")
                f.write(f"Hand Gesture Naturalness: {body.get('hand_gesture_naturalness', 'N/A')}/10\n")
                f.write(f"Self-Soothing Frequency: {body.get('self_soothing_frequency', 'N/A').upper()}\n")
                f.write(f"Distracting Movements: {body.get('distracting_movements_score', 'N/A')}/10 (lower is better)\n\n")
                
                # Posture & Composure
                f.write("=" * 100 + "\n")
                f.write("POSTURE & COMPOSURE\n")
                f.write("=" * 100 + "\n")
                posture = metrics.get('posture_composure', {})
                f.write(f"Posture Score: {posture.get('posture_score', 'N/A')}/10\n")
                f.write(f"Engagement Level: {posture.get('engagement_level', 'N/A')}/10\n")
                f.write(f"Professional Presentation: {posture.get('professional_presentation', 'N/A')}/10\n")
                f.write(f"Posture Consistency: {posture.get('posture_consistency', 'N/A')}/10\n")
                f.write(f"Physical Comfort Score: {posture.get('physical_comfort_score', 'N/A')}/10\n\n")
                
                # Communication Quality
                comm = metrics.get('communication_quality', {})
                if comm:
                    f.write("=" * 100 + "\n")
                    f.write("COMMUNICATION QUALITY\n")
                    f.write("=" * 100 + "\n")
                    f.write(f"Body Language Clarity: {comm.get('body_language_clarity', 'N/A')}/10\n")
                    f.write(f"Non-Verbal Confidence: {comm.get('non_verbal_confidence', 'N/A')}/10\n")
                    f.write(f"Attentiveness Score: {comm.get('attentiveness_score', 'N/A')}/10\n\n")
                
                # Red Flags
                red_flags = metrics.get('red_flags', {})
                if red_flags and red_flags.get('detected'):
                    f.write("=" * 100 + "\n")
                    f.write("⚠️  RED FLAGS DETECTED\n")
                    f.write("=" * 100 + "\n")
                    f.write(f"Count: {red_flags.get('count', 0)}\n")
                    f.write("Descriptions:\n")
                    for flag in red_flags.get('descriptions', []):
                        f.write(f"  • {flag}\n")
                    f.write("\n")
                
                # Positive Indicators
                positive = metrics.get('positive_indicators', {})
                if positive:
                    f.write("=" * 100 + "\n")
                    f.write("✅ POSITIVE INDICATORS\n")
                    f.write("=" * 100 + "\n")
                    f.write(f"Count: {positive.get('count', 0)}\n")
                    f.write("Descriptions:\n")
                    for indicator in positive.get('descriptions', []):
                        f.write(f"  • {indicator}\n")
                    f.write("\n")
                
                # Strengths & Improvements
                f.write("=" * 100 + "\n")
                f.write("KEY OBSERVATIONS\n")
                f.write("=" * 100 + "\n")
                f.write("STRENGTHS:\n")
                for strength in metrics.get('key_strengths', []):
                    f.write(f"  ✓ {strength}\n")
                f.write("\nAREAS FOR IMPROVEMENT:\n")
                for area in metrics.get('areas_for_improvement', []):
                    f.write(f"  → {area}\n")
                f.write("\n")
                
                # Detailed Observations
                detailed = metrics.get('detailed_observations')
                if detailed:
                    f.write("=" * 100 + "\n")
                    f.write("DETAILED OBSERVATIONS\n")
                    f.write("=" * 100 + "\n")
                    f.write(detailed)
                    f.write("\n\n")
                
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
        
        try:
            if not analysis_result.get('metrics'):
                return
            
            metrics = analysis_result['metrics']
            
            # Flatten metrics
            flat_metrics = {
                'user_id': analysis_result['user_id'],
                'session_id': analysis_result['session_id'],
                'analysis_timestamp': analysis_result['analysis_timestamp'],
                'total_snapshots': analysis_result['total_snapshots_analyzed'],
                'duration_seconds': analysis_result['metadata']['total_duration_seconds'],
                'overall_confidence_score': metrics.get('overall_confidence_score'),
                'interview_readiness_score': metrics.get('interview_readiness_score'),
                'hiring_recommendation': metrics.get('hiring_recommendation'),
                'hiring_confidence': metrics.get('hiring_confidence'),
                'emotional_stability': metrics.get('emotion_analysis', {}).get('emotional_stability_score'),
                'emotional_appropriateness': metrics.get('emotion_analysis', {}).get('emotional_appropriateness_score'),
                'anxiety_level': metrics.get('emotion_analysis', {}).get('anxiety_level'),
                'enthusiasm_level': metrics.get('emotion_analysis', {}).get('enthusiasm_level'),
                'eye_contact_quality': metrics.get('eye_contact', {}).get('quality_score'),
                'eye_contact_consistency': metrics.get('eye_contact', {}).get('consistency_score'),
                'eye_contact_percentage': metrics.get('eye_contact', {}).get('percentage_maintained'),
                'gaze_shifts_per_minute': metrics.get('gaze_behavior', {}).get('average_shifts_per_minute'),
                'looking_away_percentage': metrics.get('gaze_behavior', {}).get('looking_away_percentage'),
                'dishonesty_indicators': metrics.get('gaze_behavior', {}).get('dishonesty_indicators'),
                'fidgeting_score': metrics.get('body_movement', {}).get('fidgeting_score'),
                'gesture_naturalness': metrics.get('body_movement', {}).get('hand_gesture_naturalness'),
                'distracting_movements': metrics.get('body_movement', {}).get('distracting_movements_score'),
                'posture_score': metrics.get('posture_composure', {}).get('posture_score'),
                'engagement_level': metrics.get('posture_composure', {}).get('engagement_level'),
                'professional_presentation': metrics.get('posture_composure', {}).get('professional_presentation'),
                'posture_consistency': metrics.get('posture_composure', {}).get('posture_consistency'),
                'physical_comfort': metrics.get('posture_composure', {}).get('physical_comfort_score'),
                'body_language_clarity': metrics.get('communication_quality', {}).get('body_language_clarity'),
                'non_verbal_confidence': metrics.get('communication_quality', {}).get('non_verbal_confidence'),
                'attentiveness_score': metrics.get('communication_quality', {}).get('attentiveness_score'),
                'red_flags_detected': metrics.get('red_flags', {}).get('detected'),
                'red_flags_count': metrics.get('red_flags', {}).get('count'),
                'positive_indicators_count': metrics.get('positive_indicators', {}).get('count')
            }
            
            # Write CSV
            with open(output_path, 'w', encoding='utf-8') as f:
                # Header
                f.write(','.join(flat_metrics.keys()) + '\n')
                # Data
                f.write(','.join(str(v) if v is not None else '' for v in flat_metrics.values()) + '\n')
            
            logger.info(f"✅ CSV metrics saved to {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate CSV metrics: {e}")
    
    def _generate_executive_summary(self, analysis_result: Dict[str, Any], output_path: Path):
        """Generate a concise executive summary."""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("EXECUTIVE SUMMARY - BEHAVIORAL ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Candidate: {analysis_result['user_id']}\n")
            f.write(f"Session: {analysis_result['session_id']}\n")
            f.write(f"Date: {analysis_result['analysis_timestamp']}\n\n")
            
            if analysis_result.get('metrics'):
                metrics = analysis_result['metrics']
                
                f.write("QUICK ASSESSMENT:\n")
                f.write(f"  Overall Confidence: {metrics.get('overall_confidence_score', 'N/A')}/10\n")
                f.write(f"  Interview Readiness: {metrics.get('interview_readiness_score', 'N/A')}/10\n")
                f.write(f"  Recommendation: {metrics.get('hiring_recommendation', 'N/A').upper()}\n")
                f.write(f"  Confidence in Recommendation: {metrics.get('hiring_confidence', 'N/A')}/10\n\n")
                
                f.write("SUMMARY:\n")
                f.write(f"{metrics.get('summary', 'N/A')}\n\n")
                
                f.write("TOP 3 STRENGTHS:\n")
                for i, strength in enumerate(metrics.get('key_strengths', [])[:3], 1):
                    f.write(f"  {i}. {strength}\n")
                
                f.write("\nTOP 3 AREAS FOR IMPROVEMENT:\n")
                for i, area in enumerate(metrics.get('areas_for_improvement', [])[:3], 1):
                    f.write(f"  {i}. {area}\n")
                
                # Red flags if any
                red_flags = metrics.get('red_flags', {})
                if red_flags and red_flags.get('detected'):
                    f.write(f"\n⚠️  RED FLAGS: {red_flags.get('count', 0)} detected\n")
                    for flag in red_flags.get('descriptions', []):
                        f.write(f"  • {flag}\n")
                
                # Key metrics
                f.write("\nKEY METRICS:\n")
                f.write(f"  Anxiety Level: {metrics.get('emotion_analysis', {}).get('anxiety_level', 'N/A')}/10\n")
                f.write(f"  Eye Contact Quality: {metrics.get('eye_contact', {}).get('quality_score', 'N/A')}/10\n")
                f.write(f"  Engagement: {metrics.get('posture_composure', {}).get('engagement_level', 'N/A')}/10\n")
                f.write(f"  Professional Presentation: {metrics.get('posture_composure', {}).get('professional_presentation', 'N/A')}/10\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("For detailed analysis, see report_formatted.txt\n")
            f.write("=" * 80 + "\n")
    
    def _generate_readme(self, file_paths: Dict[str, str], output_path: Path):
        """Generate README explaining all generated files."""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("BEHAVIORAL ANALYSIS - FILE DIRECTORY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("This directory contains the complete behavioral analysis results.\n\n")
            
            f.write("FILES GENERATED:\n\n")
            
            f.write("1. executive_summary.txt\n")
            f.write("   Quick overview with key scores and recommendation\n")
            f.write("   → START HERE for a quick assessment\n\n")
            
            f.write("2. report_formatted.txt\n")
            f.write("   Comprehensive human-readable report with all metrics\n")
            f.write("   → READ THIS for detailed analysis\n\n")
            
            f.write("3. metrics_data.csv\n")
            f.write("   Flattened metrics in spreadsheet format\n")
            f.write("   → IMPORT into Excel/Google Sheets for data analysis\n\n")
            
            f.write("4. analysis_complete.json\n")
            f.write("   Complete analysis data including metadata\n")
            f.write("   → USE THIS for programmatic access to all data\n\n")
            
            f.write("5. metrics_only.json\n")
            f.write("   Simplified JSON with just the metrics\n")
            f.write("   → USE THIS for easier parsing of metrics\n\n")
            
            f.write("6. raw_response.txt\n")
            f.write("   Raw output from AI model\n")
            f.write("   → FOR DEBUGGING only\n\n")
            
            f.write("7. README.txt\n")
            f.write("   This file - explains all generated files\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("METRICS SCORING GUIDE:\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Most metrics use 1-10 scale:\n")
            f.write("  9-10: Excellent/Outstanding\n")
            f.write("  7-8:  Good/Satisfactory\n")
            f.write("  4-6:  Fair/Needs Improvement\n")
            f.write("  1-3:  Poor/Concerning\n\n")
            
            f.write("Exception (lower is better):\n")
            f.write("  - Anxiety Level\n")
            f.write("  - Fidgeting Score\n")
            f.write("  - Dishonesty Indicators\n")
            f.write("  - Distracting Movements\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("HIRING RECOMMENDATIONS:\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("  strong_yes: Highly recommended, no concerns\n")
            f.write("  yes:        Recommended, minor areas to note\n")
            f.write("  maybe:      Conditional, significant areas of concern\n")
            f.write("  no:         Not recommended, multiple red flags\n")
            f.write("  strong_no:  Strongly not recommended, serious concerns\n\n")
            
            f.write("=" * 80 + "\n")
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