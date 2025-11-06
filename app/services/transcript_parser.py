# app/services/transcript_parser.py
import re
from typing import Dict, Any, List
import logging 

logger = logging.getLogger(__name__) 

class TranscriptParser:
    """Service to parse interview transcript files"""

    @staticmethod
    def parse(transcript_content: str) -> Dict[str, Any]:
        """
        Parse the transcript file to extract metadata and Q&A pairs

        Args:
            transcript_content: Raw transcript file content

        Returns:
            Dictionary containing metadata and qa_pairs, or raises ValueError
        """
        # --- FIX: Check the stripped content directly ---
        if not transcript_content or not transcript_content.strip():
             raise ValueError("Transcript content is empty.")
        # --- END FIX ---
             
        lines = transcript_content.strip().split('\n')

        metadata = TranscriptParser._extract_metadata(lines)
        qa_entries = TranscriptParser._extract_qa_entries(lines)
        paired_qa = TranscriptParser._pair_questions_answers(qa_entries)

        if not metadata.get('session_id') or not metadata.get('user_id'):
             logger.warning("Could not extract essential metadata (session_id or user_id) from transcript header.")

        if not paired_qa:
             logger.warning("No Q&A pairs were extracted from the transcript.")
             # You might still want to raise an error here if pairs are mandatory
             # raise ValueError("No Q&A pairs could be parsed from the transcript.")


        return {
            'metadata': metadata,
            'qa_pairs': paired_qa
        }

    @staticmethod
    def _extract_metadata(lines: List[str]) -> Dict[str, str]:
        """Extract session ID, user ID, and date from transcript header"""
        metadata = {}
        for line in lines[:15]:
            line_lower = line.lower().strip()
            if line_lower.startswith('session id:'):
                metadata['session_id'] = line.split(':', 1)[1].strip()
            elif line_lower.startswith('candidate id:'):
                metadata['user_id'] = line.split(':', 1)[1].strip()
            elif line_lower.startswith('date:'):
                metadata['date'] = line.split(':', 1)[1].strip()
            if 'session_id' in metadata and 'user_id' in metadata and 'date' in metadata:
                 break
        return metadata

    @staticmethod
    def _extract_qa_entries(lines: List[str]) -> List[Dict[str, str]]:
        """Extract all conversation entries with timestamps (Handles potential follow-up tags)"""
        qa_entries = []
        current_timestamp = None
        current_speaker = None
        current_text = ""
        is_follow_up = False 

        pattern = re.compile(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s*(\[[^\]]+\])?\s*(Assistant|User):', re.IGNORECASE)
        follow_up_pattern = re.compile(r'\[Easy Follow-up\]|\[Response to Follow-up\]', re.IGNORECASE)

        for line_num, line in enumerate(lines):
            line = line.strip() 
            if not line: continue 

            timestamp_match = pattern.match(line)

            if timestamp_match:
                if current_timestamp and current_speaker and current_text.strip():
                    qa_entries.append({
                        'timestamp': current_timestamp,
                        'speaker': 'Bot' if current_speaker.lower() == 'assistant' else 'User',
                        'text': current_text.strip(),
                        'is_follow_up': is_follow_up
                    })
                    logger.debug(f"Saved entry: T={current_timestamp}, Spk={current_speaker}, FollowUp={is_follow_up}, Text={current_text.strip()[:50]}...")

                current_timestamp = timestamp_match.group(1)
                tag = timestamp_match.group(2) 
                current_speaker = timestamp_match.group(3)
                current_text = "" 
                is_follow_up = bool(tag and follow_up_pattern.search(tag)) 
                logger.debug(f"New entry started: T={current_timestamp}, Tag={tag}, Spk={current_speaker}, FollowUp={is_follow_up}")


            elif current_speaker:
                if line.strip() != '---':
                    current_text += " " + line.strip()

        if current_timestamp and current_speaker and current_text.strip():
            qa_entries.append({
                'timestamp': current_timestamp,
                'speaker': 'Bot' if current_speaker.lower() == 'assistant' else 'User',
                'text': current_text.strip(),
                'is_follow_up': is_follow_up
            })
            logger.debug(f"Saved final entry: T={current_timestamp}, Spk={current_speaker}, FollowUp={is_follow_up}, Text={current_text.strip()[:50]}...")

        logger.info(f"Extracted {len(qa_entries)} raw conversation entries.")
        return qa_entries

    @staticmethod
    def _pair_questions_answers(qa_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Pair bot questions with subsequent user answers, handling follow-ups."""
        paired_qa = []
        i = 0
        while i < len(qa_entries):
            entry = qa_entries[i]
            if entry['speaker'] == 'Bot':
                question_entry = entry
                answer_entry = None

                if i + 1 < len(qa_entries) and qa_entries[i + 1]['speaker'] == 'User':
                    answer_entry = qa_entries[i + 1]
                    i += 1 

                paired_qa.append({
                    'timestamp': question_entry['timestamp'],
                    'question': question_entry['text'],
                    'is_follow_up_question': question_entry['is_follow_up'],
                    'answer': answer_entry['text'] if answer_entry else "No answer provided or recorded",
                    'is_follow_up_answer': answer_entry['is_follow_up'] if answer_entry else False
                })
            i += 1

        logger.info(f"Paired {len(paired_qa)} Q&A entries.")
        return paired_qa