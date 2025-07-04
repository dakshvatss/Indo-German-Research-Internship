import re
import csv
import pandas as pd
import os
from enum import Enum
from typing import List, Dict, Optional, Set
import logging


class ProcessingMode(Enum):
    INIT = "INIT"
    PAGE = "PAGE"
    SPEAKER_SCAN = "SPEAKER_SCAN"
    POTENTIAL_SPEAKER = "POTENTIAL_SPEAKER"  
    SPEECH = "SPEECH"

class ParliamentProcessor:
    def __init__(self, delimiters: Set[str] = {';', ':'}):
        # Configurable delimiters
        self.delimiters = delimiters
        
        # State tracking
        self.mode = ProcessingMode.INIT
        self.current_page = ""
        self.current_metadata = ""
        
        # Speaker/Speech tracking
        self.current_speaker = ""
        self.current_speech = []
        self.word_buffer = []
        self.non_speaker_words = 0
        
        # Multi-line speaker handling
        self.potential_speaker_lines = []
        self.line_buffer = []
        
        # Records
        self.records = []
        
        # Session markers
        self.SESSION_MARKERS = [
            "[HON. SPEAKER in the Chair]",
            "The Lok Sabha met at",
            "LOK SABHA DEBATES"
        ]

    def is_bold_text(self, word: str) -> bool:
        """Check if word is in bold (surrounded by **)."""
        return word.startswith("**") and word.endswith("**")
    
    def is_capitalized(self, word: str) -> bool:
        """Check if word is in all capitals."""
        return word.isupper() and len(word) > 1
    
    def is_speaker_formatted(self, word: str) -> bool:
        """Check if word has speaker formatting."""
        return self.is_bold_text(word) or self.is_capitalized(word)
    
    def clean_speaker_text(self, text: str) -> str:
        """Remove formatting from speaker text."""
        words = text.strip().split()
        cleaned_words = []
        for word in words:  
            if self.is_bold_text(word):
                cleaned_words.append(word[2:-2])
            else:
                cleaned_words.append(word)
        return ' '.join(cleaned_words)

    def process_init_mode(self, line: str) -> bool:
        """Process line in INIT mode, looking for session markers."""
        if any(marker in line for marker in self.SESSION_MARKERS):
            self.mode = ProcessingMode.PAGE
            return True
        return False

    def process_page_mode(self, line: str) -> bool:
        """Process line in PAGE mode, looking for page numbers and metadata."""
        line = line.strip()
    
        # Check for page marker
        if line.startswith("Page"):
            match = re.search(r'Page[\s:]*(\d+)', line)
            if match:
                self.current_page = match.group(1)
                self.current_metadata = ""  # Reset metadata for the new page
                self.mode = ProcessingMode.PAGE  # Stay in PAGE mode to capture metadata
                return True

        # Capture metadata: only take the first line after the page marker
        if self.current_page and not self.current_metadata:
            self.current_metadata = line  # Capture the first line as metadata
            self.mode = ProcessingMode.SPEAKER_SCAN  # Transition to SPEAKER_SCAN after metadata
            return True
        
        return False

    def has_delimiter(self, text: str) -> bool:
        """Check if text contains any of the configured delimiters."""
        return any(delimiter in text for delimiter in self.delimiters)

    def split_at_delimiter(self, text: str) -> tuple[str, str]:
        """Split text at the first occurring delimiter."""
        for delimiter in self.delimiters:
            if delimiter in text:
                parts = text.split(delimiter, 1)
                return parts[0].strip(), parts[1].strip()
        return text.strip(), ""

    def is_potential_speaker_start(self, line: str) -> bool:
        """Check if line might be the start of a speaker section."""
        words = line.strip().split()
        if not words:
            return False
            
        # Check first two words if available
        check_words = words[:2] if len(words) >= 2 else words[:1]
        return any(self.is_speaker_formatted(word) for word in check_words)
    
    def has_sufficient_letters(self, text: str) -> bool:
        """Check if text has sufficient letters (6 for English, 4 for Hindi)."""
        english_count = 0
        hindi_count = 0
        
        for char in text:
            # Check for English letters
            if char.isalpha():
                english_count += 1
            # Check for Hindi/Devanagari letters (U+0900 to U+097F)
            elif '\u0900' <= char <= '\u097F':
                hindi_count += 1
        
        # Return True if either English has 6+ letters OR Hindi has 4+ letters
        return english_count >= 6 or hindi_count >= 3

    def validate_speaker(self, text: str) -> Optional[str]:
        """Validate text as a speaker."""
        if not text:
            return None
            
        # Check for any configured delimiter
        if not self.has_delimiter(text):
            return None
            
        speaker_part, _ = self.split_at_delimiter(text)
        
        # Clean the speaker part for validation
        cleaned_speaker = self.clean_speaker_text(speaker_part)
        
        # Check if speaker has at least 6 letters
        if not self.has_sufficient_letters(cleaned_speaker):
            return None
        
        # Validate speaker formatting
        words = speaker_part.split()
        formatted_words = sum(1 for word in words if self.is_speaker_formatted(word))

        if formatted_words == 0 or len(words) - formatted_words > 3:
            return None
            
        return cleaned_speaker
    
    def process_potential_speaker_mode(self, line: str) -> bool:
        """Process line in POTENTIAL_SPEAKER mode."""
        self.potential_speaker_lines.append(line)

        # If we find a delimiter in this line
        if self.has_delimiter(line):
            full_text = ' '.join(self.potential_speaker_lines)
            speaker = self.validate_speaker(full_text)

            if speaker:
                if self.current_speaker:
                    self.save_current_record()
                self.current_speaker = speaker

                # Extract speech part after delimiter
                _, speech_start = self.split_at_delimiter(full_text)
                if speech_start:
                    self.current_speech = [speech_start]

                self.mode = ProcessingMode.SPEECH
                self.potential_speaker_lines = []
                return True
            else:
                # If validation fails, treat accumulated lines as speech
                if self.current_speaker:
                    self.current_speech.extend(self.potential_speaker_lines)
                self.potential_speaker_lines = []
                self.mode = ProcessingMode.SPEECH
                return False

        # Count non-speaker formatted words
        non_speaker = sum(1 for word in line.split() 
                         if not self.is_speaker_formatted(word))
        self.non_speaker_words += non_speaker

        # If too many non-speaker words, treat as speech
        if self.non_speaker_words > 3:
            if self.current_speaker:
                self.current_speech.extend(self.potential_speaker_lines)
            self.potential_speaker_lines = []
            self.non_speaker_words = 0
            self.mode = ProcessingMode.SPEECH
            return False

        return True

    def process_line(self, line: str):
        """Process each line based on current mode."""
        line = line.strip()
        if not line:
            return  # Ignore empty lines

        # Debug log for current mode and line
        logging.debug(f"Mode: {self.mode}, Processing line: {line[:50]}...")

        # Process INIT mode: look for session markers
        if self.mode == ProcessingMode.INIT:
            if self.process_init_mode(line):
                logging.debug("Session marker found. Transitioning to PAGE mode.")
            else:
                logging.debug("No session marker found. Skipping line.")
            return  # Skip any further processing until session markers are found

        # Process PAGE mode: handle page numbers and metadata
        if self.mode == ProcessingMode.PAGE:
            if self.process_page_mode(line):
                logging.debug("Page or metadata processed.")
            return

        # Page marker check takes precedence over other modes
        if line.startswith("Page"):
            if self.process_page_mode(line):
                logging.debug("Page marker detected. Processed in PAGE mode.")
            return

        # Process modes based on current state
        if self.mode == ProcessingMode.SPEAKER_SCAN:
            if self.is_potential_speaker_start(line):
                self.mode = ProcessingMode.POTENTIAL_SPEAKER
                self.potential_speaker_lines = []
                self.non_speaker_words = 0
                self.process_potential_speaker_mode(line)
            else:
                if self.current_speaker:
                    self.current_speech.append(line)
                self.mode = ProcessingMode.SPEECH

        elif self.mode == ProcessingMode.POTENTIAL_SPEAKER:
            self.process_potential_speaker_mode(line)

        elif self.mode == ProcessingMode.SPEECH:
            if self.is_potential_speaker_start(line):
                self.mode = ProcessingMode.POTENTIAL_SPEAKER
                self.potential_speaker_lines = []
                self.non_speaker_words = 0
                self.process_potential_speaker_mode(line)
            else:
                if self.current_speaker:
                    self.current_speech.append(line)

    def save_current_record(self):
        """Save current speaker and speech as a record."""
        if self.current_speaker and self.current_speech:
            self.records.append({
                'page': self.current_page,
                'metadata': self.current_metadata,
                'speaker': self.current_speaker,
                'speech': ' '.join(self.current_speech).strip()
            })
            self.current_speech = []

    def process_file(self, input_file: str):
        """Process input file and generate CSV output."""
        # Generate output filename
        output_file = os.path.splitext(input_file)[0] + '.csv'
        
        with open(input_file, 'r', encoding='utf-8') as file:
            for line in file:
                self.process_line(line)
            
            # Process any remaining potential speaker lines
            if self.potential_speaker_lines:
                if self.current_speaker:
                    self.current_speech.extend(self.potential_speaker_lines)
            
            # Save final record if exists
            if self.current_speaker:
                self.save_current_record()
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(self.records)
        df.to_csv(output_file, index=False, encoding='utf-8')

def main():
    processor = ParliamentProcessor(delimiters={';', ':'})  
    processor.process_file('16-III-01.12.2014.txt')

if __name__ == "__main__":
    main()
