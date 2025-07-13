import os
import re
from typing import Dict, List, Tuple, Set, Optional, NamedTuple
import logging
from dataclasses import dataclass
from collections import defaultdict
import pandas as pd
from Levenshtein import distance
import enchant
from functools import lru_cache

class BKTreeNode:
    """Node in the BK-Tree for efficient spell checking"""
    def __init__(self, word: str):
        self.word = word
        self.children = {}  # Maps distance to child nodes

class BKTree:
    """BK-Tree implementation for fast approximate string matching"""
    def __init__(self):
        self.root = None
        self.size = 0

    def add(self, word: str):
        """Add a word to the BK-Tree"""
        if self.root is None:
            self.root = BKTreeNode(word)
            self.size = 1
            return

        current = self.root
        while True:
            dist = distance(word, current.word)

            if dist == 0:
                # Word already exists
                return

            if dist not in current.children:
                current.children[dist] = BKTreeNode(word)
                self.size += 1
                return

            current = current.children[dist]

    def search(self, word: str, max_distance: int = 2) -> List[Tuple[str, int]]:
        """Search for words within max_distance of the given word"""
        if self.root is None:
            return []

        results = []
        candidates = [self.root]

        while candidates:
            current = candidates.pop()
            dist = distance(word, current.word)

            if dist <= max_distance:
                results.append((current.word, dist))

            # Add children within the search range
            for child_dist in range(max(1, dist - max_distance), dist + max_distance + 1):
                if child_dist in current.children:
                    candidates.append(current.children[child_dist])

        return sorted(results, key=lambda x: x[1])

    def get_candidates(self, word: str, max_distance: int = 2, max_results: int = 10) -> List[Tuple[str, float]]:
        """Get spelling candidates with normalized distances"""
        matches = self.search(word, max_distance)
        word_len = len(word)

        candidates = []
        for match_word, edit_dist in matches:
            normalized_dist = edit_dist / max(word_len, len(match_word))
            candidates.append((match_word, normalized_dist))

        return candidates[:max_results]

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WordPosition(NamedTuple):
    """Store position information for a word"""
    row: int
    column: str
    word: int
    text: str  # The context text

@dataclass
class ErrorDetail:
    """Detailed information about a specific error"""
    original_word: str
    error_distance: float
    suggested_corrections: List[Tuple[str, float]]  # List of (correction, distance) tuples
    position: WordPosition

@dataclass
class RowAnalysis:
    """Data class to store analysis results for a single row"""
    row_number: int
    column_stats: Dict[str, Dict[str, float]]  # Maps column name to stats
    english_corrections: Dict[str, Tuple[str, WordPosition]]  # Maps incorrect words to (correction, position)
    hindi_corrections: Dict[str, Tuple[str, WordPosition]]  # Maps incorrect words to (correction, position)
    error_details: List[ErrorDetail]  # Detailed error information
    accuracy_percentage: float
    primary_language: str
    total_errors: int

@dataclass
class WordAnalysis:
    """Consolidated word analysis result"""
    is_english: bool
    is_hindi: bool
    primary_language: str  # 'english', 'hindi', 'mixed', or 'unknown'

class DictionaryLoader:
    """Handles dictionary loading for Hindi and initializes PyEnchant for British English"""
    @staticmethod
    def load_dictionary(file_path: str) -> Set[str]:
        """Load Hindi dictionary from file and return as set of words"""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Hindi dictionary file {file_path} not found. Using empty dictionary.")
                return set()

            with open(file_path, 'r', encoding='utf-8') as f:
                dictionary = {word.strip().lower() for word in f if word.strip()}

            return dictionary

        except Exception as e:
            logger.error(f"Error loading Hindi dictionary {file_path}: {str(e)}")
            return set()

    @staticmethod
    def load_english_enchant() -> enchant.Dict:
        """Initialize English spell checker with PyEnchant for British English"""
        try:
            # Try to create British English dictionary first
            enchant_dict = enchant.Dict("en_GB")
            logger.info("Successfully initialized PyEnchant with British English (en_GB)")
            return enchant_dict
        except Exception as e:
            logger.warning(f"Failed to initialize British English (en_GB): {str(e)}")
            try:
                # Fallback to US English
                enchant_dict = enchant.Dict("en_US")
                logger.info("Fallback to US English (en_US) dictionary")
                return enchant_dict
            except Exception as e2:
                logger.error(f"Failed to initialize any English dictionary: {str(e2)}")
                # Try default English
                try:
                    enchant_dict = enchant.Dict("en")
                    logger.info("Using default English dictionary")
                    return enchant_dict
                except Exception as e3:
                    logger.error(f"Failed to initialize default English dictionary: {str(e3)}")
                    raise RuntimeError("Cannot initialize PyEnchant English dictionary")

    @staticmethod
    def load_custom_dictionary(file_path: str) -> Set[str]:
        """Load custom dictionary from file and return as set of words"""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Custom dictionary file {file_path} not found.")
                return set()

            with open(file_path, 'r', encoding='utf-8') as f:
                dictionary = {word.strip().lower() for word in f if word.strip()}

            logger.info(f"Loaded {len(dictionary)} entries from custom dictionary {file_path}")
            return dictionary

        except Exception as e:
            logger.error(f"Error loading custom dictionary {file_path}: {str(e)}")
            return set()

    @staticmethod
    def load_english_enchant_with_custom_dicts(names_dict_path: str = None, honorifics_dict_path: str = None) -> Tuple[enchant.Dict, Set[str]]:
        """Initialize English spell checker with custom dictionaries"""
        try:
            # Initialize PyEnchant dictionary
            enchant_dict = DictionaryLoader.load_english_enchant()

            # Load custom dictionaries if provided
            custom_words = set()

            if names_dict_path and os.path.exists(names_dict_path):
                names_dict = DictionaryLoader.load_custom_dictionary(names_dict_path)
                custom_words.update(names_dict)
                logger.info(f"Added {len(names_dict)} names to custom dictionary")

            if honorifics_dict_path and os.path.exists(honorifics_dict_path):
                honorifics_dict = DictionaryLoader.load_custom_dictionary(honorifics_dict_path)
                custom_words.update(honorifics_dict)
                logger.info(f"Added {len(honorifics_dict)} honorifics to custom dictionary")

            if custom_words:
                logger.info(f"Total custom words to be considered: {len(custom_words)}")

            return enchant_dict, custom_words

        except Exception as e:
            logger.error(f"Error initializing English PyEnchant: {str(e)}")
            raise

class TextAnalyzer:
    """Handles text analysis and error detection"""
    def __init__(self, english_dict: enchant.Dict, hindi_dict: Set[str], 
                names_dict: Set[str] = None, honorifics_dict: Set[str] = None, 
                custom_words: Set[str] = None):
        self.english_dict = english_dict
        self.hindi_dict = hindi_dict
        self.names_dict = names_dict or set()
        self.honorifics_dict = honorifics_dict or set()
        self.custom_words = custom_words or set()

        # Thresholds for OCR errors - more permissive
        self.english_threshold = 0.4
        self.hindi_threshold = 0.5

        # Common prefixes/suffixes that might be incorrectly captured in OCR
        self.common_prefixes = {'hon', 'shri', 'smt', 'dr', 'mr', 'mrs', 'ms', 'श्री', 'श्रीमती'}

        # Add honorifics to common prefixes
        if self.honorifics_dict:
            self.common_prefixes.update(self.honorifics_dict)

        # Initialize BK-Trees for faster spell checking
        logger.info("Building BK-Tree for Hindi dictionary...")
        self.hindi_bktree = BKTree()
        for word in self.hindi_dict:
            self.hindi_bktree.add(word)
        logger.info(f"Hindi BK-Tree built with {self.hindi_bktree.size} words")

        # Build BK-Tree for custom words (names, honorifics)
        logger.info("Building BK-Tree for custom words...")
        self.custom_bktree = BKTree()
        for word in self.custom_words:
            self.custom_bktree.add(word)
        for word in self.names_dict:
            self.custom_bktree.add(word)
        for word in self.honorifics_dict:
            self.custom_bktree.add(word)
        logger.info(f"Custom BK-Tree built with {self.custom_bktree.size} words")

    @lru_cache(maxsize=10000)
    def analyze_word_language(self, word: str) -> WordAnalysis:
        """Consolidated language analysis for a word"""
        word_lower = word.lower()

        # Skip titles and honorifics
        if word_lower in self.common_prefixes:
            return WordAnalysis(is_english=True, is_hindi=False, primary_language='english')

        # Check for English characters
        is_english = all(ord(c) < 128 for c in word) and bool(re.search(r'[a-zA-Z]', word))

        # Check for Hindi characters
        is_hindi = bool(re.search(r'[\u0900-\u097F\u1CD0-\u1CFF\uA8E0-\uA8FF]', word))

        # Determine primary language
        if is_english and not is_hindi:
            primary_language = 'english'
        elif is_hindi and not is_english:
            primary_language = 'hindi'
        elif is_english and is_hindi:
            # Mixed script - determine by character count
            hindi_chars = sum(1 for c in word if '\u0900' <= c <= '\u097F' or '\u1CD0' <= c <= '\u1CFF' or '\uA8E0' <= c <= '\uA8FF')
            english_chars = sum(1 for c in word if 'a' <= c.lower() <= 'z')

            if hindi_chars > english_chars:
                primary_language = 'hindi'
            elif english_chars > hindi_chars:
                primary_language = 'english'
            else:
                primary_language = 'mixed'
        else:
            primary_language = 'unknown'

        return WordAnalysis(is_english=is_english, is_hindi=is_hindi, primary_language=primary_language)

    @lru_cache(maxsize=10000)
    def find_best_hindi_corrections(self, word: str, max_corrections: int = 3) -> List[Tuple[str, float]]:
        """Find best corrections for a Hindi word using BK-Tree for faster lookups"""
        word_len = len(word)

        # Use BK-Tree for faster candidate selection
        # Adjust max_distance based on word length
        if word_len <= 3:
            max_distance = 1
        elif word_len <= 6:
            max_distance = 2
        else:
            max_distance = min(3, word_len // 3)

        # Get candidates from BK-Tree
        corrections = self.hindi_bktree.get_candidates(word, max_distance, max_corrections * 2)

        # If we don't have enough candidates, try with higher distance
        if len(corrections) < max_corrections and max_distance < 3:
            additional_corrections = self.hindi_bktree.get_candidates(word, max_distance + 1, max_corrections * 2)
            corrections.extend(additional_corrections)

            # Remove duplicates and sort
            seen = set()
            unique_corrections = []
            for corr, dist in corrections:
                if corr not in seen:
                    seen.add(corr)
                    unique_corrections.append((corr, dist))

            corrections = sorted(unique_corrections, key=lambda x: x[1])

        return corrections[:max_corrections]

    @lru_cache(maxsize=10000)
    def find_best_english_corrections(self, word: str, max_corrections: int = 3) -> List[Tuple[str, float]]:
        """Find best corrections for an English word using PyEnchant and BK-Tree for custom words"""
        word_len = len(word)
        word_lower = word.lower()
        all_corrections = []

        try:
            # Get suggestions from PyEnchant
            suggestions = self.english_dict.suggest(word_lower)

            if not suggestions:
                # If no suggestions, try with original case
                suggestions = self.english_dict.suggest(word)

            # Calculate normalized edit distances for PyEnchant suggestions
            for suggestion in suggestions:
                edit_distance = distance(word_lower, suggestion.lower())
                normalized_distance = edit_distance / max(word_len, len(suggestion))
                all_corrections.append((suggestion, normalized_distance))

        except Exception as e:
            logger.warning(f"PyEnchant error for word '{word}': {e}")

        # Also check custom words using BK-Tree
        if self.custom_bktree.size > 0:
            max_distance = min(2, word_len // 2) if word_len > 3 else 1
            custom_corrections = self.custom_bktree.get_candidates(word, max_distance, max_corrections)
            all_corrections.extend(custom_corrections)

        # Remove duplicates and sort
        seen = set()
        unique_corrections = []
        for corr, dist in all_corrections:
            corr_lower = corr.lower()
            if corr_lower not in seen:
                seen.add(corr_lower)
                unique_corrections.append((corr, dist))

        unique_corrections.sort(key=lambda x: x[1])
        return unique_corrections[:max_corrections]

    @lru_cache(maxsize=10000)
    def check_english_word(self, word: str) -> Tuple[bool, Optional[str], float, List[Tuple[str, float]]]:
        """Check if an English word is an error using PyEnchant and custom dictionaries"""
        # Skip very short words, numbers, and special chars
        word_lower = word.lower()

        if len(word) <= 2 or any(c.isdigit() for c in word) or not re.search(r'[a-zA-Z]', word):
            return False, None, 0.0, []

        # Check common honorifics/titles and custom dictionaries
        if (word_lower in self.common_prefixes or 
            word_lower in self.honorifics_dict or 
            word_lower in self.names_dict or
            word_lower in self.custom_words):
            return False, None, 0.0, []

        # Additional check for potential names (words starting with capital letter)
        if word[0].isupper() and word_lower in self.names_dict:
            return False, None, 0.0, []

        # Check if word exists in PyEnchant dictionary
        try:
            if self.english_dict.check(word_lower) or self.english_dict.check(word):
                return False, None, 0.0, []
        except Exception as e:
            logger.warning(f"PyEnchant check error for word '{word}': {e}")

        # For capitalized words that aren't in dictionary but could be names
        if word[0].isupper() and len(word) > 2:
            # More lenient with potential proper names
            # Could be a name not in our dictionary
            return False, None, 0.0, []

        # Find best corrections
        corrections = self.find_best_english_corrections(word)

        if not corrections:
            # Could be a proper noun or new word
            if word[0].isupper() and len(word) > 3:
                return False, None, 0.0, []
            # If no corrections found, mark as error with no suggestions
            return True, None, 1.0, []

        best_correction, best_distance = corrections[0]

        # Adjust threshold based on word length - be more lenient with shorter words
        word_len = len(word)
        if word_len <= 4:
            threshold = self.english_threshold * 1.5  # Much more lenient (0.6)
        elif word_len <= 6:
            threshold = self.english_threshold * 1.25  # More lenient (0.5)
        else:
            threshold = self.english_threshold  # Standard threshold (0.4)

        # OCR errors are often subtle - use normalized distance
        if best_distance <= threshold:
            # This is likely a minor OCR error, not a real error
            return True, best_correction, best_distance, corrections
        else:
            # This could be a proper noun, made-up word, or severe error
            if word[0].isupper() and word_len > 3:
                # Proper nouns are likely correct
                return False, None, best_distance, corrections
            return True, best_correction, best_distance, corrections

    @lru_cache(maxsize=10000)
    def check_hindi_word(self, word: str) -> Tuple[bool, Optional[str], float, List[Tuple[str, float]]]:
        """Check if a Hindi word is misspelled using dictionary lookup and edit distance"""
        # Skip very short words and non-Hindi characters
        word_analysis = self.analyze_word_language(word)
        if len(word) <= 2 or not word_analysis.is_hindi:
            return False, None, 0.0, []

        # If the word is in dictionary, it's correct
        if word.lower() in self.hindi_dict:
            return False, None, 0.0, []

        # Find best corrections
        corrections = self.find_best_hindi_corrections(word)

        if not corrections:
            # No correction found - could be a proper name or new word
            return True, None, 1.0, []

        best_correction, best_distance = corrections[0]

        # Adjust threshold based on word length - be more permissive with Hindi
        word_len = len(word)
        if word_len <= 4:
            threshold = self.hindi_threshold * 1.5  # Much more lenient (0.75)
        elif word_len <= 6:
            threshold = self.hindi_threshold * 1.25  # More lenient (0.625)
        else:
            threshold = self.hindi_threshold  # Standard threshold (0.5)

        # Apply the threshold
        if best_distance <= threshold:
            # This is likely a minor OCR error, not a real error
            return True, best_correction, best_distance, corrections
        else:
            # This is likely a proper name or word missing from dictionary
            return True, best_correction, best_distance, corrections

    def create_error_detail(self, word: str, error_distance: float, corrections: List[Tuple[str, float]], 
                            position: WordPosition) -> ErrorDetail:
        """Create error detail object - centralized to avoid duplication"""
        return ErrorDetail(
            original_word=word,
            error_distance=error_distance,
            suggested_corrections=corrections,
            position=position
        )

    def calculate_cell_accuracy(self, text: str, row_num: int, column_name: str) -> Tuple[int, int, float, Dict[str, Tuple[str, WordPosition]], int, int, float, Dict[str, Tuple[str, WordPosition]], List[ErrorDetail]]:
        """Calculate error percentage for both English and Hindi text with position tracking and detailed error info"""
        if pd.isna(text) or text == '':
            return 0, 0, 0.0, {}, 0, 0, 0.0, {}, []

        text = str(text)  # Ensure text is a string

        # Improved word extraction - handle punctuation and special characters better
        words = re.findall(r'\b[\w\u0900-\u097F\u1CD0-\u1CFF\uA8E0-\uA8FF]+\b', text)

        eng_total = 0
        hin_total = 0
        eng_errors = 0
        hin_errors = 0
        eng_corrections = {}
        hin_corrections = {}
        error_details = []

        # Process words in this cell
        for word_num, word in enumerate(words, 1):
            # Skip very short words and filter out garbage
            if len(word) <= 1 or not re.search(r'[a-zA-Z\u0900-\u097F]', word):
                continue

            # Get context (the whole cell content serves as context for CSV)
            context = text[:100] + "..." if len(text) > 100 else text
            position = WordPosition(row=row_num, column=column_name, word=word_num, text=context)

            # Analyze word language once
            word_analysis = self.analyze_word_language(word)

            # Process based on primary language
            if word_analysis.primary_language == 'english':
                eng_total += 1
                is_error, correction, error_distance, all_corrections = self.check_english_word(word)
                if is_error and error_distance > 0.01:  # Filter out minor errors
                    eng_errors += 1
                    if correction:
                        eng_corrections[word.lower()] = (correction, position)
                    # Only include in error_details if distance is less than 0.167 for English
                    if error_distance < 0.167:
                        error_details.append(self.create_error_detail(word, error_distance, all_corrections, position))

            elif word_analysis.primary_language == 'hindi':
                hin_total += 1
                is_error, correction, error_distance, all_corrections = self.check_hindi_word(word)
                if is_error and error_distance > 0.01:  # Filter out minor errors
                    hin_errors += 1
                    if correction:
                        hin_corrections[word] = (correction, position)
                    # Only include in error_details if distance is less than 0.35 for Hindi
                    if error_distance < 0.35:
                        error_details.append(self.create_error_detail(word, error_distance, all_corrections, position))

            elif word_analysis.primary_language == 'mixed':
                # For mixed words, determine which language to use based on character count
                if word_analysis.is_hindi and word_analysis.is_english:
                    hindi_chars = sum(1 for c in word if '\u0900' <= c <= '\u097F' or '\u1CD0' <= c <= '\u1CFF' or '\uA8E0' <= c <= '\uA8FF')
                    english_chars = sum(1 for c in word if 'a' <= c.lower() <= 'z')

                    if hindi_chars > english_chars:
                        hin_total += 1
                        is_error, correction, error_distance, all_corrections = self.check_hindi_word(word)
                        if is_error and error_distance > 0.01:
                            hin_errors += 1
                            if correction:
                                hin_corrections[word] = (correction, position)
                            # Only include in error_details if distance is less than 0.35 for Hindi
                            if error_distance < 0.35:
                                error_details.append(self.create_error_detail(word, error_distance, all_corrections, position))
                    else:
                        eng_total += 1
                        is_error, correction, error_distance, all_corrections = self.check_english_word(word)
                        if is_error and error_distance > 0.01:
                            eng_errors += 1
                            if correction:
                                eng_corrections[word.lower()] = (correction, position)
                            # Only include in error_details if distance is less than 0.167 for English
                            if error_distance < 0.167:
                                error_details.append(self.create_error_detail(word, error_distance, all_corrections, position))

        # Calculate error percentages
        eng_error_percentage = (eng_errors / eng_total * 100) if eng_total > 0 else 0.0
        hin_error_percentage = (hin_errors / hin_total * 100) if hin_total > 0 else 0.0

        # Calculate accuracy percentages (100 - error percentage)
        eng_accuracy = 100 - eng_error_percentage
        hin_accuracy = 100 - hin_error_percentage

        return eng_total, eng_errors, eng_accuracy, eng_corrections, hin_total, hin_errors, hin_accuracy, hin_corrections, error_details

class CSVErrorDetector:
    """Main class for CSV OCR error detection"""
    def __init__(self, english_dict_path: str = None, hindi_dict_path: str = "hi_IN.dic",
                names_dict_path: str = None, honorifics_dict_path: str = None,
                target_column: str = "speech"):
        self.dict_loader = DictionaryLoader()
        self.target_column = target_column

        # Initialize English PyEnchant with custom dictionaries
        self.english_dict, self.custom_words = self.dict_loader.load_english_enchant_with_custom_dicts(
            names_dict_path, honorifics_dict_path
        )

        # Load Hindi dictionary
        self.hindi_dict = self.dict_loader.load_dictionary(hindi_dict_path)

        # Load custom dictionaries for direct access in the analyzer
        self.names_dict = self.dict_loader.load_custom_dictionary(names_dict_path) if names_dict_path else set()
        self.honorifics_dict = self.dict_loader.load_custom_dictionary(honorifics_dict_path) if honorifics_dict_path else set()

        # Initialize analyzer with English PyEnchant, Hindi dictionary and custom dictionaries
        self.analyzer = TextAnalyzer(
            self.english_dict, 
            self.hindi_dict,
            self.names_dict,
            self.honorifics_dict,
            self.custom_words
        )

        # Track all errors across rows
        self.all_english_errors = {}
        self.all_hindi_errors = {}

    def determine_primary_language(self, text: str) -> str:
        """Determine the primary language of the text"""
        if pd.isna(text) or text == '':
            return "Unknown"

        text = str(text)
        words = re.findall(r'\b[\w\u0900-\u097F\u1CD0-\u1CFF\uA8E0-\uA8FF]+\b', text)

        english_count = 0
        hindi_count = 0

        for word in words:
            word_analysis = self.analyzer.analyze_word_language(word)
            if word_analysis.primary_language == 'english':
                english_count += 1
            elif word_analysis.primary_language == 'hindi':
                hindi_count += 1

        if english_count > hindi_count * 1.5:
            return "English"
        elif hindi_count > english_count * 1.5:
            return "Hindi"
        elif english_count > 0 or hindi_count > 0:
            return "Mixed"
        else:
            return "Unknown"

    def process_csv(self, input_csv_path: str, output_csv_path: str, output_report_path: str = None) -> None:
        """Process input CSV file and generate enhanced output with error analysis"""
        try:
            if not os.path.exists(input_csv_path):
                raise FileNotFoundError(f"Input file not found: {input_csv_path}")

            # Load CSV data
            df = pd.read_csv(input_csv_path)

            # Analyze CSV data
            analysis_results = self._analyze_csv(df, verbose=True)

            # Generate overall CSV statistics
            csv_stats = self._calculate_csv_stats(analysis_results)

            # Create enhanced DataFrame with additional columns
            enhanced_df = self._create_enhanced_dataframe(df, analysis_results)

            # Save enhanced CSV
            enhanced_df.to_csv(output_csv_path, index=False, encoding='utf-8')

            # Write analysis report if path is provided
            if output_report_path:
                self._write_results(output_report_path, df, analysis_results, csv_stats)

            logger.info(f"Analysis completed successfully. Enhanced CSV: {output_csv_path}")

        except Exception as e:
            logger.error(f"Error processing CSV file: {str(e)}")
            raise

    def _create_enhanced_dataframe(self, df: pd.DataFrame, results: List[RowAnalysis]) -> pd.DataFrame:
        """Create enhanced DataFrame with additional accuracy and error columns"""
        # Start with original DataFrame
        enhanced_df = df.copy()

        # Add new columns specifically for speech column analysis
        speech_accuracy_percentages = []
        speech_primary_languages = []
        speech_error_counts = []
        speech_error_details_cols = []

        for result in results:
            speech_accuracy_percentages.append(round(result.accuracy_percentage, 2))
            speech_primary_languages.append(result.primary_language)
            speech_error_counts.append(result.total_errors)

            # Format error details for CSV
            error_details_str = self._format_error_details_for_csv(result.error_details)
            speech_error_details_cols.append(error_details_str)

        # Add the new columns to DataFrame with clear naming
        enhanced_df['Speech_Accuracy_Percentage'] = speech_accuracy_percentages
        enhanced_df['Speech_Primary_Language'] = speech_primary_languages
        enhanced_df['Speech_Number_of_Errors'] = speech_error_counts
        enhanced_df['Speech_Error_Details'] = speech_error_details_cols

        return enhanced_df

    def _format_error_details_for_csv(self, error_details: List[ErrorDetail]) -> str:
        """Format error details as a string for CSV storage"""
        if not error_details:
            return ""

        formatted_errors = []
        for error in error_details:
            # Get first two suggestions
            suggestions = error.suggested_corrections[:2]
            if len(suggestions) >= 2:
                first_sugg, first_dist = suggestions[0]
                second_sugg, second_dist = suggestions[1]
                error_str = f"[{error.original_word}|{error.error_distance:.3f}|{first_sugg}|{second_sugg}]"
            elif len(suggestions) == 1:
                first_sugg, first_dist = suggestions[0]
                error_str = f"[{error.original_word}|{error.error_distance:.3f}|{first_sugg}|N/A]"
            else:
                error_str = f"[{error.original_word}|{error.error_distance:.3f}|N/A|N/A]"

            formatted_errors.append(error_str)

        return "; ".join(formatted_errors[:10])  # Limit to 10 errors to keep CSV cells manageable

    def _print_progress(self, current_row: int, total_rows: int, row_num: int) -> None:
        """Print progress information to terminal"""
        progress_percentage = (current_row / total_rows) * 100
        print(f"\rProcessing row {row_num} [{current_row}/{total_rows}] ({progress_percentage:.1f}%)", end='', flush=True)

        # Print newline every 10 rows for better readability
        if current_row % 10 == 0:
            print()

    def _analyze_csv(self, df: pd.DataFrame, verbose: bool = False) -> List[RowAnalysis]:
        """Analyze each row of the CSV and return results - focusing only on 'speech' column"""
        results = []
        total_rows = len(df)

        # Check if target column exists
        if self.target_column not in df.columns:
            logger.warning(f"'{self.target_column}' column not found in CSV. Available columns: " + ", ".join(df.columns))
            raise ValueError(f"'{self.target_column}' column not found in the CSV file")

        if verbose:
            print(f"\nStarting analysis of {total_rows} rows...")
            print("=" * 60)

        # Analyze each row
        for row_idx, row in df.iterrows():
            current_row = row_idx + 1
            row_num = row_idx + 2  # Account for 1-based indexing and header row

            # Print progress if verbose
            if verbose:
                self._print_progress(current_row, total_rows, row_num)

            column_stats = {}
            all_eng_corrections = {}
            all_hin_corrections = {}
            all_error_details = []

            # Only analyze the target column
            col_name = self.target_column
            cell_value = row[col_name]

            # Determine primary language based on speech column only
            primary_language = self.determine_primary_language(str(cell_value) if pd.notna(cell_value) else "")

            total_words_in_row = 0
            total_errors_in_row = 0

            # Skip analysis if cell is empty
            if pd.isna(cell_value) or cell_value == '':
                column_stats[col_name] = {
                    'english_total': 0,
                    'english_errors': 0,
                    'english_accuracy': 100.0,
                    'hindi_total': 0,
                    'hindi_errors': 0,
                    'hindi_accuracy': 100.0,
                    'total_words': 0,
                    'overall_accuracy': 100.0
                }
            else:
                # Analyze speech column content
                eng_total, eng_errors, eng_accuracy, eng_corrections, \
                hin_total, hin_errors, hin_accuracy, hin_corrections, error_details = self.analyzer.calculate_cell_accuracy(
                    str(cell_value), row_num, col_name
                )

                # Update global error dictionaries
                for word, (correction, _) in eng_corrections.items():
                    self.all_english_errors[word] = correction
                    all_eng_corrections[word] = (correction, WordPosition(
                        row=row_num, column=col_name, word=0, text=str(cell_value)[:100]
                    ))

                for word, (correction, _) in hin_corrections.items():
                    self.all_hindi_errors[word] = correction
                    all_hin_corrections[word] = (correction, WordPosition(
                        row=row_num, column=col_name, word=0, text=str(cell_value)[:100]
                    ))

                # Collect error details
                all_error_details.extend(error_details)

                # Calculate overall cell accuracy
                total_words = eng_total + hin_total
                total_errors = eng_errors + hin_errors
                overall_accuracy = 100 - (total_errors / total_words * 100) if total_words > 0 else 100.0

                # Update row totals
                total_words_in_row += total_words
                total_errors_in_row += total_errors

                # Store column stats
                column_stats[col_name] = {
                    'english_total': eng_total,
                    'english_errors': eng_errors,
                    'english_accuracy': eng_accuracy,
                    'hindi_total': hin_total,
                    'hindi_errors': hin_errors,
                    'hindi_accuracy': hin_accuracy,
                    'total_words': total_words,
                    'overall_accuracy': overall_accuracy
                }

            # Calculate row-wise accuracy
            row_accuracy = 100 - (total_errors_in_row / total_words_in_row * 100) if total_words_in_row > 0 else 100.0

            # Create row analysis object
            row_analysis = RowAnalysis(
                row_number=row_num,
                column_stats=column_stats,
                english_corrections=all_eng_corrections,
                hindi_corrections=all_hin_corrections,
                error_details=all_error_details,
                accuracy_percentage=row_accuracy,
                primary_language=primary_language,
                total_errors=total_errors_in_row
            )

            results.append(row_analysis)

        # Print completion message if verbose
        if verbose:
            print(f"\n\nAnalysis completed! Processed {total_rows} rows.")
            print("=" * 60)

        return results

    def _get_most_frequent_errors(self, error_dict: Dict[str, str], limit: int = 10) -> List[Tuple[str, str]]:
        """Return most frequent errors"""
        # Count error occurrences
        error_counts = defaultdict(int)
        for word in error_dict:
            error_counts[word] += 1

        # Sort by frequency
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)

        # Get top errors with their corrections
        top_errors = []
        for word, count in sorted_errors[:limit]:
            correction = error_dict.get(word, "")
            top_errors.append((word, correction, count))

        return top_errors

    def _write_results(self, output_path: str, df: pd.DataFrame, results: List[RowAnalysis], csv_stats: Dict) -> None:
        """Write detailed analysis results to output file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write report header
                f.write("# OCR Error Analysis Report\n\n")
                f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # Write overall statistics
                f.write("## Overall Statistics\n\n")
                f.write(f"Total rows analyzed: {csv_stats['total_rows']}\n")
                f.write(f"Total words: {csv_stats['total_words']}\n")
                f.write(f"Total errors detected: {csv_stats['total_errors']}\n")
                f.write(f"Overall accuracy: {csv_stats['overall_accuracy']:.2f}%\n\n")

                # Language statistics
                f.write("## Language Statistics\n\n")
                f.write(f"English words: {csv_stats['total_english_words']} (Accuracy: {csv_stats['english_accuracy']:.2f}%)\n")
                f.write(f"Hindi words: {csv_stats['total_hindi_words']} (Accuracy: {csv_stats['hindi_accuracy']:.2f}%)\n\n")

                # Language distribution
                f.write("## Row Language Distribution\n\n")
                for lang, count in csv_stats['language_distribution'].items():
                    f.write(f"{lang}: {count} rows ({count/csv_stats['total_rows']*100:.1f}%)\n")
                f.write("\n")

                # Column statistics 
                f.write("## Column-wise Statistics\n\n")
                for col_name, stats in csv_stats['column_stats'].items():
                    f.write(f"### Column: {col_name}\n")
                    f.write(f"Total words: {stats['total_words']}\n")
                    f.write(f"English words: {stats['english_total']} (Errors: {stats['english_errors']}, Accuracy: {stats['english_accuracy']:.2f}%)\n")
                    f.write(f"Hindi words: {stats['hindi_total']} (Errors: {stats['hindi_errors']}, Accuracy: {stats['hindi_accuracy']:.2f}%)\n")
                    f.write(f"Overall accuracy: {stats['overall_accuracy']:.2f}%\n\n")

                # Most frequent errors
                f.write("## Most Frequent English Errors\n\n")
                for word, correction, count in csv_stats['most_frequent_english_errors']:
                    f.write(f"* '{word}' → '{correction}' (Found {count} times)\n")
                f.write("\n")

                f.write("## Most Frequent Hindi Errors\n\n")
                for word, correction, count in csv_stats['most_frequent_hindi_errors']:
                    f.write(f"* '{word}' → '{correction}' (Found {count} times)\n")
                f.write("\n")

                # Row-by-row analysis (limited to rows with errors)
                f.write("## Detailed Row Analysis\n\n")
                for result in sorted(results, key=lambda x: x.accuracy_percentage):
                    # Only show rows with errors
                    if result.total_errors > 0:
                        f.write(f"### Row {result.row_number} (Accuracy: {result.accuracy_percentage:.2f}%, Language: {result.primary_language})\n\n")

                        # List all errors in the row
                        if result.error_details:
                            f.write("Errors detected:\n\n")
                            for error in result.error_details:
                                best_suggestion = error.suggested_corrections[0][0] if error.suggested_corrections else "N/A"
                                context = error.position.text
                                if len(context) > 50:
                                    context = f"{context[:25]}...{context[-25:]}"
                                f.write(f"* '{error.original_word}' → '{best_suggestion}' (Column: {error.position.column}, Context: '{context}')\n")
                            f.write("\n")

                logger.info(f"Analysis report written to {output_path}")

        except Exception as e:
            logger.error(f"Error writing analysis report: {str(e)}")

    def _calculate_csv_stats(self, results: List[RowAnalysis]) -> Dict:
        """Calculate overall statistics for the CSV"""
        # Initialize counters
        total_english_words = 0
        total_english_errors = 0
        total_hindi_words = 0
        total_hindi_errors = 0

        # For column-wise statistics
        column_stats = defaultdict(lambda: {
            'english_total': 0,
            'english_errors': 0,
            'hindi_total': 0,
            'hindi_errors': 0,
            'total_words': 0,
            'total_errors': 0
        })

        # Count words and errors for overall statistics
        for result in results:
            for col_name, stats in result.column_stats.items():
                total_english_words += stats['english_total']
                total_english_errors += stats['english_errors']
                total_hindi_words += stats['hindi_total']
                total_hindi_errors += stats['hindi_errors']

                # Update column-wise statistics
                column_stats[col_name]['english_total'] += stats['english_total']
                column_stats[col_name]['english_errors'] += stats['english_errors']
                column_stats[col_name]['hindi_total'] += stats['hindi_total']
                column_stats[col_name]['hindi_errors'] += stats['hindi_errors']
                column_stats[col_name]['total_words'] += stats['total_words']
                column_stats[col_name]['total_errors'] += (stats['english_errors'] + stats['hindi_errors'])

        # Calculate overall accuracy
        english_accuracy = 100 - (total_english_errors / total_english_words * 100) if total_english_words > 0 else 100.0
        hindi_accuracy = 100 - (total_hindi_errors / total_hindi_words * 100) if total_hindi_words > 0 else 100.0
        total_words = total_english_words + total_hindi_words
        total_errors = total_english_errors + total_hindi_errors
        overall_accuracy = 100 - (total_errors / total_words * 100) if total_words > 0 else 100.0

        # Calculate column-wise accuracy
        for col_name in column_stats:
            col = column_stats[col_name]
            col['english_accuracy'] = 100 - (col['english_errors'] / col['english_total'] * 100) if col['english_total'] > 0 else 100.0
            col['hindi_accuracy'] = 100 - (col['hindi_errors'] / col['hindi_total'] * 100) if col['hindi_total'] > 0 else 100.0
            col['overall_accuracy'] = 100 - (col['total_errors'] / col['total_words'] * 100) if col['total_words'] > 0 else 100.0

        # Count language distributions
        language_counts = {
            'English': sum(1 for r in results if r.primary_language == 'English'),
            'Hindi': sum(1 for r in results if r.primary_language == 'Hindi'),
            'Mixed': sum(1 for r in results if r.primary_language == 'Mixed'),
            'Unknown': sum(1 for r in results if r.primary_language == 'Unknown')
        }

        # Return combined statistics
        return {
            'total_rows': len(results),
            'total_english_words': total_english_words,
            'total_english_errors': total_english_errors,
            'english_accuracy': english_accuracy,
            'total_hindi_words': total_hindi_words,
            'total_hindi_errors': total_hindi_errors,
            'hindi_accuracy': hindi_accuracy,
            'total_words': total_words,
            'total_errors': total_errors,
            'overall_accuracy': overall_accuracy,
            'column_stats': dict(column_stats),
            'language_distribution': language_counts,
            'most_frequent_english_errors': self._get_most_frequent_errors(self.all_english_errors, 20),
            'most_frequent_hindi_errors': self._get_most_frequent_errors(self.all_hindi_errors, 20)
        }

    def generate_corrected_csv(self, input_csv_path: str, corrected_csv_path: str, confidence_threshold: float = 0.3) -> None:
        """
        Generate a corrected CSV file by applying suggested corrections to detected errors.

        Args:
            input_csv_path (str): Path to the input CSV file
            corrected_csv_path (str): Path where the corrected CSV will be saved
            confidence_threshold (float): Only apply corrections with normalized distance <= this threshold
        """
        try:
            if not os.path.exists(input_csv_path):
                raise FileNotFoundError(f"Input file not found: {input_csv_path}")

            # Load CSV data
            df = pd.read_csv(input_csv_path)

            # Check if target column exists
            if self.target_column not in df.columns:
                logger.warning(f"'{self.target_column}' column not found in CSV. Available columns: " + ", ".join(df.columns))
                raise ValueError(f"'{self.target_column}' column not found in the CSV file")

            # Create a copy of the dataframe for corrections
            corrected_df = df.copy()

            # Track correction statistics
            total_corrections = 0
            corrections_applied = {}

            # Process each row
            for row_idx, row in df.iterrows():
                row_num = row_idx + 2  # Account for 1-based indexing and header row
                cell_value = row[self.target_column]

                # Skip if cell is empty
                if pd.isna(cell_value) or cell_value == '':
                    continue

                # Get original text
                original_text = str(cell_value)
                corrected_text = original_text

                # Analyze the cell to get error details
                _, _, _, eng_corrections, _, _, _, hin_corrections, error_details = self.analyzer.calculate_cell_accuracy(
                    original_text, row_num, self.target_column
                )

                # Apply corrections based on confidence threshold
                words_corrected_in_row = []

                # Apply English corrections
                for original_word, (correction, position) in eng_corrections.items():
                    # Find the best correction from error details
                    word_error_detail = None
                    for error in error_details:
                        if error.original_word.lower() == original_word.lower():
                            word_error_detail = error
                            break

                    if word_error_detail and word_error_detail.error_distance <= confidence_threshold:
                        # Use word boundaries to avoid partial matches
                        pattern = r'\b' + re.escape(word_error_detail.original_word) + r'\b'
                        if re.search(pattern, corrected_text, re.IGNORECASE):
                            corrected_text = re.sub(pattern, correction, corrected_text, count=1, flags=re.IGNORECASE)
                            words_corrected_in_row.append(f"{word_error_detail.original_word} → {correction}")
                            total_corrections += 1

                # Apply Hindi corrections
                for original_word, (correction, position) in hin_corrections.items():
                    # Find the best correction from error details
                    word_error_detail = None
                    for error in error_details:
                        if error.original_word == original_word:
                            word_error_detail = error
                            break

                    if word_error_detail and word_error_detail.error_distance <= confidence_threshold:
                        # Use word boundaries for Hindi text too
                        pattern = r'\b' + re.escape(word_error_detail.original_word) + r'\b'
                        if re.search(pattern, corrected_text):
                            corrected_text = re.sub(pattern, correction, corrected_text, count=1)
                            words_corrected_in_row.append(f"{word_error_detail.original_word} → {correction}")
                            total_corrections += 1

                # Update the corrected dataframe
                corrected_df.at[row_idx, self.target_column] = corrected_text

                # Track corrections for this row
                if words_corrected_in_row:
                    corrections_applied[row_num] = words_corrected_in_row

            # Add metadata columns to show what was corrected
            correction_summary = []
            for row_idx in range(len(df)):
                row_num = row_idx + 2
                if row_num in corrections_applied:
                    correction_summary.append("; ".join(corrections_applied[row_num]))
                else:
                    correction_summary.append("")

            corrected_df['Corrections_Applied'] = correction_summary

            # Save corrected CSV
            corrected_df.to_csv(corrected_csv_path, index=False, encoding='utf-8')

            logger.info(f"Corrected CSV saved to: {corrected_csv_path}")
            logger.info(f"Total corrections applied: {total_corrections}")
            logger.info(f"Rows with corrections: {len(corrections_applied)}")

            # Print summary of corrections
            print(f"\n=== CORRECTION SUMMARY ===")
            print(f"Total corrections applied: {total_corrections}")
            print(f"Rows with corrections: {len(corrections_applied)}")
            print(f"Confidence threshold used: {confidence_threshold}")

            if corrections_applied:
                print(f"\nFirst 10 rows with corrections:")
                for i, (row_num, corrections) in enumerate(list(corrections_applied.items())[:10]):
                    print(f"Row {row_num}: {'; '.join(corrections[:3])}{'...' if len(corrections) > 3 else ''}")

        except Exception as e:
            logger.error(f"Error generating corrected CSV: {str(e)}")
            raise

    def process_csv_with_corrections(self, input_csv_path: str, output_csv_path: str, 
                                    corrected_csv_path: str, output_report_path: str = None,
                                    confidence_threshold: float = 0.3) -> None:
        """
        Enhanced process_csv method that also generates a corrected version

        Args:
            input_csv_path (str): Path to input CSV
            output_csv_path (str): Path for enhanced CSV with analysis
            corrected_csv_path (str): Path for corrected CSV with fixes applied
            output_report_path (str, optional): Path for analysis report
            confidence_threshold (float): Confidence threshold for applying corrections
        """
        try:
            # First, run the original analysis
            self.process_csv(input_csv_path, output_csv_path, output_report_path)

            # Then generate the corrected version
            self.generate_corrected_csv(input_csv_path, corrected_csv_path, confidence_threshold)

            logger.info(f"Complete analysis finished:")
            logger.info(f"- Enhanced CSV (with analysis): {output_csv_path}")
            logger.info(f"- Corrected CSV (with fixes): {corrected_csv_path}")
            if output_report_path:
                logger.info(f"- Analysis report: {output_report_path}")

        except Exception as e:
            logger.error(f"Error in complete CSV processing: {str(e)}")
            raise

def main():
    input_path = "lsd_16_04_2015-03-03.csv"
    output_csv_path = "final_2.csv"
    output_report_path = "final_2report.txt"
    hindi_dict_path = "hi_INsmall.dic"
    names_dict_path = "Names.txt"
    honorifics_dict_path = "honorifics.txt"
    target_column = "speech"  # Specify which column to analyze

    print("🚀 Starting CSV Error Detection Analysis...")
    print(f"📁 Input file: {input_path}")
    print(f"📊 Target column: {target_column}")
    print(f"📝 Output CSV: {output_csv_path}")
    print(f"📄 Output report: {output_report_path}")
    print("-" * 60)

    try:
        # Initialize detector with custom dictionaries and target column
        print("🔧 Initializing detector with dictionaries...")
        detector = CSVErrorDetector(
            hindi_dict_path=hindi_dict_path,
            names_dict_path=names_dict_path,
            honorifics_dict_path=honorifics_dict_path,
            target_column=target_column
        )

        # Process CSV file
        print("📊 Processing CSV file...")
        detector.process_csv(input_path, output_csv_path, output_report_path)

        print("\n✅ Analysis completed successfully!")
        print(f"✨ Enhanced CSV saved to: {output_csv_path}")
        print(f"📋 Detailed report saved to: {output_report_path}")

    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
