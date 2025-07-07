import os
import re
from typing import Dict, List, Tuple, Set, Optional, NamedTuple
import logging
from dataclasses import dataclass
from collections import defaultdict
import pandas as pd
from Levenshtein import distance
from spellchecker import SpellChecker
from functools import lru_cache

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

class DictionaryLoader:
    """Handles dictionary loading for Hindi and initializes PySpellChecker for English"""
    @staticmethod
    def load_dictionary(file_path: str) -> Set[str]:
        """Load Hindi dictionary from file and return as set of words"""
        print(f"Loading Hindi dictionary from {file_path}")
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Hindi dictionary file {file_path} not found")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                dictionary = {word.strip().lower() for word in f if word.strip()}
            
            print(f"Successfully loaded {len(dictionary)} Hindi words")
            return dictionary
        
        except Exception as e:
            logger.error(f"Error loading Hindi dictionary {file_path}: {str(e)}")
            raise
    
    @staticmethod
    def load_english_spellchecker() -> SpellChecker:
        """Initialize English spell checker"""
        print("Initializing English spell checker")
        try:
            spell = SpellChecker(language='en', distance=2)
            print("English spell checker initialized successfully")
            return spell
        except Exception as e:
            logger.error(f"Error initializing English SpellChecker: {str(e)}")
            raise
    
    @staticmethod
    def load_custom_dictionary(file_path: str) -> Set[str]:
        """Load custom dictionary from file and return as set of words"""
        print(f"Loading custom dictionary from {file_path}")
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Custom dictionary file {file_path} not found.")
                return set()
            
            with open(file_path, 'r', encoding='utf-8') as f:
                dictionary = {word.strip().lower() for word in f if word.strip()}
            
            logger.info(f"Loaded {len(dictionary)} entries from custom dictionary {file_path}")
            print(f"Successfully loaded {len(dictionary)} custom dictionary entries")
            return dictionary
    
        except Exception as e:
            logger.error(f"Error loading custom dictionary {file_path}: {str(e)}")
            raise

    @staticmethod
    def load_english_spellchecker_with_custom_dicts(names_dict_path: str = None, honorifics_dict_path: str = None) -> SpellChecker:
        """Initialize English spell checker with custom dictionaries"""
        print("Initializing English spell checker with custom dictionaries")
        try:
            spell = SpellChecker(language='en', distance=2)
            
            custom_words = set()
            
            if names_dict_path and os.path.exists(names_dict_path):
                names_dict = DictionaryLoader.load_custom_dictionary(names_dict_path)
                custom_words.update(names_dict)
                logger.info(f"Added {len(names_dict)} names to custom dictionary")
                print(f"Added {len(names_dict)} names to custom dictionary")
            
            if honorifics_dict_path and os.path.exists(honorifics_dict_path):
                honorifics_dict = DictionaryLoader.load_custom_dictionary(honorifics_dict_path)
                custom_words.update(honorifics_dict)
                logger.info(f"Added {len(honorifics_dict)} honorifics to custom dictionary")
                print(f"Added {len(honorifics_dict)} honorifics to custom dictionary")
            
            if custom_words:
                spell.word_frequency.load_words(custom_words)
                logger.info(f"Added total of {len(custom_words)} custom words to spellchecker")
                print(f"Added total of {len(custom_words)} custom words to spellchecker")
            
            print("English spell checker with custom dictionaries initialized successfully")
            return spell
    
        except Exception as e:
            logger.error(f"Error initializing English SpellChecker: {str(e)}")
            raise

class TextAnalyzer:
    """Handles text analysis and error detection"""
    def __init__(self, english_spell: SpellChecker, hindi_dict: Set[str], 
                names_dict: Set[str] = None, honorifics_dict: Set[str] = None):
        self.english_spell = english_spell
        self.hindi_dict = hindi_dict
        self.names_dict = names_dict or set()
        self.honorifics_dict = honorifics_dict or set()
        
        self.english_threshold = 0.4
        self.hindi_threshold = 0.5
        
        self.common_prefixes = {'hon', 'shri', 'smt', 'dr', 'mr', 'mrs', 'ms', 'श्री', 'श्रीमती'}
        if self.honorifics_dict:
            self.common_prefixes.update(self.honorifics_dict)
        
        # Cache for word checks
        self._english_word_cache = {}
        self._hindi_word_cache = {}
    
    def is_english_word(self, word: str) -> bool:
        """Check if a word is mostly English (basic ASCII + letters only)"""
        word_lower = word.lower()
        if word_lower in self.common_prefixes:
            return True
        return all(ord(c) < 128 for c in word) and bool(re.search(r'[a-zA-Z]', word))

    def is_hindi_word(self, word: str) -> bool:
        """Check if a word contains Hindi (Devanagari script) characters"""
        word_lower = word.lower()
        if word_lower in self.common_prefixes:
            return True
        return bool(re.search(r'[\u0900-\u097F\u1CD0-\u1CFF\uA8E0-\uA8FF]', word))
    
    @lru_cache(maxsize=10000)
    def find_best_hindi_corrections(self, word: str, max_corrections: int = 3) -> List[Tuple[str, float]]:
        """Find best corrections for a Hindi word using normalized distance scores"""
        print(f"Finding Hindi corrections for word: {word}")
        corrections = []
        word_len = len(word)
        
        if word_len <= 3:
            candidates = self.hindi_dict
        else:
            candidates = [w for w in self.hindi_dict if abs(len(w) - word_len) <= max(2, word_len * 0.4)]
            if len(candidates) < 50:
                candidates = [w for w in self.hindi_dict if abs(len(w) - word_len) <= max(3, word_len * 0.5)]
        
        for dict_word in candidates:
            edit_distance = distance(word, dict_word)
            normalized_distance = edit_distance / max(word_len, len(dict_word))
            corrections.append((dict_word, normalized_distance))
        
        corrections.sort(key=lambda x: x[1])
        print(f"Found {len(corrections[:max_corrections])} Hindi corrections for {word}")
        return corrections[:max_corrections]
    
    @lru_cache(maxsize=10000)
    def find_best_english_corrections(self, word: str, max_corrections: int = 3) -> List[Tuple[str, float]]:
        """Find best corrections for an English word using PySpellChecker"""
        print(f"Finding English corrections for word: {word}")
        word_len = len(word)
        
        try:
            candidates = self.english_spell.candidates(word.lower())
            if not candidates:
                if word_len > 3:
                    prefix = word[:3].lower()
                    candidates = {w for w in self.english_spell.word_frequency.dictionary 
                                 if w.startswith(prefix) and abs(len(w) - word_len) <= 2}
        except Exception as e:
            logger.warning(f"SpellChecker error for word '{word}': {e}")
            return []
        
        corrections = []
        for candidate in candidates:
            edit_distance = distance(word.lower(), candidate)
            normalized_distance = edit_distance / max(word_len, len(candidate))
            corrections.append((candidate, normalized_distance))
        
        corrections.sort(key=lambda x: x[1])
        print(f"Found {len(corrections[:max_corrections])} English corrections for {word}")
        return corrections[:max_corrections]
    
    def check_english_word(self, word: str) -> Tuple[bool, Optional[str], float, List[Tuple[str, float]]]:
        """Check if an English word is an error using both PySpellChecker and custom dictionaries"""
        word_lower = word.lower()
        
        if len(word) <= 2 or any(c.isdigit() for c in word) or not re.search(r'[a-zA-Z]', word):
            return False, None, 0.0, []
            
        if word_lower in self._english_word_cache:
            print(f"Using cached English word check result for: {word}")
            return self._english_word_cache[word_lower]
        
        if (word_lower in self.common_prefixes or 
            word_lower in self.honorifics_dict or 
            word_lower in self.names_dict):
            self._english_word_cache[word_lower] = (False, None, 0.0, [])
            return False, None, 0.0, []
        
        if word[0].isupper() and word_lower in self.names_dict:
            self._english_word_cache[word_lower] = (False, None, 0.0, [])
            return False, None, 0.0, []
        
        if not self.english_spell.unknown([word_lower]):
            self._english_word_cache[word_lower] = (False, None, 0.0, [])
            return False, None, 0.0, []
            
        if word[0].isupper() and len(word) > 2:
            self._english_word_cache[word_lower] = (False, None, 0.0, [])
            return False, None, 0.0, []
        
        corrections = self.find_best_english_corrections(word)
        
        if not corrections:
            if word[0].isupper() and len(word) > 3:
                self._english_word_cache[word_lower] = (False, None, 0.0, [])
                return False, None, 0.0, []
            self._english_word_cache[word_lower] = (True, None, 1.0, [])
            return True, None, 1.0, []
        
        best_correction, best_distance = corrections[0]
        
        word_len = len(word)
        if word_len <= 4:
            threshold = self.english_threshold * 1.5
        elif word_len <= 6:
            threshold = self.english_threshold * 1.25
        else:
            threshold = self.english_threshold
        
        if best_distance <= threshold:
            self._english_word_cache[word_lower] = (True, best_correction, best_distance, corrections)
            return True, best_correction, best_distance, corrections
        else:
            if word[0].isupper() and word_len > 3:
                self._english_word_cache[word_lower] = (False, None, best_distance, corrections)
                return False, None, best_distance, corrections
            self._english_word_cache[word_lower] = (True, best_correction, best_distance, corrections)
            return True, best_correction, best_distance, corrections
    
    def check_hindi_word(self, word: str) -> Tuple[bool, Optional[str], float, List[Tuple[str, float]]]:
        """Check if a Hindi word is misspelled using dictionary lookup and edit distance"""
        if len(word) <= 2 or not self.is_hindi_word(word):
            return False, None, 0.0, []
        
        if word in self._hindi_word_cache:
            print(f"Using cached Hindi word check result for: {word}")
            return self._hindi_word_cache[word]
        
        if word.lower() in self.hindi_dict:
            self._hindi_word_cache[word] = (False, None, 0.0, [])
            return False, None, 0.0, []
        
        corrections = self.find_best_hindi_corrections(word)
        
        if not corrections:
            self._hindi_word_cache[word] = (True, None, 1.0, [])
            return True, None, 1.0, []
        
        best_correction, best_distance = corrections[0]
        
        word_len = len(word)
        if word_len <= 4:
            threshold = self.hindi_threshold * 1.5
        elif word_len <= 6:
            threshold = self.hindi_threshold * 1.25
        else:
            threshold = self.hindi_threshold
        
        if best_distance <= threshold:
            self._hindi_word_cache[word] = (True, best_correction, best_distance, corrections)
            return True, best_correction, best_distance, corrections
        else:
            self._hindi_word_cache[word] = (True, best_correction, best_distance, corrections)
            return True, best_correction, best_distance, corrections
    
    def calculate_cell_accuracy(self, text: str, row_num: int, column_name: str) -> Tuple[int, int, float, Dict[str, Tuple[str, WordPosition]], int, int, float, Dict[str, Tuple[str, WordPosition]], List[ErrorDetail]]:
        """Calculate error percentage for both English and Hindi text with position tracking and detailed error info"""
        print(f"Analyzing cell in row {row_num}, column {column_name}")
        if pd.isna(text) or text == '':
            return 0, 0, 0.0, {}, 0, 0, 0.0, {}, []
        
        text = str(text)
        words = re.findall(r'\b[\w\u0900-\u097F\u1CD0-\u1CFF\uA8E0-\uA8FF]+\b', text)
        
        eng_total = 0
        hin_total = 0
        eng_errors = 0
        hin_errors = 0
        eng_corrections = {}
        hin_corrections = {}
        error_details = []
    
        for word_num, word in enumerate(words, 1):
            if len(word) <= 1 or not re.search(r'[a-zA-Z\u0900-\u097F]', word):
                continue
            
            context = text[:100] + "..." if len(text) > 100 else text
            position = WordPosition(row=row_num, column=column_name, word=word_num, text=context)
            
            if self.is_english_word(word) and not self.is_hindi_word(word):
                eng_total += 1
                is_error, correction, error_distance, all_corrections = self.check_english_word(word)
                if is_error and error_distance > 0.01:
                    eng_errors += 1
                    if correction:
                        eng_corrections[word.lower()] = (correction, position)
                    
                    error_detail = ErrorDetail(
                        original_word=word,
                        error_distance=error_distance,
                        suggested_corrections=all_corrections,
                        position=position
                    )
                    error_details.append(error_detail)
        
            elif self.is_hindi_word(word) and not self.is_english_word(word):
                hin_total += 1
                is_error, correction, error_distance, all_corrections = self.check_hindi_word(word)
                if is_error and error_distance > 0.01:
                    hin_errors += 1
                    if correction:
                        hin_corrections[word] = (correction, position)
                    
                    error_detail = ErrorDetail(
                        original_word=word,
                        error_distance=error_distance,
                        suggested_corrections=all_corrections,
                        position=position
                    )
                    error_details.append(error_detail)
            
            else:
                hindi_chars = sum(1 for c in word if '\u0900' <= c <= '\u097F' or '\u1CD0' <= c <= '\u1CFF' or '\uA8E0' <= c <= '\uA8FF')
                english_chars = sum(1 for c in word if 'a' <= c.lower() <= 'z')
                
                if hindi_chars > english_chars:
                    hin_total += 1
                    is_error, correction, error_distance, all_corrections = self.check_hindi_word(word)
                    if is_error and error_distance > 0.01:
                        hin_errors += 1
                        if correction:
                            hin_corrections[word] = (correction, position)
                        
                        error_detail = ErrorDetail(
                            original_word=word,
                            error_distance=error_distance,
                            suggested_corrections=all_corrections,
                            position=position
                        )
                        error_details.append(error_detail)
                        
                elif english_chars > hindi_chars:
                    eng_total += 1
                    is_error, correction, error_distance, all_corrections = self.check_english_word(word)
                    if is_error and error_distance > 0.01:
                        eng_errors += 1
                        if correction:
                            eng_corrections[word.lower()] = (correction, position)
                        
                        error_detail = ErrorDetail(
                            original_word=word,
                            error_distance=error_distance,
                            suggested_corrections=all_corrections,
                            position=position
                        )
                        error_details.append(error_detail)
        
        eng_error_percentage = (eng_errors / eng_total * 100) if eng_total > 0 else 0.0
        hin_error_percentage = (hin_errors / hin_total * 100) if hin_total > 0 else 0.0
        
        eng_accuracy = 100 - eng_error_percentage
        hin_accuracy = 100 - hin_error_percentage
        
        print(f"Cell analysis complete: {eng_errors} English errors, {hin_errors} Hindi errors")
        return eng_total, eng_errors, eng_accuracy, eng_corrections, hin_total, hin_errors, hin_accuracy, hin_corrections, error_details

class CSVErrorDetector:
    """Main class for CSV OCR error detection"""
    def __init__(self, english_dict_path: str = None, hindi_dict_path: str = "hi_IN.dic",
                names_dict_path: str = None, honorifics_dict_path: str = None,
                target_column: str = "speech"):
        print("Initializing CSVErrorDetector")
        self.dict_loader = DictionaryLoader()
        self.target_column = target_column
        
        self.english_spell = self.dict_loader.load_english_spellchecker_with_custom_dicts(
            names_dict_path, honorifics_dict_path
        )
        
        self.hindi_dict = self.dict_loader.load_dictionary(hindi_dict_path)
        
        self.names_dict = self.dict_loader.load_custom_dictionary(names_dict_path) if names_dict_path else set()
        self.honorifics_dict = self.dict_loader.load_custom_dictionary(honorifics_dict_path) if honorifics_dict_path else set()
            
        self.analyzer = TextAnalyzer(
            self.english_spell, 
            self.hindi_dict,
            self.names_dict,
            self.honorifics_dict
        )
        
        self.all_english_errors = {}
        self.all_hindi_errors = {}
        print("CSVErrorDetector initialized successfully")
    
    def determine_primary_language(self, text: str) -> str:
        """Determine the primary language of the text"""
        if pd.isna(text) or text == '':
            return "Unknown"
        
        text = str(text)
        words = re.findall(r'\b[\w\u0900-\u097F\u1CD0-\u1CFF\uA8E0-\uA8FF]+\b', text)
        
        english_count = 0
        hindi_count = 0
        
        for word in words:
            if self.analyzer.is_english_word(word) and not self.analyzer.is_hindi_word(word):
                english_count += 1
            elif self.analyzer.is_hindi_word(word) and not self.analyzer.is_english_word(word):
                hindi_count += 1
        
        if english_count > hindi_count * 1.5:
            return "English"
        elif hindi_count > english_count * 1.5:
            return "Hindi"
        elif english_count > 0 or hindi_count > 0:
            return "Mixed"
        else:
            return "Unknown"
    
    def process_csv(self, input_csv_path: str, output_csv_path: str, output_report_path: str = None, chunk_size: int = 1000) -> None:
        """Process input CSV file in chunks and generate enhanced output with error analysis"""
        print(f"Starting CSV processing: {input_csv_path}")
        try:
            if not os.path.exists(input_csv_path):
                raise FileNotFoundError(f"Input file not found: {input_csv_path}")
            
            all_results = []
            chunk_number = 0
            
            # Process CSV in chunks
            for chunk in pd.read_csv(input_csv_path, chunksize=chunk_size):
                chunk_number += 1
                print(f"Processing chunk {chunk_number} with {len(chunk)} rows")
                
                # Analyze chunk
                chunk_results = self._analyze_csv(chunk)
                all_results.extend(chunk_results)
                print(f"Completed processing chunk {chunk_number}")
            
            # Generate overall CSV statistics
            print("Calculating overall CSV statistics")
            csv_stats = self._calculate_csv_stats(all_results)
            
            # Load full CSV for creating enhanced output
            print("Creating enhanced DataFrame")
            df = pd.read_csv(input_csv_path)
            enhanced_df = self._create_enhanced_dataframe(df, all_results)
            
            # Save enhanced CSV
            print(f"Saving enhanced CSV to {output_csv_path}")
            enhanced_df.to_csv(output_csv_path, index=False, encoding='utf-8')
            
            # Write analysis report if path is provided
            if output_report_path:
                print(f"Writing analysis report to {output_report_path}")
                self._write_results(output_report_path, df, all_results, csv_stats)
            
            logger.info(f"Analysis completed successfully. Enhanced CSV: {output_csv_path}")
            print("CSV processing completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing CSV file: {str(e)}")
            raise
    
    def _create_enhanced_dataframe(self, df: pd.DataFrame, results: List[RowAnalysis]) -> pd.DataFrame:
        """Create enhanced DataFrame with additional accuracy and error columns"""
        print("Creating enhanced DataFrame with analysis results")
        enhanced_df = df.copy()
        
        speech_accuracy_percentages = []
        speech_primary_languages = []
        speech_error_counts = []
        speech_error_details_cols = []
        
        for result in results:
            speech_accuracy_percentages.append(round(result.accuracy_percentage, 2))
            speech_primary_languages.append(result.primary_language)
            speech_error_counts.append(result.total_errors)
            
            error_details_str = self._format_error_details_for_csv(result.error_details)
            speech_error_details_cols.append(error_details_str)
        
        enhanced_df['Speech_Accuracy_Percentage'] = speech_accuracy_percentages
        enhanced_df['Speech_Primary_Language'] = speech_primary_languages
        enhanced_df['Speech_Number_of_Errors'] = speech_error_counts
        enhanced_df['Speech_Error_Details'] = speech_error_details_cols
        
        print("Enhanced DataFrame created successfully")
        return enhanced_df
    
    def _format_error_details_for_csv(self, error_details: List[ErrorDetail]) -> str:
        """Format error details as a string for CSV storage"""
        if not error_details:
            return ""
        
        formatted_errors = []
        for error in error_details:
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
        
        return "; ".join(formatted_errors[:10])
    
    def _analyze_csv(self, df: pd.DataFrame) -> List[RowAnalysis]:
        """Analyze each row of the CSV and return results - focusing only on 'speech' column"""
        print("Analyzing CSV data")
        results = []
        
        if self.target_column not in df.columns:
            logger.warning(f"'{self.target_column}' column not found in CSV. Available columns: " + ", ".join(df.columns))
            raise ValueError(f"'{self.target_column}' column not found in the CSV file")
        
        for row_idx, row in df.iterrows():
            row_num = row_idx + 2
            print(f"Analyzing row {row_num}")
            column_stats = {}
            all_eng_corrections = {}
            all_hin_corrections = {}
            all_error_details = []
            
            col_name = self.target_column
            cell_value = row[col_name]
            
            primary_language = self.determine_primary_language(str(cell_value) if pd.notna(cell_value) else "")
            
            total_words_in_row = 0
            total_errors_in_row = 0
            
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
                eng_total, eng_errors, eng_accuracy, eng_corrections, \
                hin_total, hin_errors, hin_accuracy, hin_corrections, error_details = self.analyzer.calculate_cell_accuracy(
                    str(cell_value), row_num, col_name
                )
                
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
                
                all_error_details.extend(error_details)
                
                total_words = eng_total + hin_total
                total_errors = eng_errors + hin_errors
                overall_accuracy = 100 - (total_errors / total_words * 100) if total_words > 0 else 100.0
                
                total_words_in_row += total_words
                total_errors_in_row += total_errors
                
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
            
            row_accuracy = 100 - (total_errors_in_row / total_words_in_row * 100) if total_words_in_row > 0 else 100.0
            
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
            print(f"Completed analysis for row {row_num}")
        
        print("CSV analysis completed")
        return results 
    
    def _calculate_csv_stats(self, results: List[RowAnalysis]) -> Dict:
        """Calculate overall statistics for the CSV"""
        print("Calculating CSV statistics")
        total_english_words = 0
        total_english_errors = 0
        total_hindi_words = 0
        total_hindi_errors = 0
        
        column_stats = defaultdict(lambda: {
            'english_total': 0,
            'english_errors': 0,
            'hindi_total': 0,
            'hindi_errors': 0,
            'total_words': 0,
            'total_errors': 0
        })
        
        for result in results:
            for col_name, stats in result.column_stats.items():
                total_english_words += stats['english_total']
                total_english_errors += stats['english_errors']
                total_hindi_words += stats['hindi_total']
                total_hindi_errors += stats['hindi_errors']
                
                column_stats[col_name]['english_total'] += stats['english_total']
                column_stats[col_name]['english_errors'] += stats['english_errors']
                column_stats[col_name]['hindi_total'] += stats['hindi_total']
                column_stats[col_name]['hindi_errors'] += stats['hindi_errors']
                column_stats[col_name]['total_words'] += stats['total_words']
                column_stats[col_name]['total_errors'] += (stats['english_errors'] + stats['hindi_errors'])
        
        english_accuracy = 100 - (total_english_errors / total_english_words * 100) if total_english_words > 0 else 100.0
        hindi_accuracy = 100 - (total_hindi_errors / total_hindi_words * 100) if total_hindi_words > 0 else 100.0
        total_words = total_english_words + total_hindi_words
        total_errors = total_english_errors + total_hindi_errors
        overall_accuracy = 100 - (total_errors / total_words * 100) if total_words > 0 else 100.0
        
        for col_name in column_stats:
            col = column_stats[col_name]
            col['english_accuracy'] = 100 - (col['english_errors'] / col['english_total'] * 100) if col['english_total'] > 0 else 100.0
            col['hindi_accuracy'] = 100 - (col['hindi_errors'] / col['hindi_total'] * 100) if col['hindi_total'] > 0 else 100.0
            col['overall_accuracy'] = 100 - (col['total_errors'] / col['total_words'] * 100) if col['total_words'] > 0 else 100.0
        
        language_counts = {
            'English': sum(1 for r in results if r.primary_language == 'English'),
            'Hindi': sum(1 for r in results if r.primary_language == 'Hindi'),
            'Mixed': sum(1 for r in results if r.primary_language == 'Mixed'),
            'Unknown': sum(1 for r in results if r.primary_language == 'Unknown')
        }
        
        print("CSV statistics calculated")
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
    
    def _get_most_frequent_errors(self, error_dict: Dict[str, str], limit: int = 10) -> List[Tuple[str, str]]:
        """Return most frequent errors"""
        error_counts = defaultdict(int)
        for word in error_dict:
            error_counts[word] += 1
        
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        
        top_errors = []
        for word, count in sorted_errors[:limit]:
            correction = error_dict.get(word, "")
            top_errors.append((word, correction, count))
        
        return top_errors
    
    def _write_results(self, output_path: str, df: pd.DataFrame, results: List[RowAnalysis], csv_stats: Dict) -> None:
        """Write detailed analysis results to output file"""
        print(f"Writing detailed analysis report to {output_path}")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# OCR Error Analysis Report\n\n")
                f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("## Overall Statistics\n\n")
                f.write(f"Total rows analyzed: {csv_stats['total_rows']}\n")
                f.write(f"Total words: {csv_stats['total_words']}\n")
                f.write(f"Total errors detected: {csv_stats['total_errors']}\n")
                f.write(f"Overall accuracy: {csv_stats['overall_accuracy']:.2f}%\n\n")
                
                f.write("## Language Statistics\n\n")
                f.write(f"English words: {csv_stats['total_english_words']} (Accuracy: {csv_stats['english_accuracy']:.2f}%)\n")
                f.write(f"Hindi words: {csv_stats['total_hindi_words']} (Accuracy: {csv_stats['hindi_accuracy']:.2f}%)\n\n")
                
                f.write("## Row Language Distribution\n\n")
                for lang, count in csv_stats['language_distribution'].items():
                    f.write(f"{lang}: {count} rows ({count/csv_stats['total_rows']*100:.1f}%)\n")
                f.write("\n")
                
                f.write("## Column-wise Statistics\n\n")
                for col_name, stats in csv_stats['column_stats'].items():
                    f.write(f"### Column: {col_name}\n")
                    f.write(f"Total words: {stats['total_words']}\n")
                    f.write(f"English words: {stats['english_total']} (Errors: {stats['english_errors']}, Accuracy: {stats['english_accuracy']:.2f}%)\n")
                    f.write(f"Hindi words: {stats['hindi_total']} (Errors: {stats['hindi_errors']}, Accuracy: {stats['hindi_accuracy']:.2f}%)\n")
                    f.write(f"Overall accuracy: {stats['overall_accuracy']:.2f}%\n\n")
                
                f.write("## Most Frequent English Errors\n\n")
                for word, correction, count in csv_stats['most_frequent_english_errors']:
                    f.write(f"* '{word}' → '{correction}' (Found {count} times)\n")
                f.write("\n")
                
                f.write("## Most Frequent Hindi Errors\n\n")
                for word, correction, count in csv_stats['most_frequent_hindi_errors']:
                    f.write(f"* '{word}' → '{correction}' (Found {count} times)\n")
                f.write("\n")
                
                f.write("## Detailed Row Analysis\n\n")
                for result in sorted(results, key=lambda x: x.accuracy_percentage):
                    if result.total_errors > 0:
                        f.write(f"### Row {result.row_number} (Accuracy: {result.accuracy_percentage:.2f}%, Language: {result.primary_language})\n\n")
                        
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
                print("Analysis report written successfully")
                
        except Exception as e:
            logger.error(f"Error writing analysis report: {str(e)}")
            raise

def main():
    input_path = "16-III-01.12.2014.csv"
    output_csv_path = "enhanced_16th_data.csv"
    output_report_path = "detailed16_csv_analysis_report.txt"
    hindi_dict_path = "hin.dic"
    names_dict_path = "Names.txt"
    honorifics_dict_path = "honorifics.txt"
    target_column = "speech"
    
    print("Starting main execution")
    detector = CSVErrorDetector(
        hindi_dict_path=hindi_dict_path,
        names_dict_path=names_dict_path,
        honorifics_dict_path=honorifics_dict_path,
        target_column=target_column
    )
    
    detector.process_csv(input_path, output_csv_path, output_report_path)
    print("Main execution completed")

if __name__ == "__main__":
    main()
