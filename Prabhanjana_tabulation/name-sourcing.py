import pandas as pd
import re
import string
from fuzzywuzzy import fuzz, process
from difflib import SequenceMatcher
import sys


class Config:
    # Similarity thresholds
    SPECIAL_ROLE_DETECTION_THRESHOLD = 0.85
    HIGH_SIMILARITY_THRESHOLD = 0.8
    FALLBACK_THRESHOLD = 0.5
    CONSONANT_SIMILARITY_THRESHOLD = 0.7
    NO_SPACE_SIMILARITY_THRESHOLD = 0.8
    
    # Weight schemes for combining scores
    WORD_MATCH_WEIGHT = 0.8
    STRING_SIMILARITY_WEIGHT = 0.2
    
    # Base scores for different match types
    EXACT_MATCH_SCORE = 1.0
    CONTAINMENT_BASE_SCORE = 0.85
    NO_SPACE_CONTAINMENT_SCORE = 0.95
    CONSONANT_MATCH_SCORE = 0.85
    
    # Coverage bonus multipliers
    COVERAGE_BONUS_MULTIPLIER = 0.05
    
    # Scaling factors for different similarity types
    NO_SPACE_SIMILARITY_SCALE = 0.85
    REGULAR_SIMILARITY_SCALE = 0.7
    CONSONANT_SIMILARITY_SCALE = 0.8
    WORD_BASED_MATCH_SCALE = 0.6
    
    # Sequential match bonus
    SEQUENTIAL_MATCH_BONUS = 0.05
    
    # Hindi matras (vowel marks) for removal - Keep existing proven logic
    HINDI_MATRAS = [
        '\u093e', '\u093f', '\u0940', '\u0941', '\u0942', '\u0943', '\u0944',
        '\u0945', '\u0946', '\u0947', '\u0948', '\u0949', '\u094a', '\u094b',
        '\u094c', '\u094d'
    ]
    
    # Honorifics and titles - TODO: Extend with more titles as needed
    HINDI_TITLES = ['श्री', 'श्रीमती', 'डॉ', 'प्रो', 'कुमारी']
    ENGLISH_TITLES = ['shri', 'shrimati', 'dr', 'prof', 'mr', 'mrs', 'ms']
    
    SPECIAL_ROLES = {
        'speaker': {
            'english': ['hon. speaker', 'madam speaker'],
            'hindi': ['माननीय अध्यक्ष', 'अध्यक्ष']
        },
        'deputy_speaker': {
            'english': ['hon. deputy speaker'],
            'hindi': ['उपाध्यक्ष', 'माननीय उपाध्यक्ष']
        },
        'some_hon_members': {
            'english': ['some hon. members', 'several hon. members', 'hon. members'],
            'hindi': ['कुछ माननीय सदस्य', 'कई माननीय सदस्य', 'माननीय सदस्य']
        },
        'chairperson': {
            'english': ['hon. chairperson', 'chairperson', 'chair'],
            'hindi': ['माननीय सभापति', 'सभापति']
        },
        'secretary_general': {
            'english': ['secretary-general', 'secretary general'],
            'hindi': ['महासचिव', 'सचिव सामान्य']
        },
        'president': {
            'english': ['hon. president', 'president', 'madam president'],
            'hindi': ['माननीय राष्ट्रपति', 'राष्ट्रपति']
        },
        'attorney_general': {
            'english': ['attorney general', 'attorney-general'],
            'hindi': ['महान्यायवादी', 'अटॉर्नी जनरल']
        },
        'officer_house': {
            'english': ['officer of the house', 'house officer', 'parliamentary officer'],
            'hindi': ['सदन अधिकारी', 'सदन के अधिकारी']
        }
    }

class NameNormalizer:
    """Handles normalization of names in both English and Hindi"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def normalize_name(self, name):
        """Normalize names for better comparison"""
        if not isinstance(name, str):
            return ""
        
        # Convert to lowercase for English names
        name = name.lower()
        
        # Remove punctuation
        name = name.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra spaces
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name
    
    def normalize_hindi_name(self, name):
        """Handle Hindi name variations with and without spaces"""
        if not isinstance(name, str) or not name:
            return "", ""
        
        # Regular normalization
        norm_name = self.normalize_name(name)
        
        # Create a version with spaces removed
        no_space_name = norm_name.replace(" ", "")
        
        return norm_name, no_space_name
    
    def extract_clean_name_and_constituency(self, speaker_name, is_hindi=False):
        """Extract the core name and constituency from speaker string"""
        if not isinstance(speaker_name, str):
            return "", ""
        clean_name = speaker_name.lower().strip()
        constituency = ""
    
        # For Hindi names, return the full name without any processing
        if is_hindi:
            return clean_name, ""
    
        # Check for "SUBMISSIONS BY MEMBERS" with 85% accuracy
        if self._contains_phrase_with_accuracy(clean_name, "submissions by members", 0.85):
            # Return special marker for submissions by members
            return "STATEMENT:" + clean_name, ""
        if self._contains_phrase_with_accuracy(clean_name, "motion re", 0.9):
            # Return special marker for submissions by members
            return "STATEMENT:" + clean_name, ""
        
    
        # Check for brackets and handle special cases - find the LAST set of brackets
        bracket_matches = list(re.finditer(r'\([^)]+\)', clean_name))
        if bracket_matches:
            # Get the last (rightmost) bracket match
            bracket_match = bracket_matches[-1]
            text_before_bracket = clean_name[:bracket_match.start()].strip()
            bracketed_content = bracket_match.group(0)[1:-1].strip()  # Remove the brackets
        
            # Check if "nominated" appears in brackets with 85% accuracy
            if self._contains_phrase_with_accuracy(bracketed_content, "nominated", 0.85):
                # Return special marker for nominated members
                return "NOMINATED_MEMBER:" + text_before_bracket, ""
        
            # Check if "minister of" appears before brackets with 85% accuracy
            if self._contains_phrase_with_accuracy(text_before_bracket, "minister of", 0.85):
                # Return only the content within brackets as name, no constituency
                return bracketed_content, ""
            else:
                # Store bracketed content as constituency and continue with normal processing
                constituency = bracketed_content
                clean_name = text_before_bracket
    
        # Split into words and filter out titles
        words = clean_name.split()
        filtered_words = []
        for word in words:
            word_clean = word.strip('.,')
            if (word_clean not in self.config.HINDI_TITLES and
                word_clean.lower() not in self.config.ENGLISH_TITLES):
                filtered_words.append(word_clean)
        final_name = ' '.join(filtered_words).strip()
        return final_name, constituency


    def _contains_phrase_with_accuracy(self, text, target_phrase, threshold):
        """Check if target phrase exists in text with given accuracy threshold using fuzzy matching"""
        words = text.split()
        target_words = target_phrase.split()
        
        # Check all possible n-grams of the same length as target phrase
        for i in range(len(words) - len(target_words) + 1):
            ngram = ' '.join(words[i:i + len(target_words)])
            similarity = SequenceMatcher(None, ngram, target_phrase).ratio()
            if similarity >= threshold:
                return True
        
        return False

class HindiTextProcessor:
    """Specialized processor for Hindi text with matra removal logic"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def remove_matras(self, text):
        """Remove matras (vowel marks) from Hindi text to get base consonants"""
        if not isinstance(text, str):
            return ""
        
        result = text
        for matra in self.config.HINDI_MATRAS:
            result = result.replace(matra, '')
        
        return result
    
    def calculate_consonant_similarity(self, word1, word2):
        """Calculate similarity based on base consonants (without matras)"""
        if not word1 or not word2:
            return 0
        
        # Remove matras from both words
        base1 = self.remove_matras(word1)
        base2 = self.remove_matras(word2)
        
        if not base1 or not base2:
            return 0
        
        # Check for exact base consonant match
        if base1 == base2:
            return self.config.CONSONANT_MATCH_SCORE
        
        # Use sequence matcher for partial consonant similarity
        return SequenceMatcher(None, base1, base2).ratio() * 0.5

class SimilarityScorer:
    """Handles similarity calculations between names using FuzzyWuzzy for enhanced matching"""
    
    def __init__(self, config: Config, normalizer: NameNormalizer, hindi_processor: HindiTextProcessor):
        self.config = config
        self.normalizer = normalizer
        self.hindi_processor = hindi_processor
    
    def calculate_constituency_similarity(self, extracted_constituency, mp_constituency):
        """Calculate similarity between extracted constituency and MP constituency using FuzzyWuzzy"""
        if not extracted_constituency or not isinstance(mp_constituency, str):
            return 0
        
        norm_extracted = self.normalizer.normalize_name(extracted_constituency)
        norm_mp_const = self.normalizer.normalize_name(mp_constituency)
        
        if not norm_extracted or not norm_mp_const:
            return 0
        
        # Use FuzzyWuzzy for constituency matching
        # token_set_ratio is best for constituency names as it handles word order and extra words
        token_set_score = fuzz.token_set_ratio(norm_extracted, norm_mp_const) / 100.0
        partial_score = fuzz.partial_ratio(norm_extracted, norm_mp_const) / 100.0
        basic_score = fuzz.ratio(norm_extracted, norm_mp_const) / 100.0
        
        # Weight the scores - token_set is most important for constituencies
        combined_score = (token_set_score * 0.5) + (partial_score * 0.3) + (basic_score * 0.2)
        
        return combined_score
    
    def check_name_words_match_with_constituency(self, speaker_name, mp_name, mp_constituency, is_hindi=False):
        """Enhanced matching with constituency consideration for English names"""
        if not isinstance(speaker_name, str) or not isinstance(mp_name, str):
            return 0, [], 0
        
        # Extract clean names and constituency
        if is_hindi:
            clean_speaker, _ = self.normalizer.extract_clean_name_and_constituency(speaker_name, is_hindi=True)
            clean_mp = self.normalizer.extract_clean_name_and_constituency(mp_name, is_hindi=True)[0]
            constituency_score = 0  # No constituency matching for Hindi
        else:
            clean_speaker, extracted_constituency = self.normalizer.extract_clean_name_and_constituency(speaker_name, is_hindi=False)
            clean_mp = self.normalizer.extract_clean_name_and_constituency(mp_name, is_hindi=False)[0]
            constituency_score = self.calculate_constituency_similarity(extracted_constituency, mp_constituency)
        
        # Calculate name similarity using existing logic
        name_score, matched_words = self.check_name_words_match(clean_speaker, clean_mp, is_hindi)
        
        return name_score, matched_words, constituency_score
    
    def check_name_words_match(self, speaker_name, mp_name, is_hindi=False):
        """Enhanced matching with FuzzyWuzzy for English and improved Hindi full-string comparison"""
        if not isinstance(speaker_name, str) or not isinstance(mp_name, str):
            return 0, []
        
        # Extract clean names - pass is_hindi parameter
        clean_speaker = self.normalizer.extract_clean_name_and_constituency(speaker_name, is_hindi=False)[0]  # Speaker name is always processed as English for special cases
        clean_mp = self.normalizer.extract_clean_name_and_constituency(mp_name, is_hindi=is_hindi)[0]
        
        # For Hindi names, use existing logic (keep unchanged for backward compatibility)
        if is_hindi:
            norm_speaker, speaker_no_space = self.normalizer.normalize_hindi_name(clean_speaker)
            norm_mp, mp_no_space = self.normalizer.normalize_hindi_name(clean_mp)
            
            # Full string comparison without spaces
            if speaker_no_space and mp_no_space:
                # Check if the MP name (no spaces) is contained in speaker name (no spaces)
                if mp_no_space in speaker_no_space:
                    coverage = len(mp_no_space) / len(speaker_no_space) if len(speaker_no_space) > 0 else 0
                    base_score = self.config.NO_SPACE_CONTAINMENT_SCORE
                    coverage_bonus = coverage * self.config.COVERAGE_BONUS_MULTIPLIER
                    return min(base_score + coverage_bonus, 1.0), norm_mp.split()
                
                # Check for exact match (no spaces)
                if speaker_no_space == mp_no_space:
                    return self.config.EXACT_MATCH_SCORE, norm_mp.split()
                
                # Check string similarity for no-space versions using FuzzyWuzzy
                no_space_similarity = fuzz.ratio(mp_no_space, speaker_no_space) / 100.0
                if no_space_similarity >= self.config.NO_SPACE_SIMILARITY_THRESHOLD:
                    return no_space_similarity * self.config.NO_SPACE_SIMILARITY_SCALE, norm_mp.split()
            
            # Check with spaces as secondary approach
            if norm_speaker and norm_mp:
                if norm_mp in norm_speaker:
                    coverage = len(norm_mp) / len(norm_speaker) if len(norm_speaker) > 0 else 0
                    base_score = self.config.CONTAINMENT_BASE_SCORE
                    coverage_bonus = coverage * self.config.COVERAGE_BONUS_MULTIPLIER
                    return min(base_score + coverage_bonus, 1.0), norm_mp.split()
                
                if norm_speaker == norm_mp:
                    return 0.95, norm_mp.split()
            
            # Consonant-based comparison as fallback
            if speaker_no_space and mp_no_space:
                consonant_similarity = self.hindi_processor.calculate_consonant_similarity(speaker_no_space, mp_no_space)
                if consonant_similarity >= self.config.CONSONANT_SIMILARITY_THRESHOLD:
                    return consonant_similarity * self.config.CONSONANT_SIMILARITY_SCALE, norm_mp.split()
        
        else:
            # Enhanced English name processing with FuzzyWuzzy
            norm_speaker = self.normalizer.normalize_name(clean_speaker)
            norm_mp = self.normalizer.normalize_name(clean_mp)
            
            if not norm_speaker or not norm_mp:
                return 0, []
            
            # Use FuzzyWuzzy for English name matching
            # Multiple ratio types for comprehensive matching
            basic_ratio = fuzz.ratio(norm_mp, norm_speaker) / 100.0
            partial_ratio = fuzz.partial_ratio(norm_mp, norm_speaker) / 100.0
            token_sort_ratio = fuzz.token_sort_ratio(norm_mp, norm_speaker) / 100.0
            token_set_ratio = fuzz.token_set_ratio(norm_mp, norm_speaker) / 100.0
            
            # Check for exact match first
            if norm_speaker == norm_mp:
                return self.config.EXACT_MATCH_SCORE, norm_mp.split()
            
            # Check if MP name is contained in speaker name (for backward compatibility)
            if norm_mp in norm_speaker:
                coverage = len(norm_mp) / len(norm_speaker) if len(norm_speaker) > 0 else 0
                base_score = self.config.CONTAINMENT_BASE_SCORE
                coverage_bonus = coverage * self.config.COVERAGE_BONUS_MULTIPLIER
                containment_score = min(base_score + coverage_bonus, 1.0)
                
                # Use the higher of containment score or FuzzyWuzzy scores
                fuzzy_score = max(basic_ratio, partial_ratio, token_sort_ratio, token_set_ratio)
                final_score = max(containment_score, fuzzy_score)
                return final_score, norm_mp.split()
            
            # Use weighted combination of FuzzyWuzzy scores
            # token_set_ratio is most important for names with titles/extra words
            # token_sort_ratio handles word order differences
            # partial_ratio handles partial matches
            # basic_ratio for overall similarity
            weighted_score = (
                token_set_ratio * 0.35 +      # Handles extra words, titles
                token_sort_ratio * 0.25 +     # Handles word order
                partial_ratio * 0.25 +        # Handles partial matches
                basic_ratio * 0.15            # Overall similarity
            )
            
            # Apply minimum threshold
            if weighted_score >= self.config.FALLBACK_THRESHOLD:
                return weighted_score, norm_mp.split()
        
        # Fallback to word-by-word matching for very low scores
        return self._word_by_word_matching(norm_speaker, norm_mp, is_hindi)
    
    def _word_by_word_matching(self, norm_speaker, norm_mp, is_hindi):
        """Enhanced word-by-word matching with FuzzyWuzzy for English"""
        if not norm_speaker or not norm_mp:
            return 0, []
        
        speaker_words = norm_speaker.split()
        mp_words = norm_mp.split()
        
        matched_words = []
        word_scores = []
        
        for mp_word in mp_words:
            best_match_score = 0
            best_match_word = None
            
            for speaker_word in speaker_words:
                if is_hindi:
                    # Keep existing Hindi logic
                    if mp_word == speaker_word:
                        best_match_score = 1.0
                        best_match_word = mp_word
                        break
                    else:
                        consonant_sim = self.hindi_processor.calculate_consonant_similarity(mp_word, speaker_word)
                        if consonant_sim > best_match_score:
                            best_match_score = consonant_sim
                            best_match_word = mp_word
                else:
                    # Enhanced English word matching with FuzzyWuzzy
                    if mp_word == speaker_word:
                        best_match_score = 1.0
                        best_match_word = mp_word
                        break
                    else:
                        # Use FuzzyWuzzy for individual word matching
                        word_ratio = fuzz.ratio(mp_word, speaker_word) / 100.0
                        if word_ratio > best_match_score:
                            best_match_score = word_ratio
                            best_match_word = mp_word
            
            if best_match_score >= self.config.FALLBACK_THRESHOLD:
                matched_words.append(best_match_word)
                word_scores.append(best_match_score)
        
        if matched_words:
            avg_word_score = sum(word_scores) / len(word_scores)
            coverage_score = len(matched_words) / len(mp_words)
            match_score = avg_word_score * coverage_score * self.config.WORD_BASED_MATCH_SCALE
            
            # Add boost for sequential matches
            mp_name_str = ' '.join(mp_words)
            for i in range(len(matched_words) - 1):
                if mp_name_str.find(f"{matched_words[i]} {matched_words[i+1]}") >= 0:
                    match_score += self.config.SEQUENTIAL_MATCH_BONUS
            
            return min(match_score, 1.0), matched_words
        
        return 0, []
    
    def calculate_string_similarity_with_constituency(self, speaker_str, mp_str, mp_constituency, is_hindi=False):
        """Enhanced similarity calculation with constituency consideration using FuzzyWuzzy"""
        if not isinstance(speaker_str, str) or not isinstance(mp_str, str):
            return 0, 0
        
        # Calculate name similarity using existing logic
        name_similarity = self.calculate_string_similarity(speaker_str, mp_str, is_hindi)
        
        # Calculate constituency similarity for English names
        if not is_hindi:
            _, extracted_constituency = self.normalizer.extract_clean_name_and_constituency(speaker_str, is_hindi=False)
            constituency_similarity = self.calculate_constituency_similarity(extracted_constituency, mp_constituency)
        else:
            constituency_similarity = 0
        
        return name_similarity, constituency_similarity
    
    def calculate_string_similarity(self, speaker_str, mp_str, is_hindi=False):
        """Enhanced similarity calculation with FuzzyWuzzy for English names"""
        if not isinstance(speaker_str, str) or not isinstance(mp_str, str):
            return 0
        
        # Extract clean names - pass is_hindi parameter
        clean_speaker = self.normalizer.extract_clean_name_and_constituency(speaker_str, is_hindi=False)[0]  # Speaker name is always processed as English for special cases
        clean_mp = self.normalizer.extract_clean_name_and_constituency(mp_str, is_hindi=is_hindi)[0]
        
        if is_hindi:
            # Keep existing Hindi logic unchanged
            norm_speaker, speaker_no_space = self.normalizer.normalize_hindi_name(clean_speaker)
            norm_mp, mp_no_space = self.normalizer.normalize_hindi_name(clean_mp)
            
            # Prioritize full-string comparison
            if mp_no_space and speaker_no_space:
                if mp_no_space in speaker_no_space:
                    coverage = len(mp_no_space) / len(speaker_no_space) if len(speaker_no_space) > 0 else 0
                    return 0.9 + (coverage * 0.1)
                
                # Use FuzzyWuzzy for Hindi no-space similarity
                no_space_sim = fuzz.ratio(mp_no_space, speaker_no_space) / 100.0
                if no_space_sim >= self.config.NO_SPACE_SIMILARITY_THRESHOLD:
                    return no_space_sim * self.config.NO_SPACE_SIMILARITY_SCALE
            
            # Regular similarity as fallback
            if norm_mp and norm_speaker:
                if norm_mp in norm_speaker:
                    coverage = len(norm_mp) / len(norm_speaker) if len(norm_speaker) > 0 else 0
                    return 0.8 + (coverage * 0.1)
                
                # Use FuzzyWuzzy for Hindi regular similarity
                regular_sim = fuzz.ratio(norm_mp, norm_speaker) / 100.0
                return regular_sim * self.config.REGULAR_SIMILARITY_SCALE
            
            return 0
        else:
            # Enhanced English processing with FuzzyWuzzy
            norm_speaker = self.normalizer.normalize_name(clean_speaker)
            norm_mp = self.normalizer.normalize_name(clean_mp)
            
            if not norm_speaker or not norm_mp:
                return 0
            
            # Check for containment first (backward compatibility)
            if norm_mp in norm_speaker:
                coverage = len(norm_mp) / len(norm_speaker) if len(norm_speaker) > 0 else 0
                return 0.9 + (coverage * 0.1)
            
            # Use FuzzyWuzzy for comprehensive English name similarity
            basic_ratio = fuzz.ratio(norm_mp, norm_speaker) / 100.0
            partial_ratio = fuzz.partial_ratio(norm_mp, norm_speaker) / 100.0
            token_sort_ratio = fuzz.token_sort_ratio(norm_mp, norm_speaker) / 100.0
            token_set_ratio = fuzz.token_set_ratio(norm_mp, norm_speaker) / 100.0
            
            # Weighted combination optimized for name similarity
            weighted_similarity = (
                token_set_ratio * 0.3 +      # Handles extra words, titles
                token_sort_ratio * 0.3 +     # Handles word order
                partial_ratio * 0.25 +       # Handles partial matches
                basic_ratio * 0.15           # Overall similarity
            )
            
            return weighted_similarity

class SpecialRoleDetector:
    """Detects special parliamentary roles like Speaker, Chairperson, Secretary-General, etc."""
    
    def __init__(self, config: Config, scorer: SimilarityScorer):
        self.config = config
        self.scorer = scorer
    
    def check_special_members(self, speaker_name):
        """
        Check if a speaker name matches any special parliamentary role.
        Returns tuple: (is_special_member, role_type, english_title, hindi_title)
        """
        if not isinstance(speaker_name, str) or not speaker_name:
            return False, None, "", ""
    
        for role_type, role_refs in self.config.SPECIAL_ROLES.items():
            # Check Hindi references - direct containment
            for ref in role_refs['hindi']:
                if ref in speaker_name:
                    return True, role_type, self._get_standard_title(role_type, 'english'), self._get_standard_title(role_type, 'hindi')
            
            # Check English references - direct containment
            for ref in role_refs['english']:
                norm_ref = ref.lower().strip()
                norm_speaker_name = speaker_name.lower().strip()
                if norm_ref in norm_speaker_name:
                    return True, role_type, self._get_standard_title(role_type, 'english'), self._get_standard_title(role_type, 'hindi')
        
        # Track the best match across all roles
        best_match = {
            'score': 0,
            'role_type': None,
            'english_title': '',
            'hindi_title': ''
        }
    
        # Check each special role
        for role_type, role_refs in self.config.SPECIAL_ROLES.items():
            # Check English references
            for ref in role_refs['english']:
                sim_score = self.scorer.calculate_string_similarity(speaker_name, ref, is_hindi=False)
                
                if sim_score > best_match['score']:
                    best_match['score'] = sim_score
                    best_match['role_type'] = role_type
                    best_match['english_title'] = self._get_standard_title(role_type, 'english')
                    best_match['hindi_title'] = self._get_standard_title(role_type, 'hindi')

            # Check Hindi references
            for ref in role_refs['hindi']:
                sim_score = self.scorer.calculate_string_similarity(speaker_name, ref, is_hindi=True)
                
                if sim_score > best_match['score']:
                    best_match['score'] = sim_score
                    best_match['role_type'] = role_type
                    best_match['english_title'] = self._get_standard_title(role_type, 'english')
                    best_match['hindi_title'] = self._get_standard_title(role_type, 'hindi')

        # Return the best match if it meets the threshold
        if best_match['score'] >= self.config.SPECIAL_ROLE_DETECTION_THRESHOLD:
            return True, best_match['role_type'], best_match['english_title'], best_match['hindi_title']

        return False, None, "", ""

    def _get_standard_title(self, role_type, language):
        """Get standardized title for a role type"""
        title_mapping = {
            'speaker': {
                'english': 'HON. SPEAKER',
                'hindi': 'माननीय अध्यक्ष'
            },
            'some_hon_members': {
                'english': 'SOME HON. MEMBERS',
                'hindi': 'कुछ माननीय सदस्य'
            },
            'deputy_speaker': {
                'english': 'HON. DEPUTY SPEAKER',
                'hindi': 'माननीय अध्यक्ष'
            },
            'chairperson': {
                'english': 'HON. CHAIRPERSON',
                'hindi': 'माननीय सभापति'
            },
            'secretary_general': {
                'english': 'SECRETARY-GENERAL',
                'hindi': 'महासचिव'
            },
            'president': {
                'english': 'HON. PRESIDENT',
                'hindi': 'माननीय राष्ट्रपति'
            },
            'attorney_general': {
                'english': 'ATTORNEY GENERAL',
                'hindi': 'महान्यायवादी'
            },
            'officer_house': {
                'english': 'OFFICER OF THE HOUSE',
                'hindi': 'सदन अधिकारी'
            }
        }
        
        return title_mapping.get(role_type, {}).get(language, "")
    
    # Keep these methods for backward compatibility (they now use check_special_members internally)
    def is_speaker_chair(self, speaker_name):
        """Check if a speaker name is the Speaker of the House"""
        is_special, role_type, _, _ = self.check_special_members(speaker_name)
        return is_special and role_type == 'speaker'
    
    def is_chair_chair(self, speaker_name):
        """Check if a speaker name is the Chairperson"""
        is_special, role_type, _, _ = self.check_special_members(speaker_name)
        return is_special and role_type == 'chairperson'
    
class DataLoader:
    """Handles loading and combining data from multiple sources"""
    
    @staticmethod
    def load_all_names_data(mp_file_path, rajya_sabha_file_path, mp_eng_name_col_idx, mp_hindi_name_col_idx):
        """Load and combine names from both Lok Sabha members and Rajya Sabha ministers"""
        all_names = []
        
        # Load Lok Sabha members
        try:
            print(f"Reading Lok Sabha MP data from {mp_file_path}...")
            mp_data = pd.read_csv(mp_file_path)
            
            for _, row in mp_data.iterrows():
                eng_name = row.iloc[mp_eng_name_col_idx] if mp_eng_name_col_idx < len(row) else ""
                hindi_name = row.iloc[mp_hindi_name_col_idx] if mp_hindi_name_col_idx < len(row) else ""
                constituency = row.iloc[1] if len(row) > 1 else ""  # Constituency is in column 2 (index 1)
                
                all_names.append({
                    'eng_name': eng_name if isinstance(eng_name, str) else "",
                    'hindi_name': hindi_name if isinstance(hindi_name, str) else "",
                    'constituency': constituency if isinstance(constituency, str) else "",
                    'source': 'lok_sabha'
                })
            
            print(f"Loaded {len(mp_data)} Lok Sabha members")
        
        except Exception as e:
            print(f"Warning: Could not load Lok Sabha data from {mp_file_path}: {str(e)}")
        
        # Load Rajya Sabha ministers
        try:
            print(f"Reading Rajya Sabha ministers data from {rajya_sabha_file_path}...")
            rs_data = pd.read_csv(rajya_sabha_file_path)
            
            for _, row in rs_data.iterrows():
                eng_name = row.iloc[0] if len(row) > 0 and isinstance(row.iloc[0], str) else ""
                hindi_name = row.iloc[1] if len(row) > 1 and isinstance(row.iloc[1], str) else ""
                
                all_names.append({
                    'eng_name': eng_name,
                    'hindi_name': hindi_name,
                    'constituency': "",  # Rajya Sabha ministers don't have constituencies in the same way
                    'source': 'rajya_sabha'
                })
            
            print(f"Loaded {len(rs_data)} Rajya Sabha ministers")
        
        except Exception as e:
            print(f"Warning: Could not load Rajya Sabha data from {rajya_sabha_file_path}: {str(e)}")
        
        print(f"Total names loaded: {len(all_names)}")
        return all_names    

class CachedMPData:
    """Handles preprocessing and caching of MP data to avoid redundant operations"""
    
    def __init__(self, normalizer: NameNormalizer, hindi_processor: HindiTextProcessor):
        self.normalizer = normalizer
        self.hindi_processor = hindi_processor
        self.cached_mp_data = []
        
    def add_nominated_member(self, name):
        """Add a nominated member to the cached data"""
        if not isinstance(name, str) or not name:
            return
        
        # Check if this nominated member already exists in cache
        clean_name = name.replace(' (Nominated)', '').strip()
        for entry in self.cached_mp_data:
            if (entry['original_eng_name'].lower() == clean_name.lower() or 
                entry['original_eng_name'].lower() == name.lower()):
                return  # Already exists
        
        print(f"Adding nominated member to cache: {clean_name}")
        
        # Preprocess the nominated member name
        eng_processed = self._preprocess_english_name(clean_name, "")
        hindi_processed = self._preprocess_hindi_name("")  # No Hindi name for nominated members
        
        cached_entry = {
            'original_eng_name': clean_name,
            'original_hindi_name': "",
            'constituency': "",
            'source': 'nominated',
            'eng_processed': eng_processed,
            'hindi_processed': hindi_processed
        }
        
        self.cached_mp_data.append(cached_entry)
    
    def preprocess_mp_data(self, all_names_data):
        """Preprocess all MP names once and cache the results"""
        print("Preprocessing and caching MP data...")
        
        self.cached_mp_data = []
        
        for name_entry in all_names_data:
            eng_name = name_entry['eng_name']
            hindi_name = name_entry['hindi_name']
            constituency = name_entry.get('constituency', '')
            
            # Skip entries with empty names
            if not eng_name and not hindi_name:
                continue
            
            # Preprocess English name
            eng_processed = self._preprocess_english_name(eng_name, constituency)
            
            # Preprocess Hindi name
            hindi_processed = self._preprocess_hindi_name(hindi_name)
            
            cached_entry = {
                'original_eng_name': eng_name,
                'original_hindi_name': hindi_name,
                'constituency': constituency,
                'source': name_entry['source'],
                'eng_processed': eng_processed,
                'hindi_processed': hindi_processed
            }
            
            self.cached_mp_data.append(cached_entry)
        
        print(f"Cached {len(self.cached_mp_data)} MP entries")
    
    def _preprocess_english_name(self, eng_name, constituency):
        """Preprocess English name and return cached data"""
        if not isinstance(eng_name, str):
            return {
                'clean_name': "",
                'norm_name': "",
                'words': [],
                'constituency': constituency
            }
        
        clean_name = self.normalizer.extract_clean_name_and_constituency(eng_name, is_hindi=False)[0]
        norm_name = self.normalizer.normalize_name(clean_name)
        words = norm_name.split() if norm_name else []
        
        return {
            'clean_name': clean_name,
            'norm_name': norm_name,
            'words': words,
            'constituency': constituency
        }
    
    def _preprocess_hindi_name(self, hindi_name):
        """Preprocess Hindi name and return cached data"""
        if not isinstance(hindi_name, str):
            return {
                'clean_name': "",
                'norm_name': "",
                'norm_name_no_space': "",
                'words': [],
                'consonants_only': "",
                'consonants_no_space': ""
            }
        
        clean_name = self.normalizer.extract_clean_name_and_constituency(hindi_name, is_hindi=True)[0]
        norm_name, norm_name_no_space = self.normalizer.normalize_hindi_name(clean_name)
        words = norm_name.split() if norm_name else []
        
        # Precompute consonant versions
        consonants_only = self.hindi_processor.remove_matras(norm_name)
        consonants_no_space = self.hindi_processor.remove_matras(norm_name_no_space)
        
        return {
            'clean_name': clean_name,
            'norm_name': norm_name,
            'norm_name_no_space': norm_name_no_space,
            'words': words,
            'consonants_only': consonants_only,
            'consonants_no_space': consonants_no_space
        }
    
    def get_cached_data(self):
        """Return the cached MP data"""
        return self.cached_mp_data

class MPNameMatcher:
    """Main class that orchestrates the MP name matching process"""
    
    def __init__(self):
        self.config = Config()
        self.normalizer = NameNormalizer(self.config)
        self.hindi_processor = HindiTextProcessor(self.config)
        self.scorer = SimilarityScorer(self.config, self.normalizer, self.hindi_processor)
        self.role_detector = SpecialRoleDetector(self.config, self.scorer)
        self.cached_mp_data = CachedMPData(self.normalizer, self.hindi_processor)
    
    def _detect_primary_language(self, speaker_name):
        """
        Detect if the speaker name is primarily Hindi or English
        Returns 'hindi' if Hindi characters > 50%, else 'english'
        """
        if not speaker_name:
            return 'english'
        
        # Remove numbers, special characters, and spaces for analysis
        cleaned_name = ''.join(char for char in speaker_name if char.isalpha())
        
        if not cleaned_name:
            return 'english'
        
        hindi_char_count = 0
        total_char_count = len(cleaned_name)
        
        for char in cleaned_name:
            # Check if character is in Devanagari script range (Hindi)
            if '\u0900' <= char <= '\u097F':
                hindi_char_count += 1
        
        hindi_percentage = hindi_char_count / total_char_count if total_char_count > 0 else 0
        
        return 'hindi' if hindi_percentage > 0.5 else 'english'
    
    def find_top_matches_cached(self, speaker_name, cached_mp_data):
        """Enhanced matching using cached MP data with language-first approach"""
        matches = []
        
        # Determine primary language of speaker name
        primary_language = self._detect_primary_language(speaker_name)
        
        # Preprocess speaker name once based on primary language
        if primary_language == 'hindi':
            speaker_clean_eng = self.normalizer.extract_clean_name_and_constituency(speaker_name, is_hindi=True)[0]
            speaker_norm_hindi, speaker_norm_hindi_no_space = self.normalizer.normalize_hindi_name(speaker_clean_eng)
            speaker_extracted_constituency = self.normalizer.extract_clean_name_and_constituency(speaker_name, is_hindi=True)[1]
        else:
            speaker_clean_eng = self.normalizer.extract_clean_name_and_constituency(speaker_name, is_hindi=False)[0]
            speaker_norm_eng = self.normalizer.normalize_name(speaker_clean_eng)
            speaker_extracted_constituency = self.normalizer.extract_clean_name_and_constituency(speaker_name, is_hindi=False)[1]
        
        for mp_entry in cached_mp_data:
            if primary_language == 'hindi':
                # Only process Hindi matching
                hindi_processed = mp_entry['hindi_processed']
                
                # Skip if no Hindi data available
                if not hindi_processed['norm_name']:
                    continue
                
                # Calculate Hindi scores using cached data
                hindi_score, hindi_matched_words = self._calculate_hindi_score_cached(
                    speaker_norm_hindi, speaker_norm_hindi_no_space, hindi_processed)
                
                # Calculate constituency score
                constituency_score = self.scorer.calculate_constituency_similarity(
                    speaker_extracted_constituency, mp_entry['constituency'])
                
                # Calculate string similarities
                hindi_string_sim = self._calculate_hindi_string_similarity_cached(
                    speaker_norm_hindi, speaker_norm_hindi_no_space, hindi_processed)
                
                # Combine scores
                if constituency_score > 0:
                    hindi_name_combined = hindi_score * self.config.WORD_MATCH_WEIGHT + hindi_string_sim * self.config.STRING_SIMILARITY_WEIGHT
                    combined_score = (hindi_name_combined * 0.7) + (constituency_score * 0.3)
                else:
                    combined_score = hindi_score * self.config.WORD_MATCH_WEIGHT + hindi_string_sim * self.config.STRING_SIMILARITY_WEIGHT
                
                matched_words = hindi_matched_words
                string_sim = hindi_string_sim
                
            else:
                # Only process English matching
                eng_processed = mp_entry['eng_processed']
                
                # Skip if no English data available
                if not eng_processed['norm_name']:
                    continue
                
                # Calculate English scores using cached data
                eng_score, eng_matched_words = self._calculate_english_score_cached(
                    speaker_norm_eng, eng_processed)
                
                # Calculate constituency score
                constituency_score = self.scorer.calculate_constituency_similarity(
                    speaker_extracted_constituency, mp_entry['constituency'])
                
                # Calculate string similarities
                eng_string_sim = self._calculate_english_string_similarity_cached(
                    speaker_norm_eng, eng_processed)
                
                # Combine scores
                if constituency_score > 0:
                    eng_name_combined = eng_score * self.config.WORD_MATCH_WEIGHT + eng_string_sim * self.config.STRING_SIMILARITY_WEIGHT
                    combined_score = (eng_name_combined * 0.7) + (constituency_score * 0.3)
                else:
                    combined_score = eng_score * self.config.WORD_MATCH_WEIGHT + eng_string_sim * self.config.STRING_SIMILARITY_WEIGHT
                
                matched_words = eng_matched_words
                string_sim = eng_string_sim
            
            # Add to matches if score is valid
            if combined_score > 0 or mp_entry['original_eng_name'] or mp_entry['original_hindi_name']:
                matches.append({
                    'score': combined_score,
                    'eng_name': mp_entry['original_eng_name'],
                    'hindi_name': mp_entry['original_hindi_name'],
                    'matched_words': matched_words,
                    'word_count': len(matched_words),
                    'string_sim': string_sim,
                    'source': mp_entry['source'],
                    'primary_language': primary_language
                })
        
        # Sort and return top matches
        matches.sort(key=lambda x: (-x['score'], -x['word_count'], -x['string_sim']))
        valid_matches = [m for m in matches if m['score'] > 0 or m['eng_name'] or m['hindi_name']]
        
        top_matches = []
        for i in range(min(2, len(valid_matches))):
            match = valid_matches[i]
            top_matches.append((match['score'], match['eng_name'], match['hindi_name']))
        
        while len(top_matches) < 2:
            if len(top_matches) == 1:
                top_matches.append(top_matches[0])
            else:
                top_matches.append((0, "", ""))
        
        return top_matches
    
    def _calculate_english_score_cached(self, speaker_norm, eng_processed):
        """Calculate English word match score using cached data"""
        if not speaker_norm or not eng_processed['norm_name']:
            return 0, []
        
        # Check for exact match
        if speaker_norm == eng_processed['norm_name']:
            return self.config.EXACT_MATCH_SCORE, eng_processed['words']
        
        # Check for containment
        if eng_processed['norm_name'] in speaker_norm:
            coverage = len(eng_processed['norm_name']) / len(speaker_norm) if len(speaker_norm) > 0 else 0
            base_score = self.config.CONTAINMENT_BASE_SCORE
            coverage_bonus = coverage * self.config.COVERAGE_BONUS_MULTIPLIER
            return min(base_score + coverage_bonus, 1.0), eng_processed['words']
        
        # Use FuzzyWuzzy for comprehensive matching
        from fuzzywuzzy import fuzz
        basic_ratio = fuzz.ratio(eng_processed['norm_name'], speaker_norm) / 100.0
        partial_ratio = fuzz.partial_ratio(eng_processed['norm_name'], speaker_norm) / 100.0
        token_sort_ratio = fuzz.token_sort_ratio(eng_processed['norm_name'], speaker_norm) / 100.0
        token_set_ratio = fuzz.token_set_ratio(eng_processed['norm_name'], speaker_norm) / 100.0
        
        weighted_score = (
            token_set_ratio * 0.35 +
            token_sort_ratio * 0.25 +
            partial_ratio * 0.25 +
            basic_ratio * 0.15
        )
        
        if weighted_score >= self.config.FALLBACK_THRESHOLD:
            return weighted_score, eng_processed['words']
        
        return 0, []
    
    def _calculate_hindi_score_cached(self, speaker_norm, speaker_norm_no_space, hindi_processed):
        """Calculate Hindi word match score using cached data"""
        if not hindi_processed['norm_name']:
            return 0, []
        
        # Check no-space containment
        if (speaker_norm_no_space and hindi_processed['norm_name_no_space'] and 
            hindi_processed['norm_name_no_space'] in speaker_norm_no_space):
            coverage = len(hindi_processed['norm_name_no_space']) / len(speaker_norm_no_space) if len(speaker_norm_no_space) > 0 else 0
            base_score = self.config.NO_SPACE_CONTAINMENT_SCORE
            coverage_bonus = coverage * self.config.COVERAGE_BONUS_MULTIPLIER
            return min(base_score + coverage_bonus, 1.0), hindi_processed['words']
        
        # Check exact match (no spaces)
        if speaker_norm_no_space == hindi_processed['norm_name_no_space']:
            return self.config.EXACT_MATCH_SCORE, hindi_processed['words']
        
        # Check no-space similarity
        if speaker_norm_no_space and hindi_processed['norm_name_no_space']:
            from fuzzywuzzy import fuzz
            no_space_similarity = fuzz.ratio(hindi_processed['norm_name_no_space'], speaker_norm_no_space) / 100.0
            if no_space_similarity >= self.config.NO_SPACE_SIMILARITY_THRESHOLD:
                return no_space_similarity * self.config.NO_SPACE_SIMILARITY_SCALE, hindi_processed['words']
        
        # Check regular containment
        if hindi_processed['norm_name'] in speaker_norm:
            coverage = len(hindi_processed['norm_name']) / len(speaker_norm) if len(speaker_norm) > 0 else 0
            base_score = self.config.CONTAINMENT_BASE_SCORE
            coverage_bonus = coverage * self.config.COVERAGE_BONUS_MULTIPLIER
            return min(base_score + coverage_bonus, 1.0), hindi_processed['words']
        
        # Consonant-based comparison
        if (speaker_norm_no_space and hindi_processed['consonants_no_space'] and 
            hindi_processed['consonants_no_space']):
            from difflib import SequenceMatcher
            speaker_consonants = self.hindi_processor.remove_matras(speaker_norm_no_space)
            if speaker_consonants == hindi_processed['consonants_no_space']:
                return self.config.CONSONANT_MATCH_SCORE, hindi_processed['words']
            
            consonant_similarity = SequenceMatcher(None, speaker_consonants, hindi_processed['consonants_no_space']).ratio()
            if consonant_similarity >= self.config.CONSONANT_SIMILARITY_THRESHOLD:
                return consonant_similarity * self.config.CONSONANT_SIMILARITY_SCALE, hindi_processed['words']
        
        return 0, []
    
    def _calculate_english_string_similarity_cached(self, speaker_norm, eng_processed):
        """Calculate English string similarity using cached data"""
        if not speaker_norm or not eng_processed['norm_name']:
            return 0
        
        # Check for containment
        if eng_processed['norm_name'] in speaker_norm:
            coverage = len(eng_processed['norm_name']) / len(speaker_norm) if len(speaker_norm) > 0 else 0
            return 0.9 + (coverage * 0.1)
        
        # Use FuzzyWuzzy
        from fuzzywuzzy import fuzz
        basic_ratio = fuzz.ratio(eng_processed['norm_name'], speaker_norm) / 100.0
        partial_ratio = fuzz.partial_ratio(eng_processed['norm_name'], speaker_norm) / 100.0
        token_sort_ratio = fuzz.token_sort_ratio(eng_processed['norm_name'], speaker_norm) / 100.0
        token_set_ratio = fuzz.token_set_ratio(eng_processed['norm_name'], speaker_norm) / 100.0
        
        weighted_similarity = (
            token_set_ratio * 0.3 +
            token_sort_ratio * 0.3 +
            partial_ratio * 0.25 +
            basic_ratio * 0.15
        )
        
        return weighted_similarity
    
    def _calculate_hindi_string_similarity_cached(self, speaker_norm, speaker_norm_no_space, hindi_processed):
        """Calculate Hindi string similarity using cached data"""
        if not hindi_processed['norm_name']:
            return 0
        
        # Check no-space containment
        if (speaker_norm_no_space and hindi_processed['norm_name_no_space'] and 
            hindi_processed['norm_name_no_space'] in speaker_norm_no_space):
            coverage = len(hindi_processed['norm_name_no_space']) / len(speaker_norm_no_space) if len(speaker_norm_no_space) > 0 else 0
            return 0.9 + (coverage * 0.1)
        
        # Use FuzzyWuzzy for no-space similarity
        if speaker_norm_no_space and hindi_processed['norm_name_no_space']:
            from fuzzywuzzy import fuzz
            no_space_sim = fuzz.ratio(hindi_processed['norm_name_no_space'], speaker_norm_no_space) / 100.0
            if no_space_sim >= self.config.NO_SPACE_SIMILARITY_THRESHOLD:
                return no_space_sim * self.config.NO_SPACE_SIMILARITY_SCALE
        
        # Regular similarity
        if hindi_processed['norm_name'] in speaker_norm:
            coverage = len(hindi_processed['norm_name']) / len(speaker_norm) if len(speaker_norm) > 0 else 0
            return 0.8 + (coverage * 0.1)
        
        if speaker_norm and hindi_processed['norm_name']:
            from fuzzywuzzy import fuzz
            regular_sim = fuzz.ratio(hindi_processed['norm_name'], speaker_norm) / 100.0
            return regular_sim * self.config.REGULAR_SIMILARITY_SCALE
        
        return 0

    def match_mp_names(self, mp_file_path, speech_file_path, output_path, rajya_sabha_file_path, mp_eng_name_col_idx=2, mp_hindi_name_col_idx=3, speech_speaker_col_idx=2):
        """Main matching function with caching for improved performance and nominated member handling"""
        try:
            # Load all names data from both sources
            all_names_data = DataLoader.load_all_names_data(
                mp_file_path, rajya_sabha_file_path, mp_eng_name_col_idx, mp_hindi_name_col_idx)
            
            # Preprocess and cache MP data once
            self.cached_mp_data.preprocess_mp_data(all_names_data)
            
            print(f"Reading speech data from {speech_file_path}...")
            speech_data = pd.read_csv(speech_file_path)
            
            print(f"Found {len(self.cached_mp_data.get_cached_data())} cached MP names and {len(speech_data)} speech entries")
            
            # Create a new dataframe for output
            result_df = speech_data.copy()
            
            # Insert new columns after the speaker column
            new_columns = ['eng name(pref 1)', 'hind name(pref 1)', 'eng name(pref 2)', 'hind name(pref 2)']
            for i, col_name in enumerate(new_columns):
                result_df.insert(speech_speaker_col_idx + 1 + i, col_name, "")
            
            # Process each row in speech data
            total_rows = len(speech_data)
            print(f"Processing {total_rows} speech entries...")
            
            for idx, row in speech_data.iterrows():
                speaker_name = row.iloc[speech_speaker_col_idx]
                
                if pd.isna(speaker_name):
                    continue
                    
                if speaker_name:
                    # First check if it's a nominated member
                    clean_name, _ = self.normalizer.extract_clean_name_and_constituency(speaker_name, is_hindi=False)
                    if clean_name.startswith("NOMINATED_MEMBER:"):
                        # Extract the actual name and add (Nominated)
                        actual_name = clean_name.replace("NOMINATED_MEMBER:", "").strip()
                        formatted_name = ' '.join(word.capitalize() for word in actual_name.split())
                        nominated_name = f"{formatted_name} (Nominated)"
                        
                        # Add this nominated member to the cache for future matching
                        self.cached_mp_data.add_nominated_member(formatted_name)
                        
                        result_df.at[idx, 'eng name(pref 1)'] = re.sub(r'\s*\(.*?\)', '', nominated_name).strip() + ' (Nominated)'
                        result_df.at[idx, 'hind name(pref 1)'] = "nan"
                        result_df.at[idx, 'eng name(pref 2)'] = re.sub(r'\s*\(.*?\)', '', nominated_name).strip() + ' (Nominated)'
                        result_df.at[idx, 'hind name(pref 2)'] = "nan"
                    elif clean_name.startswith("STATEMENT:"):
                        # For submissions by members, keep the original speaker name as is
                        result_df.at[idx, 'eng name(pref 1)'] = f"{speaker_name} (Statement)"
                        result_df.at[idx, 'hind name(pref 1)'] = "nan"
                        result_df.at[idx, 'eng name(pref 2)'] = f"{speaker_name} (Statement)"
                        result_df.at[idx, 'hind name(pref 2)'] = "nan"
                    else:
                        # Check if the speaker is a special parliamentary member
                        is_special, role_type, eng_title, hindi_title = self.role_detector.check_special_members(speaker_name)
                        
                        if is_special:
                            result_df.at[idx, 'eng name(pref 1)'] = eng_title
                            result_df.at[idx, 'hind name(pref 1)'] = hindi_title
                            result_df.at[idx, 'eng name(pref 2)'] = eng_title
                            result_df.at[idx, 'hind name(pref 2)'] = hindi_title
                        else:
                            # Use cached MP name matching with language-first approach
                            cached_data = self.cached_mp_data.get_cached_data()
                            top_matches = self.find_top_matches_cached(speaker_name, cached_data)
                            
                            # Check if the match is a nominated member and format appropriately
                            match1_eng = top_matches[0][1]
                            match1_hindi = top_matches[0][2]
                            match2_eng = top_matches[1][1]
                            match2_hindi = top_matches[1][2]
                            
                            # Check if matches are from nominated source and add (Nominated) tag
                            cached_data = self.cached_mp_data.get_cached_data()
                            for entry in cached_data:
                                if entry['source'] == 'nominated':
                                    if entry['original_eng_name'] == match1_eng:
                                        match1_eng = f"{match1_eng} (Nominated)"
                                    if entry['original_eng_name'] == match2_eng:
                                        match2_eng = f"{match2_eng} (Nominated)"
                            
                            # Assign matches to the result dataframe
                            result_df.at[idx, 'eng name(pref 1)'] = match1_eng
                            result_df.at[idx, 'hind name(pref 1)'] = match1_hindi
                            result_df.at[idx, 'eng name(pref 2)'] = match2_eng
                            result_df.at[idx, 'hind name(pref 2)'] = match2_hindi
                
                # Print progress
                if (idx + 1) % 100 == 0 or idx == total_rows - 1:
                    print(f"Progress: {idx + 1}/{total_rows} entries processed ({(idx + 1)/total_rows*100:.1f}%)")
            
            # Save to CSV
            print(f"Saving results to {output_path}...")
            result_df.to_csv(output_path, index=False)
            
            print(f"Processing complete. Output saved to {output_path}")
            return True
        
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

# Command line interface
if __name__ == "__main__":
    print(f"Starting enhanced MP name matching process with dual data sources...")
    
    # Default file paths and column indices
    mp_file_path = "16th_Lok_Sabha_Members.csv"
    rajya_sabha_file_path = "rajyasabha_ministers_16.csv"
    speech_file_path = "lsd_16_04_2015-05-13.csv"
    output_path = "matched_speeches.csv"
    
    # Default column indices (0-based)
    mp_eng_name_col_idx = 2
    mp_hindi_name_col_idx = 3
    speech_speaker_col_idx = 2
    
    # Process command line arguments
    if len(sys.argv) > 1:
        mp_file_path = sys.argv[1]
    if len(sys.argv) > 2:
        rajya_sabha_file_path = sys.argv[2]
    if len(sys.argv) > 3:
        speech_file_path = sys.argv[3]
    if len(sys.argv) > 4:
        output_path = sys.argv[4]
    if len(sys.argv) > 5:
        mp_eng_name_col_idx = int(sys.argv[5])
    if len(sys.argv) > 6:
        mp_hindi_name_col_idx = int(sys.argv[6])
    if len(sys.argv) > 7:
        speech_speaker_col_idx = int(sys.argv[7])
    
    print(f"Lok Sabha MP data file: {mp_file_path}")
    print(f"Rajya Sabha ministers data file: {rajya_sabha_file_path}")
    print(f"Speech data file: {speech_file_path}")
    print(f"Output file: {output_path}")
    print(f"MP English name column index: {mp_eng_name_col_idx}")
    print(f"MP Hindi name column index: {mp_hindi_name_col_idx}")
    print(f"Speech speaker column index: {speech_speaker_col_idx}")
    
    # Initialize matcher and run
    matcher = MPNameMatcher()
    success = matcher.match_mp_names(
        mp_file_path, 
        speech_file_path, 
        output_path,
        rajya_sabha_file_path,
        mp_eng_name_col_idx, 
        mp_hindi_name_col_idx, 
        speech_speaker_col_idx
    )
    
    if success:
        print("MP name matching completed successfully!")
    else:
        print("MP name matching failed. Please check the error messages above.")
        sys.exit(1)
