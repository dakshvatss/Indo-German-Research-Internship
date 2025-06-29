import pandas as pd
import re
import string
from difflib import SequenceMatcher
import sys
import logging

# Configuration - TODO: Move to external config file (JSON/YAML) for better maintainability
class Config:
    # Similarity thresholds
    SPECIAL_ROLE_DETECTION_THRESHOLD = 0.6
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
    ENGLISH_TITLES = ['shri', 'shrimati', 'dr', 'prof', 'mr', 'mrs', 'ms', 'the minister of']
    
    SPECIAL_ROLES = {
        'speaker': {
            'english': ['hon. speaker', 'madam speaker', 'speaker'],
            'hindi': ['माननीय अध्यक्ष', 'अध्यक्ष']
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
    
    def extract_clean_name(self, speaker_name):
        """Extract the core name from speaker string by removing common titles and parenthetical info"""
        if not isinstance(speaker_name, str):
            return ""
        
        clean_name = speaker_name.lower().strip()
        
        # Split into words and filter out titles
        words = clean_name.split()
        filtered_words = []
        
        for word in words:
            word_clean = word.strip('.,')
            if (word_clean not in self.config.HINDI_TITLES and 
                word_clean.lower() not in self.config.ENGLISH_TITLES):
                filtered_words.append(word_clean)
        
        return ' '.join(filtered_words).strip()

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
    """Handles similarity calculations between names"""
    
    def __init__(self, config: Config, normalizer: NameNormalizer, hindi_processor: HindiTextProcessor):
        self.config = config
        self.normalizer = normalizer
        self.hindi_processor = hindi_processor
    
    def check_name_words_match(self, speaker_name, mp_name, is_hindi=False):
        """Enhanced matching with improved Hindi full-string comparison"""
        if not isinstance(speaker_name, str) or not isinstance(mp_name, str):
            return 0, []
        
        # Extract clean names
        clean_speaker = self.normalizer.extract_clean_name(speaker_name)
        clean_mp = self.normalizer.extract_clean_name(mp_name) if not is_hindi else mp_name
        
        # For Hindi names, prioritize full-string comparison
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
                
                # Check string similarity for no-space versions
                no_space_similarity = SequenceMatcher(None, mp_no_space, speaker_no_space).ratio()
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
            # Regular English name processing
            norm_speaker = self.normalizer.normalize_name(clean_speaker)
            norm_mp = self.normalizer.normalize_name(clean_mp)
            
            # Check for full name match
            if norm_speaker == norm_mp:
                return self.config.EXACT_MATCH_SCORE, norm_mp.split()
            
            # Check if MP name is contained in speaker name
            if norm_mp in norm_speaker:
                coverage = len(norm_mp) / len(norm_speaker) if len(norm_speaker) > 0 else 0
                base_score = self.config.CONTAINMENT_BASE_SCORE
                coverage_bonus = coverage * self.config.COVERAGE_BONUS_MULTIPLIER
                return min(base_score + coverage_bonus, 1.0), norm_mp.split()
        
        # Fallback to word-by-word matching
        return self._word_by_word_matching(norm_speaker, norm_mp, is_hindi)
    
    def _word_by_word_matching(self, norm_speaker, norm_mp, is_hindi):
        """Fallback word-by-word matching logic"""
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
                    if mp_word == speaker_word:
                        best_match_score = 1.0
                        best_match_word = mp_word
                        break
            
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
    
    def calculate_string_similarity(self, speaker_str, mp_str, is_hindi=False):
        """Enhanced similarity calculation with improved Hindi full-string comparison"""
        if not isinstance(speaker_str, str) or not isinstance(mp_str, str):
            return 0
        
        clean_speaker = self.normalizer.extract_clean_name(speaker_str)
        clean_mp = self.normalizer.extract_clean_name(mp_str) if not is_hindi else mp_str
        
        if is_hindi:
            norm_speaker, speaker_no_space = self.normalizer.normalize_hindi_name(clean_speaker)
            norm_mp, mp_no_space = self.normalizer.normalize_hindi_name(clean_mp)
            
            # Prioritize full-string comparison
            if mp_no_space and speaker_no_space:
                if mp_no_space in speaker_no_space:
                    coverage = len(mp_no_space) / len(speaker_no_space) if len(speaker_no_space) > 0 else 0
                    return 0.9 + (coverage * 0.1)
                
                no_space_sim = SequenceMatcher(None, mp_no_space, speaker_no_space).ratio()
                if no_space_sim >= self.config.NO_SPACE_SIMILARITY_THRESHOLD:
                    return no_space_sim * self.config.NO_SPACE_SIMILARITY_SCALE
            
            # Regular similarity as fallback
            if norm_mp and norm_speaker:
                if norm_mp in norm_speaker:
                    coverage = len(norm_mp) / len(norm_speaker) if len(norm_speaker) > 0 else 0
                    return 0.8 + (coverage * 0.1)
                
                regular_sim = SequenceMatcher(None, norm_mp, norm_speaker).ratio()
                return regular_sim * self.config.REGULAR_SIMILARITY_SCALE
            
            return 0
        else:
            # English processing
            norm_speaker = self.normalizer.normalize_name(clean_speaker)
            norm_mp = self.normalizer.normalize_name(clean_mp)
            
            if not norm_speaker or not norm_mp:
                return 0
            
            if norm_mp in norm_speaker:
                coverage = len(norm_mp) / len(norm_speaker) if len(norm_speaker) > 0 else 0
                return 0.9 + (coverage * 0.1)
            
            return SequenceMatcher(None, norm_mp, norm_speaker).ratio()

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
        
        # Check each special role
        for role_type, role_refs in self.config.SPECIAL_ROLES.items():
            # Check English references
            for ref in role_refs['english']:
                eng_score, _ = self.scorer.check_name_words_match(speaker_name, ref, is_hindi=False)
                sim_score = self.scorer.calculate_string_similarity(speaker_name, ref, is_hindi=False)
                combined_score = eng_score * self.config.WORD_MATCH_WEIGHT + sim_score * self.config.STRING_SIMILARITY_WEIGHT
                
                if combined_score >= self.config.SPECIAL_ROLE_DETECTION_THRESHOLD:
                    return True, role_type, self._get_standard_title(role_type, 'english'), self._get_standard_title(role_type, 'hindi')
            
            # Check Hindi references
            for ref in role_refs['hindi']:
                hindi_score, _ = self.scorer.check_name_words_match(speaker_name, ref, is_hindi=True)
                sim_score = self.scorer.calculate_string_similarity(speaker_name, ref, is_hindi=True)
                combined_score = hindi_score * self.config.WORD_MATCH_WEIGHT + sim_score * self.config.STRING_SIMILARITY_WEIGHT
                
                if combined_score >= self.config.SPECIAL_ROLE_DETECTION_THRESHOLD:
                    return True, role_type, self._get_standard_title(role_type, 'english'), self._get_standard_title(role_type, 'hindi')
        
        return False, None, "", ""
    
    def _get_standard_title(self, role_type, language):
        """Get standardized title for a role type"""
        title_mapping = {
            'speaker': {
                'english': 'HON. SPEAKER',
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
    # TODO: Add support for different file formats (Excel, JSON)
    # TODO: Add data validation and schema checking
    # TODO: Add streaming for large files
    
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
                
                all_names.append({
                    'eng_name': eng_name if isinstance(eng_name, str) else "",
                    'hindi_name': hindi_name if isinstance(hindi_name, str) else "",
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
                    'source': 'rajya_sabha'
                })
            
            print(f"Loaded {len(rs_data)} Rajya Sabha ministers")
        
        except Exception as e:
            print(f"Warning: Could not load Rajya Sabha data from {rajya_sabha_file_path}: {str(e)}")
        
        print(f"Total names loaded: {len(all_names)}")
        return all_names

class MPNameMatcher:
    """Main class that orchestrates the MP name matching process"""
    
    def __init__(self):
        self.config = Config()
        self.normalizer = NameNormalizer(self.config)
        self.hindi_processor = HindiTextProcessor(self.config)
        self.scorer = SimilarityScorer(self.config, self.normalizer, self.hindi_processor)
        self.role_detector = SpecialRoleDetector(self.config, self.scorer)
        
        # TODO: Add caching for normalized names and similarity scores
        # TODO: Add performance monitoring and metrics collection
    
    def find_top_matches(self, speaker_name, all_names_data):
        """Enhanced matching with improved scoring for Hindi names"""
        matches = []
        
        for name_entry in all_names_data:
            eng_name = name_entry['eng_name']
            hindi_name = name_entry['hindi_name']
            
            # Skip entries with empty names
            if not eng_name and not hindi_name:
                continue
                
            # Check word matches for English name
            eng_score, eng_matched_words = self.scorer.check_name_words_match(speaker_name, eng_name, is_hindi=False)
            
            # Check word matches for Hindi name
            hindi_score, hindi_matched_words = self.scorer.check_name_words_match(
                speaker_name, hindi_name, is_hindi=True) if isinstance(hindi_name, str) else (0, [])
            
            # Add string similarity
            eng_string_sim = self.scorer.calculate_string_similarity(speaker_name, eng_name, is_hindi=False)
            hindi_string_sim = self.scorer.calculate_string_similarity(
                speaker_name, hindi_name, is_hindi=True) if isinstance(hindi_name, str) else 0
            
            # Combine scores with weights
            eng_combined = eng_score * self.config.WORD_MATCH_WEIGHT + eng_string_sim * self.config.STRING_SIMILARITY_WEIGHT
            hindi_combined = hindi_score * self.config.WORD_MATCH_WEIGHT + hindi_string_sim * self.config.STRING_SIMILARITY_WEIGHT
            
            # Determine which match is better
            if eng_combined >= hindi_combined:
                score = eng_combined
                matched_words = eng_matched_words
                string_sim = eng_string_sim
            else:
                score = hindi_combined
                matched_words = hindi_matched_words
                string_sim = hindi_string_sim
            
            # Only add matches with meaningful scores or names
            if score > 0 or eng_name or hindi_name:
                matches.append({
                    'score': score,
                    'eng_name': eng_name,
                    'hindi_name': hindi_name,
                    'matched_words': matched_words,
                    'word_count': len(matched_words),
                    'string_sim': string_sim,
                    'source': name_entry['source']
                })
        
        # Sort matches by combined score first, then word count, then string similarity
        matches.sort(key=lambda x: (-x['score'], -x['word_count'], -x['string_sim']))
        
        # Filter out matches with empty names AND zero scores
        valid_matches = [m for m in matches if m['score'] > 0 or m['eng_name'] or m['hindi_name']]
        
        # Return top 2 valid matches
        top_matches = []
        for i in range(min(2, len(valid_matches))):
            match = valid_matches[i]
            top_matches.append((match['score'], match['eng_name'], match['hindi_name']))
        
        # Ensure we always return exactly 2 matches
        while len(top_matches) < 2:
            if len(top_matches) == 1:
                # Duplicate the first match if we only have one
                top_matches.append(top_matches[0])
            else:
                # Add empty match if we have none
                top_matches.append((0, "", ""))
        
        return top_matches

    def match_mp_names(self, mp_file_path, speech_file_path, output_path, rajya_sabha_file_path,
                      mp_eng_name_col_idx=2, mp_hindi_name_col_idx=3, speech_speaker_col_idx=2):
        """Main matching function with enhanced Hindi comparison logic and dual data sources"""
        try:
            # Load all names data from both sources
            all_names_data = DataLoader.load_all_names_data(
                mp_file_path, rajya_sabha_file_path, mp_eng_name_col_idx, mp_hindi_name_col_idx)
            
            print(f"Reading speech data from {speech_file_path}...")
            speech_data = pd.read_csv(speech_file_path)
            
            print(f"Found {len(all_names_data)} total names and {len(speech_data)} speech entries")
            
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
                    # Check if the speaker is a special parliamentary member
                    is_special, role_type, eng_title, hindi_title = self.role_detector.check_special_members(speaker_name)
                    
                    if is_special:
                        result_df.at[idx, 'eng name(pref 1)'] = eng_title
                        result_df.at[idx, 'hind name(pref 1)'] = hindi_title
                        result_df.at[idx, 'eng name(pref 2)'] = eng_title
                        result_df.at[idx, 'hind name(pref 2)'] = hindi_title
                    else:
                        # Use enhanced MP name matching with combined data
                        top_matches = self.find_top_matches(speaker_name, all_names_data)
                        
                        # Assign matches to the result dataframe
                        result_df.at[idx, 'eng name(pref 1)'] = top_matches[0][1]
                        result_df.at[idx, 'hind name(pref 1)'] = top_matches[0][2]
                        result_df.at[idx, 'eng name(pref 2)'] = top_matches[1][1]
                        result_df.at[idx, 'hind name(pref 2)'] = top_matches[1][2]
                
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
    mp_file_path = "15th_Lok_Sabha_Members.csv"
    rajya_sabha_file_path = "rajyasabha_ministers_15.csv"
    speech_file_path = "lsd_15_13_2013-03-22.csv"
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
