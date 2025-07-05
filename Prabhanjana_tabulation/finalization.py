import pandas as pd
import re
from collections import Counter
import langdetect
from datetime import datetime

def standardize_columns(df):
    """
    Standardize column names by removing whitespace
    
    Args:
        df (DataFrame): DataFrame to standardize
    
    Returns:
        DataFrame: DataFrame with standardized column names
    """
    df.columns = [col.strip() for col in df.columns]
    return df

def clean_name(name):
    """
    Clean name by removing common prefixes
    
    Args:
        name (str): Name to clean
    
    Returns:
        str: Cleaned name
    """
    if not isinstance(name, str):
        return name
    return name.replace('Shri ', '').replace('Smt. ', '').replace('Dr. ', '').strip()

def is_null_or_empty(value):
    """
    Check if value is null, nan, or empty string
    
    Args:
        value: Value to check
    
    Returns:
        bool: True if null/empty, False otherwise
    """
    return pd.isna(value) or value == '' or str(value).lower() == 'nan'

def find_member_in_lookup(name, lookup_dict):
    """
    Find member in lookup dictionary with fallback to cleaned name
    
    Args:
        name (str): Name to search for
        lookup_dict (dict): Lookup dictionary
    
    Returns:
        dict or None: Member info if found, None otherwise
    """
    if is_null_or_empty(name):
        return None
    
    name = str(name).strip()
    
    # Try exact match first
    member_info = lookup_dict.get(name)
    if member_info:
        return member_info
    
    # Try cleaned name
    clean_name_str = clean_name(name)
    if clean_name_str != name:
        member_info = lookup_dict.get(clean_name_str)
        if member_info:
            return member_info
    
    return None

def load_members_list(members_file):
    """
    Load the members list CSV file and create a lookup dictionary
    
    Args:
        members_file (str): Path to the members CSV file
    
    Returns:
        dict: Dictionary with MP names as keys and member info as values
    """
    try:
        members_df = pd.read_csv(members_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Members file not found: {members_file}")
    
    # Standardize column names
    members_df = standardize_columns(members_df)
    
    # Create lookup dictionary using both English and Hindi names
    members_lookup = {}
    
    for _, row in members_df.iterrows():
        mp_name = str(row.get('MP Name', '')).strip()
        mp_name_hindi = str(row.get('MP Name (Hindi)', '')).strip()
        initial_party = str(row.get('Initial Party', '')).strip()
        date = str(row.get('Date', '')).strip()
        new_party = str(row.get('New Party', '')).strip()
        
        member_info = {
            'initial_party': initial_party,
            'date': date,
            'new_party': new_party,
            'hindi_name': mp_name_hindi
        }
        
        # Add both English and Hindi names as keys (handle variations)
        if not is_null_or_empty(mp_name):
            members_lookup[mp_name] = member_info
            # Also add name variations (remove common prefixes/suffixes)
            clean_name_str = clean_name(mp_name)
            if clean_name_str != mp_name:
                members_lookup[clean_name_str] = member_info
        
        if not is_null_or_empty(mp_name_hindi):
            members_lookup[mp_name_hindi] = member_info
    
    return members_lookup

def load_rajya_sabha_list(rajya_sabha_file):
    """
    Load the Rajya Sabha members list CSV file and create a lookup dictionary
    
    Args:
        rajya_sabha_file (str): Path to the Rajya Sabha CSV file
    
    Returns:
        dict: Dictionary with Rajya Sabha member names as keys and member info as values
    """
    try:
        rajya_sabha_df = pd.read_csv(rajya_sabha_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Rajya Sabha file not found: {rajya_sabha_file}")
    
    # Standardize column names
    rajya_sabha_df = standardize_columns(rajya_sabha_df)
    
    rajya_sabha_lookup = {}
    
    for _, row in rajya_sabha_df.iterrows():
        minister_name = str(row.get('Minister', '')).strip()
        hindi_name = str(row.get('Hindi names', '')).strip()
        party = str(row.get('Party', '')).strip()
        
        member_info = {
            'party': party,
            'hindi_name': hindi_name,
            'source': 'rajya_sabha'
        }
        
        # Add English name as key
        if not is_null_or_empty(minister_name):
            rajya_sabha_lookup[minister_name] = member_info
            # Also add name variations (remove common prefixes/suffixes)
            clean_name_str = clean_name(minister_name)
            if clean_name_str != minister_name:
                rajya_sabha_lookup[clean_name_str] = member_info
        
        # Add Hindi name as key
        if not is_null_or_empty(hindi_name):
            rajya_sabha_lookup[hindi_name] = member_info
    
    return rajya_sabha_lookup

def extract_date_from_metadata(metadata_list):
    """
    Extract the most common date from metadata fields
    
    Args:
        metadata_list: List of metadata strings
    
    Returns:
        str: Most common date found
    """
    date_pattern = r'\d{2}\.\d{2}\.\d{4}'
    dates_found = []
    
    for metadata in metadata_list:
        if not is_null_or_empty(metadata):
            matches = re.findall(date_pattern, str(metadata))
            dates_found.extend(matches)
    
    if dates_found:
        # Return the most common date
        date_counter = Counter(dates_found)
        return date_counter.most_common(1)[0][0]
    
    return "Unknown"

def parse_party_changes(date_str, new_party_str, reference_date):
    """
    Parse party change information and determine current party using pandas datetime
    
    Args:
        date_str: String containing comma-separated dates
        new_party_str: String containing comma-separated new parties
        reference_date: Date to compare against (from metadata)
    
    Returns:
        str: Current party at the reference date
    """
    if is_null_or_empty(date_str) or is_null_or_empty(new_party_str):
        return None
    
    try:
        # Parse reference date
        ref_date = pd.to_datetime(reference_date, format='%d.%m.%Y')
        
        # Split dates and parties
        dates = [d.strip() for d in str(date_str).split(',')]
        parties = [p.strip() for p in str(new_party_str).split(',')]
        
        # Create DataFrame for easier processing
        if len(dates) != len(parties):
            return None
        
        changes_df = pd.DataFrame({
            'date_str': dates,
            'party': parties
        })
        
        # Convert dates to datetime
        changes_df['date'] = pd.to_datetime(changes_df['date_str'], format='%d.%m.%Y', errors='coerce')
        
        # Filter changes that occurred before or on reference date
        valid_changes = changes_df[changes_df['date'] <= ref_date]
        
        if valid_changes.empty:
            return None
        
        # Get the latest change
        latest_change = valid_changes.loc[valid_changes['date'].idxmax()]
        return latest_change['party']
        
    except:
        return None

def is_special_character(name):
    """
    Check if the name corresponds to a special parliamentary character
    
    Args:
        name (str): Name to check
    
    Returns:
        bool: True if it's a special character, False otherwise
    """
    special_characters = {
        'HON. SPEAKER': True,
        'माननीय अध्यक्ष': True,
        'DEPUTY SPEAKER': True,
        'माननीय उपाध्यक्ष': True,
        'SOME HON. MEMBERS': True,
        'कुछ माननीय सदस्य': True,
        'HON. CHAIRPERSON': True,
        'माननीय सभापति': True,
        'SECRETARY-GENERAL': True,
        'महासचिव': True,
        'HON. PRESIDENT': True,
        'माननीय राष्ट्रपति': True,
        'ATTORNEY GENERAL': True,
        'महान्यायवादी': True,
        'OFFICER OF THE HOUSE': True,
        'सदन अधिकारी': True
    }
    
    return special_characters.get(name, False)

def has_special_tag(name):
    """
    Check if the name contains '(Nominated)' tag
    
    Args:
        name (str): Name to check
    
    Returns:
        bool: True if name contains (Nominated), False otherwise
    """
    if not isinstance(name, str):
        return False
    return '(Nominated)' in name or '(Statement)' in name

def process_csv(input_file, members_file, output_file, rajya_sabha_file=None):
    """
    Process the CSV file with member information and enhanced logic
    
    Args:
        input_file (str): Path to the input CSV file
        members_file (str): Path to the members CSV file
        output_file (str): Path to save the processed CSV file
        rajya_sabha_file (str): Path to the Rajya Sabha CSV file (optional)
    """
    print(f"Reading data from {input_file}...")
    
    # Load members list
    members_lookup = load_members_list(members_file)
    
    # Load Rajya Sabha list if provided
    rajya_sabha_lookup = {}
    if rajya_sabha_file:
        rajya_sabha_lookup = load_rajya_sabha_list(rajya_sabha_file)
        print(f"Loaded {len(rajya_sabha_lookup)} Rajya Sabha members")
    
    # Read the main CSV file
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Standardize column names
    df = standardize_columns(df)
    
    # Required columns for input
    required_columns = ['page', 'metadata', 'speaker', 
                       'eng name(pref 1)', 'hind name(pref 1)', 
                       'eng name(pref 2)', 'hind name(pref 2)', 
                       'speech']
    
    # Check if all required columns are present
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Extract the most common date from metadata
    common_date = extract_date_from_metadata(df['metadata'].tolist())
    print(f"Extracted common date: {common_date}")
    
    # Count frequency of speakers in both preference sets
    pref1_speakers_count = Counter(df['eng name(pref 1)'].fillna('Unknown'))
    
    # Get names that appear more than 5 times in preference 1
    frequent_speakers = {name for name, count in pref1_speakers_count.items() 
                        if count > 5 and name != 'Unknown'}
    
    print(f"Found {len(frequent_speakers)} speakers with more than 5 occurrences: {frequent_speakers}")
    
    # Helper function to detect if text is Hindi
    def is_hindi(text):
        if not isinstance(text, str) or text.strip() == '':
            return False
        
        try:
            lang = langdetect.detect(text)
            return lang == 'hi'
        except:
            hindi_pattern = re.compile(r'[\u0900-\u097F]')
            return bool(hindi_pattern.search(text))
    
    # Identify if original speaker column contains Hindi text
    df['is_hindi_speaker'] = df['speaker'].apply(is_hindi)
    
    # Initialize output data as lists for better performance
    output_data = {
        'sl_no': list(range(1, len(df) + 1)),
        'page': df['page'].tolist(),
        'date': [common_date] * len(df),
        'eng name': [],
        'hindi name': [],
        'party during election': [],
        'party during speech': [],
        'speech': df['speech'].astype(str).str.replace('*', '', regex=False).tolist()
    }
    
    # Track which rows used preference 2
    pref2_used_rows = []
    pref2_reasons = []
    
    # Process each row to determine correct speaker
    for i, row in df.iterrows():
        speaker_pref1 = row['eng name(pref 1)']
        speaker_pref2 = row['eng name(pref 2)']
        
        # Decision logic for determining correct speaker
        selected_speaker = None
        reason = ""
        
        # Logic Path 1: If a name occurs only once in pref 1, but pref 2 has a frequent speaker (>5 times)
        #NEEDS MORE TESTING
        # if (not is_null_or_empty(speaker_pref1) and not is_null_or_empty(speaker_pref2) and 
        #     pref1_speakers_count[speaker_pref1] == 1 and speaker_pref2 in frequent_speakers):
        #     selected_speaker = speaker_pref2
        #     reason = f"Single occurrence in pref1, pref2 is frequent speaker ({speaker_pref2})"
        
        # Logic Path 2: If original speaker is in Hindi and pref 2 is a frequent speaker
        # elif row['is_hindi_speaker'] and not is_null_or_empty(speaker_pref2) and speaker_pref2 in frequent_speakers:
        #     eng_speaker_rows = df[~df['is_hindi_speaker']]
        #     if speaker_pref2 in eng_speaker_rows['eng name(pref 2)'].values:
        #         selected_speaker = speaker_pref2
        #         reason = f"Hindi original speaker, pref2 is frequent ({speaker_pref2})"
        #     else:
        #         selected_speaker = speaker_pref1
        
        # Default to pref 1
        # else: #(indent the following if testing path 1)
        selected_speaker = speaker_pref1 if not is_null_or_empty(speaker_pref1) else speaker_pref2
        
        # If still no valid speaker, use whatever is available
        if is_null_or_empty(selected_speaker):
            if not is_null_or_empty(speaker_pref2):
                selected_speaker = speaker_pref2
                reason = "Fallback to pref2 (pref1 unavailable)"
            else:
                selected_speaker = speaker_pref1 if not is_null_or_empty(speaker_pref1) else "Unknown"
        
        # Track if pref2 was used (only count if pref1 != pref2)
        if (selected_speaker == speaker_pref2 and not is_null_or_empty(speaker_pref2) and 
            speaker_pref1 != speaker_pref2):
            pref2_used_rows.append(i + 1)  # +1 for 1-based row numbering
            pref2_reasons.append(reason if reason else "Default selection")
        
        # Set the eng name
        output_data['eng name'].append(selected_speaker)
        
        # Set the corresponding hindi name based on which preference was chosen
        if selected_speaker == speaker_pref1:
            output_data['hindi name'].append(row['hind name(pref 1)'])
        elif selected_speaker == speaker_pref2:
            output_data['hindi name'].append(row['hind name(pref 2)'])
        else:
            hindi_name = row['hind name(pref 1)'] if not is_null_or_empty(row['hind name(pref 1)']) else row['hind name(pref 2)']
            output_data['hindi name'].append(hindi_name)
    
    # Initialize party columns
    matches_found = 0
    special_characters_count = 0
    nominated_count = 0
    rajya_sabha_matches = 0
    
    for i, speaker_name in enumerate(output_data['eng name']):
        # Check if it's a special character
        if is_special_character(speaker_name):
            output_data['party during election'].append('Invalid')
            output_data['party during speech'].append('Invalid')
            special_characters_count += 1
            continue
        
        # Check if name contains (Nominated)
        if has_special_tag(speaker_name):
            output_data['party during election'].append('Invalid')
            output_data['party during speech'].append('Invalid')
            nominated_count += 1
            continue
        
        # Try to find in Lok Sabha members first
        member_info = find_member_in_lookup(speaker_name, members_lookup)
        
        # If still no match, try Rajya Sabha list
        if not member_info and rajya_sabha_lookup:
            rajya_member_info = find_member_in_lookup(speaker_name, rajya_sabha_lookup)
            
            if rajya_member_info:
                # For Rajya Sabha members, use the same party for both election and speech
                output_data['party during election'].append(rajya_member_info['party'])
                output_data['party during speech'].append(rajya_member_info['party'])
                rajya_sabha_matches += 1
                continue
        
        if member_info:
            matches_found += 1
            # Set party during election (Initial Party)
            output_data['party during election'].append(member_info['initial_party'])
            
            # Determine party during speech
            current_party = parse_party_changes(
                member_info['date'], 
                member_info['new_party'], 
                common_date
            )
            
            if current_party:
                output_data['party during speech'].append(current_party)
            else:
                # If no party change or change date is after speech date, use initial party
                output_data['party during speech'].append(member_info['initial_party'])
        else:
            # If member not found, write nil
            output_data['party during election'].append('nil')
            output_data['party during speech'].append('nil')
    
    # Create output DataFrame from the data
    output_df = pd.DataFrame(output_data)
    
    # Save the processed data
    print(f"Saving processed data to {output_file}...")
    output_df.to_csv(output_file, index=False)
    print("Processing complete!")
    
    # Return enhanced statistics
    return {
        "total_rows": len(df),
        "pref1_used": sum([1 for i, name in enumerate(output_df['eng name']) if name == df.iloc[i]['eng name(pref 1)']]),
        "pref2_used": len(pref2_used_rows),
        "pref2_rows": pref2_used_rows,
        "pref2_reasons": pref2_reasons,
        "unique_speakers": len(set(output_df['eng name'].dropna())),
        "frequent_speakers_count": len(frequent_speakers),
        "frequent_speakers": frequent_speakers,
        "lok_sabha_matches": matches_found,
        "rajya_sabha_matches": rajya_sabha_matches,
        "special_characters": special_characters_count,
        "nominated_members": nominated_count,
        "common_date": common_date
    }

def main():
    """
    Main function with file paths
    """
    # File paths - update these as needed
    input_file = "matched_speeches.csv"          # Input CSV file with speeches
    members_file = "16th_Lok_Sabha_Members.csv"  # CSV file with MP information
    rajya_sabha_file = "rajyasabha_ministers_16.csv" 
    output_file = "processed_speeches.csv"       # Output file
    
    try:
        # Run the processing
        stats = process_csv(input_file, members_file, output_file, rajya_sabha_file)
        
        # Display statistics
        print("\nProcessing Statistics:")
        print(f"Total rows processed: {stats['total_rows']}")
        print(f"Rows using preference 1: {stats['pref1_used']} ({stats['pref1_used']/stats['total_rows']*100:.1f}%)")
        print(f"Rows using preference 2: {stats['pref2_used']} ({stats['pref2_used']/stats['total_rows']*100:.1f}%)")
        
        # Show which rows used preference 2
        if stats['pref2_rows']:
            print(f"\nRows where preference 2 was used:")
            print(f"Row numbers: {', '.join(map(str, stats['pref2_rows']))}")
            
            # Show detailed breakdown with reasons
            print(f"\nDetailed breakdown of preference 2 usage:")
            for i, (row_num, reason) in enumerate(zip(stats['pref2_rows'], stats['pref2_reasons'])):
                print(f"  Row {row_num}: {reason}")
                # Limit output to first 20 rows to avoid overwhelming display
                if i >= 19 and len(stats['pref2_rows']) > 20:
                    remaining = len(stats['pref2_rows']) - 20
                    print(f"  ... and {remaining} more rows")
                    break
        else:
            print("\nNo rows used preference 2.")
        
        print(f"\nUnique speakers identified: {stats['unique_speakers']}")
        print(f"Frequent speakers (>5 occurrences): {stats['frequent_speakers_count']}")
        print(f"Frequent speakers: {', '.join(sorted(stats['frequent_speakers']))}")
        print(f"Lok Sabha members matched: {stats['lok_sabha_matches']}")
        print(f"Rajya Sabha members matched: {stats['rajya_sabha_matches']}")
        print(f"Special characters identified: {stats['special_characters']}")
        print(f"Nominated members identified: {stats['nominated_members']}")
        print(f"Common date extracted: {stats['common_date']}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
