import pandas as pd
import re
from collections import Counter
import langdetect
from datetime import datetime

def load_members_list(members_file):
    """
    Load the members list CSV file and create a lookup dictionary
    
    Args:
        members_file (str): Path to the members CSV file
    
    Returns:
        dict: Dictionary with MP names as keys and member info as values
    """
    members_df = pd.read_csv(members_file)
    
    # Standardize column names
    members_df.columns = [col.strip() for col in members_df.columns]
    
    # Create lookup dictionary using both English and Hindi names
    members_lookup = {}
    
    for _, row in members_df.iterrows():
        mp_name = str(row.get('MP Name', '')).strip()
        mp_name_hindi = str(row.get('MP Name (Hindi)', '')).strip()
        constituency = str(row.get('Constituency', '')).strip()
        initial_party = str(row.get('Initial Party', '')).strip()
        date = str(row.get('Date', '')).strip()
        new_party = str(row.get('New Party', '')).strip()
        state = str(row.get('State', '')).strip()
        
        member_info = {
            'constituency': constituency,
            'initial_party': initial_party,
            'date': date,
            'new_party': new_party,
            'state': state,
            'hindi_name': mp_name_hindi
        }
        
        # Add both English and Hindi names as keys (handle variations)
        if mp_name and mp_name != 'nan' and mp_name != '':
            members_lookup[mp_name] = member_info
            # Also add name variations (remove common prefixes/suffixes)
            clean_name = mp_name.replace('Shri ', '').replace('Smt. ', '').replace('Dr. ', '').strip()
            if clean_name != mp_name:
                members_lookup[clean_name] = member_info
        
        if mp_name_hindi and mp_name_hindi != 'nan' and mp_name_hindi != '':
            members_lookup[mp_name_hindi] = member_info
    
    return members_lookup

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
        if pd.notna(metadata):
            matches = re.findall(date_pattern, str(metadata))
            dates_found.extend(matches)
    
    if dates_found:
        # Return the most common date
        date_counter = Counter(dates_found)
        return date_counter.most_common(1)[0][0]
    
    return "Unknown"

def parse_party_changes(date_str, new_party_str, reference_date):
    """
    Parse party change information and determine current party
    
    Args:
        date_str: String containing comma-separated dates
        new_party_str: String containing comma-separated new parties
        reference_date: Date to compare against (from metadata)
    
    Returns:
        str: Current party at the reference date
    """
    if pd.isna(date_str) or pd.isna(new_party_str) or date_str == '' or new_party_str == '':
        return None
    
    try:
        # Parse reference date
        ref_date = datetime.strptime(reference_date, '%d.%m.%Y')
        
        # Split dates and parties
        dates = [d.strip() for d in str(date_str).split(',')]
        parties = [p.strip() for p in str(new_party_str).split(',')]
        
        # Find the latest party change before or on the reference date
        current_party = None
        latest_change_date = None
        
        for i, date_str in enumerate(dates):
            if i < len(parties):
                try:
                    change_date = datetime.strptime(date_str.strip(), '%d.%m.%Y')
                    if change_date <= ref_date:
                        if latest_change_date is None or change_date > latest_change_date:
                            latest_change_date = change_date
                            current_party = parties[i]
                except:
                    continue
        
        return current_party
    except:
        return None

def process_csv(input_file, members_file, output_file):
    """
    Process the CSV file with member information and enhanced logic
    
    Args:
        input_file (str): Path to the input CSV file
        members_file (str): Path to the members CSV file
        output_file (str): Path to save the processed CSV file
    """
    print(f"Reading data from {input_file}...")
    
    # Load members list
    members_lookup = load_members_list(members_file)
    
    # Read the main CSV file
    df = pd.read_csv(input_file)
    
    # Standardize column names by removing whitespace
    df.columns = [col.strip() for col in df.columns]
    
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
    pref2_speakers_count = Counter(df['eng name(pref 2)'].fillna('Unknown'))
    
    # Get the top 4 most occurring names in preference 1
    top_speakers = [name for name, _ in pref1_speakers_count.most_common(4) if name != 'Unknown']
    
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
    
    # Identify alternating pattern function
    def get_alternating_pattern(sequence):
        filtered_seq = [s for s in sequence if pd.notna(s) and s != '']
        if len(filtered_seq) < 4:
            return None
        
        patterns = []
        for i in range(0, len(filtered_seq) - 3):
            if (filtered_seq[i] == filtered_seq[i+2] and 
                filtered_seq[i+1] == filtered_seq[i+3] and
                filtered_seq[i] != filtered_seq[i+1]):
                patterns.append((filtered_seq[i], filtered_seq[i+1]))
        
        if patterns:
            pattern_counts = Counter(patterns)
            return pattern_counts.most_common(1)[0][0]
        
        return None
    
    # Detect patterns
    pref1_speaker_sequence = df['eng name(pref 1)'].tolist()
    pref1_pattern = get_alternating_pattern(pref1_speaker_sequence)
    
    # Identify if original speaker column contains Hindi text
    df['is_hindi_speaker'] = df['speaker'].apply(is_hindi)
    
    # Create output DataFrame
    output_df = pd.DataFrame()
    
    # Add serial number column
    output_df['sl_no'] = range(1, len(df) + 1)
    
    # Copy page column
    output_df['page'] = df['page']
    
    # Add date column (extracted from metadata)
    output_df['date'] = common_date
    
    # Initialize output columns
    output_df['eng name'] = ''
    output_df['hindi name'] = ''
    
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
        
        # Logic Path 1: If a name occurs only once in pref 1, but pref 2 has one of the top 4 most frequent names
        if (pd.notna(speaker_pref1) and pd.notna(speaker_pref2) and 
            pref1_speakers_count[speaker_pref1] == 1 and speaker_pref2 in top_speakers):
            selected_speaker = speaker_pref2
            reason = f"Single occurrence in pref1, pref2 is top speaker ({speaker_pref2})"
        
        # Logic Path 2: If original speaker is in Hindi and pref 2 is a prominent speaker
        elif row['is_hindi_speaker'] and pd.notna(speaker_pref2) and speaker_pref2 in top_speakers:
            eng_speaker_rows = df[~df['is_hindi_speaker']]
            if speaker_pref2 in eng_speaker_rows['eng name(pref 2)'].values:
                selected_speaker = speaker_pref2
                reason = f"Hindi original speaker, pref2 is prominent ({speaker_pref2})"
            else:
                selected_speaker = speaker_pref1
        
        # Logic Path 3: ABAB pattern recognition
        elif pref1_pattern and pd.notna(speaker_pref1) and pd.notna(speaker_pref2):
            expected_speaker = None
            if i > 0 and pd.notna(output_df.at[i-1, 'eng name']):
                last_speaker = output_df.at[i-1, 'eng name']
                if last_speaker == pref1_pattern[0]:
                    expected_speaker = pref1_pattern[1]
                elif last_speaker == pref1_pattern[1]:
                    expected_speaker = pref1_pattern[0]
            
            if expected_speaker and speaker_pref1 != expected_speaker and speaker_pref2 == expected_speaker:
                selected_speaker = speaker_pref2
                reason = f"ABAB pattern match ({speaker_pref2})"
            else:
                selected_speaker = speaker_pref1
        
        # Default to pref 1
        else:
            selected_speaker = speaker_pref1 if pd.notna(speaker_pref1) else speaker_pref2
        
        # If still no valid speaker, use whatever is available
        if not selected_speaker or pd.isna(selected_speaker):
            if pd.notna(speaker_pref2):
                selected_speaker = speaker_pref2
                reason = "Fallback to pref2 (pref1 unavailable)"
            else:
                selected_speaker = speaker_pref1 if pd.notna(speaker_pref1) else "Unknown"
        
        # Track if pref2 was used (only count if pref1 != pref2)
        if (selected_speaker == speaker_pref2 and pd.notna(speaker_pref2) and 
            speaker_pref1 != speaker_pref2):
            pref2_used_rows.append(i + 1)  # +1 for 1-based row numbering
            pref2_reasons.append(reason if reason else "Default selection")
        
        # Set the eng name
        output_df.at[i, 'eng name'] = selected_speaker
        
        # Set the corresponding hindi name based on which preference was chosen
        if selected_speaker == speaker_pref1:
            output_df.at[i, 'hindi name'] = row['hind name(pref 1)']
        elif selected_speaker == speaker_pref2:
            output_df.at[i, 'hindi name'] = row['hind name(pref 2)']
        else:
            output_df.at[i, 'hindi name'] = row['hind name(pref 1)'] if pd.notna(row['hind name(pref 1)']) else row['hind name(pref 2)']
    
    
    # Add party information
    output_df['party during election'] = ''
    output_df['party during speech'] = ''
    
    matches_found = 0
    
    for i, row in output_df.iterrows():
        speaker_name = row['eng name']
        
        # Try exact match first
        member_info = members_lookup.get(speaker_name)
        
        # If no exact match, try fuzzy matching with common prefixes removed
        if not member_info:
            clean_speaker = speaker_name.replace('Shri ', '').replace('Smt. ', '').replace('Dr. ', '').strip()
            member_info = members_lookup.get(clean_speaker)
        
        if member_info:
            matches_found += 1
            # Set party during election (Initial Party)
            output_df.at[i, 'party during election'] = member_info['initial_party']
            
            # Determine party during speech
            current_party = parse_party_changes(
                member_info['date'], 
                member_info['new_party'], 
                common_date
            )
            
            if current_party:
                output_df.at[i, 'party during speech'] = current_party
            else:
                # If no party change or change date is after speech date, use initial party
                output_df.at[i, 'party during speech'] = member_info['initial_party']
        else:
            # If member not found, write nil
            output_df.at[i, 'party during election'] = 'nil'
            output_df.at[i, 'party during speech'] = 'nil'
            
    # Clean speech text - remove all asterisks
    cleaned_speech = df['speech'].astype(str).str.replace('*', '', regex=False)
    output_df['speech'] = cleaned_speech
    
    # Save the processed data
    print(f"Saving processed data to {output_file}...")
    output_df.to_csv(output_file, index=False)
    print("Processing complete!")
    
    # Return enhanced statistics
    return {
        "total_rows": len(df),
        "pref1_used": sum(output_df['eng name'] == df['eng name(pref 1)']),
        "pref2_used": len(pref2_used_rows),
        "pref2_rows": pref2_used_rows,
        "pref2_reasons": pref2_reasons,
        "unique_speakers": len(set(output_df['eng name'].dropna())),
        "members_matched": sum(output_df['party during election'] != 'nil'),
        "common_date": common_date
    }

def main():
    """
    Main function with file paths
    """
    # File paths - update these as needed
    input_file = "matched_speeches.csv"          # Input CSV file with speeches
    members_file = "17th_Lok_Sabha_Members.csv"  # CSV file with MP information
    output_file = "processed_speeches.csv"       # Output file
    
    try:
        # Run the processing
        stats = process_csv(input_file, members_file, output_file)
        
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
        print(f"Members matched with party info: {stats['members_matched']}")
        print(f"Common date extracted: {stats['common_date']}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()