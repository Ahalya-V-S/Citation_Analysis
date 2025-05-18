import pandas as pd
import numpy as np
import os
import io
import zipfile
import tempfile
import streamlit as st


def load_citation_data(file):
    """
    Load and process the citation data CSV
    
    Parameters:
    -----------
    file : UploadedFile
        The uploaded citation data file
    
    Returns:
    --------
    DataFrame
        Processed citation data
    """
    try:
        df = pd.read_csv(file, encoding='latin1')
        # ✅ Normalize column names
        df.columns = [col.strip().title() for col in df.columns]
        # Optional: Streamlit preview
        st.write("📄 Raw Citation Data Preview:")
        st.write(df)
        # Define expected columns
        essential_cols = ['Article Id']  # Only Article Id is truly required
        recommended_cols = ['Title', 'Author', 'Cited By']
        year_cols = [str(year) for year in range(1992, 2024)]

        # Check for missing essential columns
        missing_essential = [
            col for col in essential_cols if col not in df.columns
        ]
        if missing_essential:
            raise ValueError(
                f"Missing required columns: {missing_essential}. The dataset must contain an 'Article Id' column."
            )

        # Check for recommended columns and warn if missing
        missing_recommended = [
            col for col in recommended_cols if col not in df.columns
        ]
        if missing_recommended:
            st.warning(
                f"Some recommended columns are missing: {missing_recommended}. Some functionality may be limited."
            )

        # Extract year columns that are available
        available_years = [col for col in year_cols if col in df.columns]

        # Create empty columns for missing recommended columns
        for col in recommended_cols:
            if col not in df.columns:
                df[col] = np.nan if col != 'Cited By' else 0

        # Determine columns to keep
        cols_to_keep = essential_cols + recommended_cols + available_years
        cols_to_keep = [col for col in cols_to_keep if col in df.columns]
        df = df[cols_to_keep]

        # Calculate total citations if 'Cited By' is missing or needs verification
        if 'Cited By' in df.columns and not df['Cited By'].isnull().all():
            # Verify 'Cited By' as sum of yearly citations (if year columns exist)
            if available_years:
                year_sum = df[available_years].sum(axis=1)
                # If there's a significant difference, use the calculated sum
                significant_diff = abs(df['Cited By'] - year_sum) > 1
                if significant_diff.any():
                    df.loc[significant_diff,
                           'Cited By'] = year_sum[significant_diff]
        else:
            # Calculate 'Cited By' as sum of yearly citations if years exist
            if available_years:
                df['Cited By'] = df[available_years].sum(axis=1)
            else:
                # No year columns, set 'Cited By' to 0
                df['Cited By'] = 0
                st.warning(
                    "No citation count or year columns found. Using zero for all citation counts."
                )

        return df

    except Exception as e:
        raise Exception(f"Error loading citation data: {str(e)}")


def load_topic_model_data(file):
    """
    Load and process the topic model data CSV
    
    Parameters:
    -----------
    file : UploadedFile
        The uploaded topic model data file
    
    Returns:
    --------
    DataFrame
        Processed topic model data
    """
    try:
        df = pd.read_csv(file, encoding='latin1')

        # Optional: Preview the data
        st.write("📄 Raw Topic Model Data Preview:")
        st.write(df)

        # Check for potential article ID columns with flexible naming
        article_id_cols = [
            'ArticleID', 'Article_ID', 'Article Id', 'article_id', 'id', 'ID',
            'paper_id', 'PaperID'
        ]
        found_id_col = None

        for col in article_id_cols:
            if col in df.columns:
                found_id_col = col
                break

        # If no recognized ID column, look for any column with "id" in the name
        if not found_id_col:
            id_candidates = [col for col in df.columns if 'id' in col.lower()]
            if id_candidates:
                found_id_col = id_candidates[0]
                st.warning(f"Using '{found_id_col}' as the article ID column.")

        # If we still can't find an ID column, raise an error
        if not found_id_col:
            raise ValueError(
                "No article ID column found. Please ensure your data has an article identifier column."
            )

        # If the ID column isn't named 'ArticleID', rename it
        if found_id_col != 'ArticleID':
            df = df.rename(columns={found_id_col: 'ArticleID'})
            st.info(
                f"Renamed '{found_id_col}' to 'ArticleID' for consistency.")

        # Identify topic model columns
        topic_model_prefixes = ['LDA', 'HDP', 'CTM', 'DLDA', 'DHDP', 'DCTM']
        topic_cols = [
            col for col in df.columns
            if any(prefix in col for prefix in topic_model_prefixes)
        ]

        # Check if we found topic columns
        if not topic_cols:
            st.error(
                "No topic model columns found. Looking for columns containing LDA, HDP, CTM, etc."
            )
            # Try a more lenient search
            topic_cols = [
                col for col in df.columns if any(
                    x in col.upper() for x in ['TOPIC', 'MODEL', 'LDA', 'HDP'])
            ]

            if topic_cols:
                st.warning(f"Using alternative topic columns: {topic_cols}")
            else:
                raise ValueError(
                    "No topic model columns found in the data. Please check your file format."
                )

        # Check for citation count column with flexible naming
        citation_cols = [
            'CITATIONCOUNT', 'CitationCount', 'Citation_Count',
            'citation_count', 'Citations', 'citations', 'Cited_By', 'cited_by',
            'Cited By'
        ]
        found_citation_col = None

        for col in citation_cols:
            if col in df.columns:
                found_citation_col = col
                break

        # Keep necessary columns
        cols_to_keep = ['ArticleID'] + topic_cols

        # Handle citation count column if found
        if found_citation_col:
            if found_citation_col != 'CITATIONCOUNT':
                df = df.rename(columns={found_citation_col: 'CITATIONCOUNT'})
                st.info(
                    f"Renamed '{found_citation_col}' to 'CITATIONCOUNT' for consistency."
                )
            cols_to_keep.append('CITATIONCOUNT')

        # Ensure we only keep columns that exist in the dataframe
        cols_to_keep = [col for col in cols_to_keep if col in df.columns]
        df = df[cols_to_keep]

        return df

    except Exception as e:
        raise Exception(f"Error loading topic model data: {str(e)}")


def load_paper_text(file, article_id=None):
    """
    Extract and load paper text from a ZIP file where text files are directly inside a '1992' folder.

    Parameters:
    -----------
    file : UploadedFile
        The uploaded ZIP file containing paper texts
    article_id : str, optional
        Specific article ID to extract, if None extract all

    Returns:
    --------
    dict
        Dictionary mapping article IDs to their text content
    """
    try:
        # Create a temporary directory to extract files
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # Dictionary to store paper texts
            paper_texts = {}

            # Expected folder path: temp_dir/1992
            folder_path = os.path.join(temp_dir, '1992')

            # Check if the '1992' folder exists and is a directory
            if not os.path.isdir(folder_path):
                raise Exception("Expected '1992' folder not found in ZIP file")

            # If a specific article_id is provided, look for that file
            if article_id:
                # Construct the expected file name (article_id + .txt)
                file_name = f"{article_id}.txt"
                file_path = os.path.join(folder_path, file_name)

                if os.path.isfile(file_path):
                    with open(file_path, 'r', errors='ignore') as f:
                        paper_texts[article_id] = f.read()
                else:
                    raise Exception(
                        f"Text file for article ID '{article_id}' not found")
            else:
                # Load all text files in the '1992' folder
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)

                    # Process only text files
                    if os.path.isfile(file_path) and (
                            file_name.endswith('.txt')
                            or not file_name.endswith(
                                ('.pdf', '.doc', '.docx'))):
                        # Derive article ID from the file name (without extension)
                        article_id = os.path.splitext(file_name)[0]

                        with open(file_path, 'r', errors='ignore') as f:
                            paper_texts[article_id] = f.read()

            return paper_texts

    except Exception as e:
        raise Exception(f"Error loading paper texts: {str(e)}")


def load_paper_metadata(text_content):
    """
    Extract metadata from paper text content
    
    Parameters:
    -----------
    text_content : str
        The text content of the paper
    
    Returns:
    --------
    dict
        Dictionary containing extracted metadata
    """
    metadata = {
        'paper': None,
        'from': None,
        'title': None,
        'authors': None,
        'date': None,
        'comments': None,
        'journal': None,
        'abstract': None
    }

    lines = text_content.split('\n')

    # Find the slash markers indicating section boundaries
    slash_markers = []
    for i, line in enumerate(lines):
        # Check for single backslash format (most common in the papers)
        if line.strip() == '\\':
            slash_markers.append(i)
        # Also check for double backslash format
        elif line.strip() == '\\\\':
            slash_markers.append(i)

    print(
        f"Found {len(slash_markers)} slash markers at positions: {slash_markers}"
    )

    # Extract metadata from the section before "Abstract" marker
    # In the format shown, metadata is between the 1st and 2nd slash markers
    metadata_section = []
    if len(slash_markers) >= 2:
        # Start after the first marker and end before the second marker
        start_index = slash_markers[0] + 1
        end_index = slash_markers[1]

        for i in range(start_index, end_index):
            metadata_section.append(lines[i].strip())

    # Process the metadata section line by line
    for line in metadata_section:
        if line.startswith('Paper:'):
            metadata['paper'] = line[6:].strip()
        elif line.startswith('From:'):
            metadata['from'] = line[5:].strip()
        elif line.startswith('Date:'):
            metadata['date'] = line[5:].strip()
        elif line.startswith('Title:'):
            metadata['title'] = line[6:].strip()
        elif line.startswith('Authors:') or line.startswith('Author:'):
            metadata['authors'] = line[line.find(':') + 1:].strip()
        elif line.startswith('Comments:'):
            metadata['comments'] = line[9:].strip()
        elif line.startswith('Journal-ref:'):
            metadata['journal'] = line[12:].strip()

    # Extract abstract section - should be between the 2nd and 3rd slash markers
    # In the format shown, the abstract is explicitly labeled with "Abstract" and is
    # between the 2nd and 3rd slash markers
    abstract_lines = []
    if len(slash_markers) >= 3:
        # Check if there's an explicit "Abstract" label
        has_abstract_label = False
        abstract_section_start = slash_markers[1] + 1

        # Look for "Abstract" label right after 2nd slash marker
        for i in range(abstract_section_start,
                       min(abstract_section_start + 3, len(lines))):
            if lines[i].strip().lower() == "abstract":
                has_abstract_label = True
                abstract_section_start = i + 1  # Start after the "Abstract" label
                break

        # Extract abstract content
        for i in range(abstract_section_start, slash_markers[2]):
            line = lines[i].strip()
            if line:  # Skip empty lines
                abstract_lines.append(line)

    # If no abstract label was found, try to find the abstract after the metadata
    # This is a fallback for papers that don't use the explicit "Abstract" label
    if not abstract_lines and len(slash_markers) >= 3:
        for i in range(slash_markers[1] + 1, slash_markers[2]):
            line = lines[i].strip()
            if line and not any(
                    line.startswith(prefix) for prefix in [
                        'Paper:', 'From:', 'Date:', 'Title:', 'Authors:',
                        'Author:', 'Comments:', 'Journal-ref:', 'Abstract:'
                    ]):
                abstract_lines.append(line)

    # Combine abstract lines into a single text
    if abstract_lines:
        metadata['abstract'] = ' '.join(abstract_lines).strip()

    # Print diagnostic information for debugging
    print(f"Paper: {metadata['paper']}")
    print(f"From: {metadata['from']}")
    print(f"Date: {metadata['date']}")
    print(f"Title: {metadata['title']}")
    print(f"Authors: {metadata['authors']}")
    print(f"Comments: {metadata['comments']}")
    print(f"Journal: {metadata['journal']}")
    if metadata['abstract']:
        print(f"Abstract (first 50 chars): {metadata['abstract'][:50]}...")
    else:
        print("No abstract found")

    return metadata
