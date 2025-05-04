import pandas as pd
import numpy as np
import os
import io
import zipfile
import tempfile

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
        df = pd.read_csv(file)
        
        # Ensure the expected columns are present
        expected_cols = ['Article Id', 'Title', 'Author', 'Cited By']
        year_cols = [str(year) for year in range(1992, 2024)]
        
        # Check for missing essential columns
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing essential columns: {missing_cols}")
        
        # Extract year columns that are available 
        available_years = [col for col in year_cols if col in df.columns]
        
        # Keep only the necessary columns
        cols_to_keep = expected_cols + available_years
        df = df[cols_to_keep]
        
        # Calculate total citations if 'Cited By' is missing or needs verification
        if 'Cited By' in df.columns:
            # Verify 'Cited By' as sum of yearly citations
            year_sum = df[available_years].sum(axis=1)
            # If there's a significant difference, use the calculated sum
            significant_diff = abs(df['Cited By'] - year_sum) > 1
            if significant_diff.any():
                df.loc[significant_diff, 'Cited By'] = year_sum[significant_diff]
        else:
            # Calculate 'Cited By' as sum of yearly citations
            df['Cited By'] = df[available_years].sum(axis=1)
        
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
        df = pd.read_csv(file)
        
        # Check for expected column
        if 'ArticleID' not in df.columns:
            raise ValueError("Missing essential column: ArticleID")
        
        # Identify topic model columns
        topic_model_prefixes = ['LDA', 'HDP', 'CTM', 'DLDA', 'DHDP', 'DCTM']
        topic_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in topic_model_prefixes)]
        
        if not topic_cols:
            raise ValueError("No topic model columns found in the data")
            
        # Keep only necessary columns
        cols_to_keep = ['ArticleID'] + topic_cols
        if 'CITATIONCOUNT' in df.columns:
            cols_to_keep.append('CITATIONCOUNT')
            
        df = df[cols_to_keep]
        
        return df
    
    except Exception as e:
        raise Exception(f"Error loading topic model data: {str(e)}")

def load_paper_text(file, article_id=None):
    """
    Extract and load paper text from a ZIP file
    
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
            
            # If article_id is specified, only load that paper
            if article_id:
                paper_path = os.path.join(temp_dir, article_id)
                if os.path.isdir(paper_path):
                    # Find text files in the directory
                    for file_name in os.listdir(paper_path):
                        if file_name.endswith('.txt') or not file_name.endswith(('.pdf', '.doc', '.docx')):
                            with open(os.path.join(paper_path, file_name), 'r', errors='ignore') as f:
                                paper_texts[article_id] = f.read()
                                break
            else:
                # Load all papers
                for folder_name in os.listdir(temp_dir):
                    folder_path = os.path.join(temp_dir, folder_name)
                    if os.path.isdir(folder_path):
                        # Assume folder name is the article ID
                        article_id = folder_name
                        
                        # Find text files in the directory
                        for file_name in os.listdir(folder_path):
                            if file_name.endswith('.txt') or not file_name.endswith(('.pdf', '.doc', '.docx')):
                                with open(os.path.join(folder_path, file_name), 'r', errors='ignore') as f:
                                    paper_texts[article_id] = f.read()
                                    break
            
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
        'title': None,
        'authors': None,
        'date': None,
        'journal': None,
        'abstract': None
    }
    
    lines = text_content.split('\n')
    
    # Extract metadata from the text
    for i, line in enumerate(lines):
        line = line.strip()
        
        if line.startswith('Title:'):
            metadata['title'] = line[6:].strip()
        
        elif line.startswith('Authors:') or line.startswith('Author:'):
            metadata['authors'] = line[line.find(':')+1:].strip()
        
        elif line.startswith('Date:'):
            metadata['date'] = line[5:].strip()
        
        elif line.startswith('Journal-ref:'):
            metadata['journal'] = line[12:].strip()
        
        # Extract abstract (typically after \\ marker and before another \\)
        elif line.strip() == '\\\\':
            abstract_lines = []
            j = i + 1
            while j < len(lines) and not lines[j].strip() == '\\\\':
                abstract_lines.append(lines[j].strip())
                j += 1
            if abstract_lines:
                metadata['abstract'] = ' '.join(abstract_lines).strip()
    
    return metadata
