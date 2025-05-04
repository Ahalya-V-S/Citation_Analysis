import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

def calculate_citation_metrics(citation_df):
    """
    Calculate various citation metrics for the dataset
    
    Parameters:
    -----------
    citation_df : DataFrame
        DataFrame containing citation data
    
    Returns:
    --------
    dict
        Dictionary of citation metrics
    """
    total_papers = len(citation_df)
    
    # Extract citation counts
    citation_counts = citation_df['Cited By'].values
    
    # Calculate basic statistics
    total_citations = citation_counts.sum()
    mean_citations = citation_counts.mean()
    median_citations = np.median(citation_counts)
    max_citations = citation_counts.max()
    min_citations = citation_counts.min()
    
    # Calculate citation percentiles
    percentiles = np.percentile(citation_counts, [10, 25, 50, 75, 90, 95, 99])
    
    # Count papers with zero citations
    zero_citation_count = np.sum(citation_counts == 0)
    zero_citation_percent = (zero_citation_count / total_papers) * 100
    
    # Calculate h-index for the collection
    h_index = calculate_h_index(citation_counts)
    
    # Calculate Gini coefficient (measure of citation inequality)
    gini = calculate_gini(citation_counts)
    
    # Return metrics as a dictionary
    metrics = {
        'total_papers': total_papers,
        'total_citations': total_citations,
        'mean_citations': mean_citations,
        'median_citations': median_citations,
        'max_citations': max_citations,
        'min_citations': min_citations,
        'percentile_10': percentiles[0],
        'percentile_25': percentiles[1],
        'percentile_50': percentiles[2],
        'percentile_75': percentiles[3],
        'percentile_90': percentiles[4],
        'percentile_95': percentiles[5],
        'percentile_99': percentiles[6],
        'zero_citation_count': zero_citation_count,
        'zero_citation_percent': zero_citation_percent,
        'h_index': h_index,
        'gini_coefficient': gini
    }
    
    return metrics

def calculate_h_index(citation_counts):
    """
    Calculate the h-index for a set of papers
    
    Parameters:
    -----------
    citation_counts : array-like
        Array of citation counts
    
    Returns:
    --------
    int
        h-index value
    """
    # Sort citation counts in descending order
    sorted_counts = np.sort(citation_counts)[::-1]
    
    # Find the h-index
    h = 0
    for i, count in enumerate(sorted_counts):
        if count >= i + 1:
            h = i + 1
        else:
            break
    
    return h

def calculate_gini(citation_counts):
    """
    Calculate the Gini coefficient for citation distribution
    
    Parameters:
    -----------
    citation_counts : array-like
        Array of citation counts
    
    Returns:
    --------
    float
        Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    """
    # Sort citation counts
    sorted_counts = np.sort(citation_counts)
    n = len(sorted_counts)
    
    # Calculate Gini coefficient
    index = np.arange(1, n + 1)
    gini = ((2 * index - n - 1) * sorted_counts).sum() / (n * sorted_counts.sum())
    
    return gini

def analyze_citation_trends(citation_df):
    """
    Analyze citation trends over time
    
    Parameters:
    -----------
    citation_df : DataFrame
        DataFrame containing citation data with yearly columns
    
    Returns:
    --------
    dict
        Dictionary containing citation trend analysis
    """
    # Extract year columns
    year_cols = [col for col in citation_df.columns if col.isdigit() and 1992 <= int(col) <= 2023]
    
    # Calculate yearly citation totals
    yearly_totals = citation_df[year_cols].sum()
    
    # Calculate yearly average citations per paper
    yearly_averages = citation_df[year_cols].mean()
    
    # Calculate the cumulative citations over time for each paper
    cumulative_df = citation_df[year_cols].cumsum(axis=1)
    
    # Calculate median time to first citation
    time_to_first = []
    for _, row in citation_df.iterrows():
        year_data = row[year_cols]
        years_with_citations = year_data[year_data > 0]
        if not years_with_citations.empty:
            first_citation_year = years_with_citations.index[0]
            publication_year = row.get('Publication Year', year_cols[0])
            if isinstance(publication_year, str):
                publication_year = year_cols[0]  # Default to first year if not available
            time_to_first.append(int(first_citation_year) - int(publication_year))
    
    # Median time to first citation
    median_time_to_first = np.median(time_to_first) if time_to_first else None
    
    # Calculate citation half-life
    half_lives = []
    for _, row in citation_df.iterrows():
        total_citations = row['Cited By']
        if total_citations > 0:
            cumulative = 0
            for year in year_cols:
                cumulative += row[year]
                if cumulative >= total_citations / 2:
                    half_lives.append(int(year))
                    break
    
    # Median citation half-life
    median_half_life = np.median(half_lives) if half_lives else None
    
    # Calculate peak citation years
    peak_years = []
    for _, row in citation_df.iterrows():
        year_data = row[year_cols]
        if year_data.max() > 0:
            peak_year = year_cols[year_data.argmax()]
            peak_years.append(int(peak_year))
    
    # Most common peak citation year
    most_common_peak = np.bincount(peak_years).argmax() if peak_years else None
    
    # Return trends analysis
    trends = {
        'yearly_totals': yearly_totals,
        'yearly_averages': yearly_averages,
        'median_time_to_first_citation': median_time_to_first,
        'median_citation_half_life': median_half_life,
        'most_common_peak_year': most_common_peak
    }
    
    return trends

def identify_citation_patterns(citation_df):
    """
    Identify different citation patterns in papers
    
    Parameters:
    -----------
    citation_df : DataFrame
        DataFrame containing citation data with yearly columns
    
    Returns:
    --------
    dict
        Dictionary mapping pattern types to paper IDs
    """
    # Extract year columns
    year_cols = [col for col in citation_df.columns if col.isdigit() and 1992 <= int(col) <= 2023]
    
    # Initialize pattern categories
    patterns = {
        'flash_in_pan': [],      # High early citations, quick decline
        'sleeping_beauty': [],   # Low early citations, late rise
        'steady_riser': [],      # Gradual increase over time
        'constant_performer': [] # Relatively steady citations 
    }
    
    # Analyze each paper's citation pattern
    for idx, row in citation_df.iterrows():
        paper_id = row.get('Article Id', idx)
        year_data = row[year_cols].values
        
        # Skip papers with very few citations
        if row['Cited By'] < 5:
            continue
            
        # Calculate citation slope for first third, middle third, and last third
        n_years = len(year_cols)
        first_third = year_data[:n_years//3]
        middle_third = year_data[n_years//3:2*n_years//3]
        last_third = year_data[2*n_years//3:]
        
        # Calculate average citations for each third
        first_avg = np.mean(first_third)
        middle_avg = np.mean(middle_third)
        last_avg = np.mean(last_third)
        
        # Determine pattern based on citation distribution
        if first_avg > middle_avg * 2 and first_avg > last_avg * 2:
            patterns['flash_in_pan'].append(paper_id)
        elif last_avg > first_avg * 2 and middle_avg < last_avg:
            patterns['sleeping_beauty'].append(paper_id)
        elif last_avg > first_avg * 1.5 and middle_avg > first_avg:
            patterns['steady_riser'].append(paper_id)
        elif abs(first_avg - last_avg) < first_avg * 0.5:
            patterns['constant_performer'].append(paper_id)
    
    return patterns

def compare_high_low_cited_papers(citation_df, percentile_threshold=25):
    """
    Compare characteristics of highly cited vs. lowly cited papers
    
    Parameters:
    -----------
    citation_df : DataFrame
        DataFrame containing citation data
    percentile_threshold : int
        Percentile threshold for defining high/low cited papers
    
    Returns:
    --------
    tuple
        (high_cited_df, low_cited_df, comparison_stats)
    """
    # Define high and low citation thresholds
    high_threshold = np.percentile(citation_df['Cited By'], 100 - percentile_threshold)
    low_threshold = np.percentile(citation_df['Cited By'], percentile_threshold)
    
    # Create dataframes for high and low cited papers
    high_cited_df = citation_df[citation_df['Cited By'] >= high_threshold].copy()
    low_cited_df = citation_df[citation_df['Cited By'] <= low_threshold].copy()
    
    # Extract year columns
    year_cols = [col for col in citation_df.columns if col.isdigit() and 1992 <= int(col) <= 2023]
    
    # Compare statistical properties
    comparison_stats = {}
    
    # Compare time to first citation
    high_time_to_first = []
    low_time_to_first = []
    
    for df, time_list in [(high_cited_df, high_time_to_first), (low_cited_df, low_time_to_first)]:
        for _, row in df.iterrows():
            year_data = row[year_cols]
            years_with_citations = year_data[year_data > 0]
            if not years_with_citations.empty:
                first_citation_year = years_with_citations.index[0]
                publication_year = row.get('Publication Year', year_cols[0])
                if isinstance(publication_year, str):
                    publication_year = year_cols[0]  # Default to first year if not available
                time_list.append(int(first_citation_year) - int(publication_year))
    
    # Add comparison statistics
    comparison_stats['high_median_time_to_first'] = np.median(high_time_to_first) if high_time_to_first else None
    comparison_stats['low_median_time_to_first'] = np.median(low_time_to_first) if low_time_to_first else None
    
    # Compare citation patterns
    high_patterns = identify_citation_patterns(high_cited_df)
    low_patterns = identify_citation_patterns(low_cited_df)
    
    comparison_stats['high_patterns'] = high_patterns
    comparison_stats['low_patterns'] = low_patterns
    
    return high_cited_df, low_cited_df, comparison_stats
