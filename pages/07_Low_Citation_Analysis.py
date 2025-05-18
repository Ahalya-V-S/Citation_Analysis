import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
from utils.data_loader import load_paper_text, load_paper_metadata
from utils.text_analysis import preprocess_text, extract_keyphrases

# Set page configuration as the first Streamlit command
st.set_page_config(page_title="Low Citation Analysis",
                   page_icon="📉",
                   layout="wide")

st.title("📉 Low Citation Analysis")
st.write("""
This tool analyzes factors contributing to low citation rates using citation counts and abstract content from research papers, identifying patterns in low-cited papers.
""")

# Check if required data is available
if 'citation_data' not in st.session_state:
    st.error(
        "No citation data available. Please upload data in the main page.")
    st.stop()
if 'paper_text_file' not in st.session_state:
    st.error(
        "No paper text file uploaded. Please upload the paper texts in the main page."
    )
    st.stop()

# Get citation data
citation_df = st.session_state['citation_data']

# Load paper texts
with st.spinner("Loading paper texts..."):
    try:
        paper_texts = load_paper_text(st.session_state['paper_text_file'])
        st.success(f"Loaded {len(paper_texts)} paper texts.")
    except Exception as e:
        st.error(f"Error loading paper texts: {str(e)}")
        st.stop()


# Normalize Article Id in citation_df
def normalize_article_id(article_id):
    if pd.isna(article_id):
        return ''
    # Convert to string and remove .0 suffix if present
    article_id_str = str(article_id)
    if article_id_str.endswith('.0'):
        article_id_str = article_id_str[:-2]
    return article_id_str


citation_df['Article Id'] = citation_df['Article Id'].apply(
    normalize_article_id)

# Create abstract DataFrame from paper texts
abstract_data = []
empty_abstract_count = 0
empty_title_count = 0
missing_title_papers = []
for paper_id, text in paper_texts.items():
    metadata = load_paper_metadata(text)
    abstract = metadata.get('abstract', '')
    title = metadata.get('title', '')
    # Fallback: Use file name or first line of text if title is missing
    if not title or title.strip() == '':
        # Attempt to derive title from file name or text
        try:
            # Assume paper_id is derived from file name (e.g., 9202048.txt)
            title = f"Paper {paper_id}"
            # Optionally, extract first line of text as title
            first_line = text.split('\n')[0].strip()
            if first_line and len(first_line) > 10:
                title = first_line[:
                                   100]  # Limit length to avoid overly long titles
        except:
            title = "No title available"
        empty_title_count += 1
        missing_title_papers.append({
            'Article Id': str(paper_id),
            'Reason': 'Missing title in metadata'
        })
    if not abstract or abstract.strip() == '':
        empty_abstract_count += 1
        abstract = "No abstract available"
    abstract_data.append({
        'Article Id':
        str(paper_id),
        'Abstract':
        abstract,
        'Title':
        title,
        'Authors':
        metadata.get('authors', 'No authors available')
    })

abstract_df = pd.DataFrame(abstract_data)

# Warn about empty abstracts and titles
if empty_abstract_count > 0:
    st.warning(
        f"{empty_abstract_count} papers have empty or missing abstracts. Using placeholder text for analysis."
    )
if empty_title_count > 0:
    st.warning(
        f"{empty_title_count} papers have empty or missing titles. Using derived or placeholder text for analysis."
    )
    st.write("Papers with missing titles:")
    st.dataframe(pd.DataFrame(missing_title_papers))

# Debug: Show sample of loaded abstracts, titles, and Article Ids
st.write("Sample of loaded abstracts and titles (first 5):")
st.dataframe(abstract_df[['Article Id', 'Title', 'Abstract']].head())
st.write("Sample of citation Article Ids (first 5):")
st.dataframe(citation_df[['Article Id']].head())

# Merge citation and abstract data
try:
    data_df = pd.merge(
        citation_df,
        abstract_df[['Article Id', 'Abstract', 'Authors']],
        on='Article Id',
        how='left')
except ValueError as e:
    st.error(
        f"Error merging datasets: {str(e)}. Please ensure 'Article Id' columns are consistent."
    )
    st.stop()

# Validate data
if 'Cited By' not in data_df.columns:
    st.error("Required column 'Cited By' not found in dataset.")
    st.stop()

# Debug: Check columns in data_df after merge
st.write("Columns in merged data_df:", data_df.columns.tolist())

# Warn if abstracts or titles are missing after merge
missing_abstracts = data_df['Abstract'].isna().sum(
) if 'Abstract' in data_df.columns else len(data_df)
missing_titles = data_df['Title'].isna().sum(
) if 'Title' in data_df.columns else len(data_df)
if missing_abstracts > 0:
    st.warning(
        f"{missing_abstracts} papers lack abstracts after merging. Using placeholder text."
    )
    data_df['Abstract'] = data_df['Abstract'].fillna("No abstract available")
if missing_titles > 0:
    st.warning(
        f"{missing_titles} papers lack titles after merging. Using placeholder text."
    )
    if 'Title' not in data_df.columns:
        data_df['Title'] = "No title available"
    else:
        data_df['Title'] = data_df['Title'].fillna("No title available")

# Debug: Show merged data sample
st.write("Sample of merged data (first 5 rows):")
display_columns = ['Article Id', 'Cited By', 'Abstract']
if 'Title' in data_df.columns:
    display_columns.insert(1, 'Title')
st.dataframe(data_df[display_columns].head())

# Filter papers with valid abstracts for text analysis
valid_abstract_df = data_df[data_df['Abstract'] !=
                            "No abstract available"].copy()
if len(valid_abstract_df) == 0:
    st.error(
        "No papers with valid abstracts found. Text analysis cannot proceed.")
    st.stop()
else:
    st.write(
        f"Number of papers with valid abstracts: {len(valid_abstract_df)}")


# Helper function to extract abstract features
def extract_abstract_features(abstract):
    if pd.isna(
            abstract
    ) or abstract == "No abstract available" or not abstract.strip():
        return {'word_count': 0, 'unique_words': 0, 'top_keywords': []}
    words = re.findall(r'\b\w+\b', abstract.lower())
    if not words:
        return {'word_count': 0, 'unique_words': 0, 'top_keywords': []}
    word_count = len(words)
    unique_words = len(set(words))
    stop_words = {
        'is', 'in', 'to', 'and', 'the', 'a', 'of', 'for', 'with', 'by'
    }
    word_freq = Counter(word for word in words if word not in stop_words)
    top_keywords = word_freq.most_common(3)
    return {
        'word_count': word_count,
        'unique_words': unique_words,
        'top_keywords': [kw[0] for kw in top_keywords]
    }


# Apply abstract feature extraction to valid abstracts
abstract_features = valid_abstract_df['Abstract'].apply(
    extract_abstract_features)
valid_abstract_df['Word Count'] = abstract_features.apply(
    lambda x: x['word_count'])
valid_abstract_df['Unique Words'] = abstract_features.apply(
    lambda x: x['unique_words'])
valid_abstract_df['Top Keywords'] = abstract_features.apply(
    lambda x: ', '.join(x['top_keywords']))

# Merge text features back into data_df
data_df = data_df.merge(valid_abstract_df[[
    'Article Id', 'Word Count', 'Unique Words', 'Top Keywords'
]],
                        on='Article Id',
                        how='left')
data_df['Word Count'] = data_df['Word Count'].fillna(0)
data_df['Unique Words'] = data_df['Unique Words'].fillna(0)
data_df['Top Keywords'] = data_df['Top Keywords'].fillna('')

# Debug: Check if word counts are non-zero
non_zero_abstracts = data_df[data_df['Word Count'] > 0]
st.write(
    f"Number of papers with non-zero word count: {len(non_zero_abstracts)}")

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "Citation Pattern Analysis", "Abstract Content Analysis",
    "Comparative Case Studies"
])

with tab1:
    st.header("🔍 Citation Pattern Analysis")

    # Diagnostic: Display data info
    st.subheader("Data Inspection")
    st.write("Columns in dataset:", data_df.columns.tolist())
    # Display available columns
    display_columns = ['Article Id', 'Cited By']
    if 'Title' in data_df.columns:
        display_columns.insert(1, 'Title')
    if 'Word Count' in data_df.columns:
        display_columns.append('Word Count')
    st.write("Sample data (first 5 rows):")
    st.dataframe(data_df[display_columns].head())

    # Citation distribution
    st.subheader("Citation Distribution")
    fig = px.histogram(data_df,
                       x='Cited By',
                       nbins=50,
                       title="Distribution of Citation Counts",
                       labels={
                           'Cited By': "Citation Count",
                           'count': "Number of Papers"
                       },
                       color_discrete_sequence=['#3366CC'])
    fig.update_layout(xaxis_title="Citation Count",
                      yaxis_title="Number of Papers")
    st.plotly_chart(fig, use_container_width=True)

    # Yearly citation trends
    st.subheader("Yearly Citation Trends")
    year_cols = [
        col for col in data_df.columns
        if col.isdigit() and 1992 <= int(col) <= 2023
    ]
    if year_cols:
        yearly_sums = data_df[year_cols].sum()
        fig = px.line(x=year_cols,
                      y=yearly_sums,
                      title="Total Citations by Year",
                      labels={
                          'x': "Year",
                          'y': "Total Citations"
                      })
        fig.update_traces(line_color='#3366CC')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No yearly citation data found.")

    # Citation statistics
    citation_stats = data_df['Cited By'].describe()
    st.write("Citation Statistics:")
    st.write(pd.DataFrame(citation_stats).T)

with tab2:
    st.header("🔢 Abstract Content Analysis")

    # Define low-citation threshold
    threshold_option = st.radio("Define low-citation threshold using:",
                                ["Percentile", "Absolute Value"],
                                horizontal=True)

    if threshold_option == "Percentile":
        percentile = st.slider("Citation percentile threshold",
                               10,
                               50,
                               25,
                               step=5)
        citation_threshold = np.percentile(valid_abstract_df['Cited By'],
                                           percentile)
    else:
        max_citation = int(valid_abstract_df['Cited By'].max())
        citation_threshold = st.slider(
            "Maximum citation count for low-cited papers", 0, max_citation,
            min(10, max_citation))

    # Create low and high citation groups
    low_cited = valid_abstract_df[valid_abstract_df['Cited By'] <=
                                  citation_threshold]
    high_cited = valid_abstract_df[valid_abstract_df['Cited By'] >
                                   citation_threshold]

    # Display threshold and group sizes
    st.write(f"Low citation threshold: ≤ {citation_threshold} citations")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Low-cited Papers",
            f"{len(low_cited)} ({len(low_cited)/len(valid_abstract_df)*100:.1f}%)"
        )
    with col2:
        st.metric(
            "High-cited Papers",
            f"{len(high_cited)} ({len(high_cited)/len(valid_abstract_df)*100:.1f}%)"
        )

    # Compare abstract features
    low_word_count_mean = low_cited['Word Count'].mean()
    high_word_count_mean = high_cited['Word Count'].mean()
    low_unique_words_mean = low_cited['Unique Words'].mean()
    high_unique_words_mean = high_cited['Unique Words'].mean()

    comparison_df = pd.DataFrame({
        'Group': ['Low-cited Papers', 'High-cited Papers'],
        'Average Word Count': [low_word_count_mean, high_word_count_mean],
        'Average Unique Words':
        [low_unique_words_mean, high_unique_words_mean]
    })

    st.write("Abstract Feature Comparison:")
    st.dataframe(comparison_df)

    # Plot comparison
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=['Low-cited', 'High-cited'],
               y=[low_word_count_mean, high_word_count_mean],
               name='Word Count',
               marker_color='#FF6B6B'))
    fig.add_trace(
        go.Bar(x=['Low-cited', 'High-cited'],
               y=[low_unique_words_mean, high_unique_words_mean],
               name='Unique Words',
               marker_color='#4ECDC4'))
    fig.update_layout(title="Abstract Features: Low vs. High Cited Papers",
                      xaxis_title="Citation Group",
                      yaxis_title="Average Value",
                      barmode='group')
    st.plotly_chart(fig, use_container_width=True)

    # Keyword analysis
    st.subheader("Keyword Analysis")
    low_keywords = Counter()
    high_keywords = Counter()
    for keywords in low_cited['Top Keywords']:
        if keywords:
            for kw in keywords.split(', '):
                low_keywords[kw] += 1
    for keywords in high_cited['Top Keywords']:
        if keywords:
            for kw in keywords.split(', '):
                high_keywords[kw] += 1

    keyword_df = pd.DataFrame({
        'Keyword':
        list(set(low_keywords.keys()) | set(high_keywords.keys())),
        'Low-cited Count': [
            low_keywords.get(kw, 0)
            for kw in set(low_keywords.keys()) | set(high_keywords.keys())
        ],
        'High-cited Count': [
            high_keywords.get(kw, 0)
            for kw in set(low_keywords.keys()) | set(high_keywords.keys())
        ]
    }).sort_values('Low-cited Count', ascending=False)

    st.write("Top Keywords Comparison (Top 10):")
    st.dataframe(keyword_df.head(10))

    # Analyze low-cited papers
    st.subheader("Characteristics of Low-Cited Papers")
    if len(low_cited) > 0:
        low_stats = low_cited[['Word Count', 'Unique Words']].describe()
        st.write("Abstract Feature Statistics for Low-Cited Papers:")
        st.write(low_stats)

        # Plot word count distribution
        fig = px.histogram(low_cited,
                           x='Word Count',
                           nbins=50,
                           title="Word Count Distribution in Low-Cited Papers",
                           labels={
                               'Word Count': "Word Count",
                               'count': "Number of Papers"
                           },
                           color_discrete_sequence=['#FF6B6B'])
        fig.update_layout(xaxis_title="Word Count",
                          yaxis_title="Number of Papers")
        st.plotly_chart(fig, use_container_width=True)

        # Reasons for low citations
        st.subheader("Potential Reasons for Low Citations")
        reasons = []
        if low_word_count_mean < high_word_count_mean * 0.8:
            reasons.append(
                f"**Short Abstracts**: Low-cited papers have shorter abstracts (avg: {low_word_count_mean:.1f} words) compared to high-cited (avg: {high_word_count_mean:.1f} words), potentially limiting clarity."
            )
        if low_unique_words_mean < high_unique_words_mean * 0.8:
            reasons.append(
                f"**Limited Vocabulary**: Low-cited papers use fewer unique words (avg: {low_unique_words_mean:.1f}) than high-cited (avg: {high_unique_words_mean:.1f}), possibly indicating narrower scope."
            )
        if keyword_df['Low-cited Count'].sum(
        ) < keyword_df['High-cited Count'].sum():
            reasons.append(
                "**Keyword Usage**: Low-cited papers use fewer prominent keywords, potentially reducing discoverability."
            )

        if not reasons:
            reasons.append(
                "**General Factors**: Low citations may result from limited relevance, poor visibility, or misalignment with trending research themes."
            )

        for reason in reasons:
            st.markdown(f"- {reason}")

with tab3:
    st.header("📋 Comparative Case Studies")

    # Sample low-cited papers
    low_citation_threshold = np.percentile(valid_abstract_df['Cited By'], 25)
    low_cited_papers = valid_abstract_df[valid_abstract_df['Cited By'] <=
                                         low_citation_threshold]

    if len(low_cited_papers) == 0:
        st.warning(
            "No low-cited papers with valid abstracts found for analysis.")
    else:
        sample_size = min(5, len(low_cited_papers))
        paper_samples = low_cited_papers.sample(sample_size)

        def find_similar_high_cited(paper_id,
                                    word_count,
                                    low_threshold,
                                    high_threshold=None):
            if high_threshold is None:
                high_threshold = np.percentile(valid_abstract_df['Cited By'],
                                               75)
            high_cited = valid_abstract_df[valid_abstract_df['Cited By'] >=
                                           high_threshold]
            similarities = []
            for idx, row in high_cited.iterrows():
                if row['Article Id'] != paper_id:
                    word_count_diff = abs(word_count - row['Word Count'])
                    similarities.append({
                        'Article Id': row['Article Id'],
                        'Citations': row['Cited By'],
                        'Word Count Difference': word_count_diff,
                        'Word Count': row['Word Count'],
                        'Unique Words': row['Unique Words'],
                        'Top Keywords': row['Top Keywords']
                    })
            if not similarities:
                return pd.DataFrame()
            sim_df = pd.DataFrame(similarities).sort_values(
                'Word Count Difference', ascending=True)
            return sim_df.head(3)

        for i, (idx, row) in enumerate(paper_samples.iterrows()):
            paper_id = row['Article Id']
            citations = row['Cited By']
            word_count = row['Word Count']
            unique_words = row['Unique Words']
            keywords = row['Top Keywords']
            title = row['Title']

            st.subheader(f"Case Study {i+1}: {title}")
            st.write(f"**Paper ID:** {paper_id}")
            st.write(f"**Citations:** {citations}")
            st.write(f"**Word Count:** {word_count}")
            st.write(f"**Unique Words:** {unique_words}")
            st.write(f"**Top Keywords:** {keywords}")

            # Find similar high-cited papers
            similar_high_cited = find_similar_high_cited(
                paper_id, word_count, low_citation_threshold)
            if len(similar_high_cited) == 0:
                st.warning("No similar high-cited papers found.")
                continue

            st.write(
                "**Comparable High-Cited Papers (by Abstract Word Count):**")
            for j, sim_row in similar_high_cited.iterrows():
                sim_paper_id = sim_row['Article Id']
                sim_citations = sim_row['Citations']
                sim_word_count = sim_row['Word Count']
                sim_unique_words = sim_row['Unique Words']
                sim_keywords = sim_row['Top Keywords']

                sim_title = valid_abstract_df.loc[
                    valid_abstract_df['Article Id'] == sim_paper_id,
                    'Title'].values[0]

                st.markdown(f"**{j+1}. {sim_title}**")
                st.write(
                    f"Citations: {sim_citations} (compared to {citations})")
                st.write(
                    f"Word Count: {sim_word_count} (difference: {sim_word_count - word_count})"
                )
                st.write(f"Unique Words: {sim_unique_words}")
                st.write(f"Top Keywords: {sim_keywords}")
                st.markdown("---")

            # Analysis of differences
            st.write("**Analysis of Key Differences:**")
            if len(similar_high_cited) > 0:
                avg_sim_word_count = similar_high_cited['Word Count'].mean()
                avg_sim_unique_words = similar_high_cited['Unique Words'].mean(
                )
                if avg_sim_word_count > word_count + 10:
                    st.write(
                        f"- High-cited papers have longer abstracts (avg: {avg_sim_word_count:.1f} vs. {word_count})"
                    )
                elif avg_sim_word_count < word_count - 10:
                    st.write(
                        f"- High-cited papers have shorter abstracts (avg: {avg_sim_word_count:.1f} vs. {word_count})"
                    )
                if avg_sim_unique_words > unique_words + 5:
                    st.write(
                        f"- High-cited papers use more unique words (avg: {avg_sim_unique_words:.1f} vs. {unique_words})"
                    )
                st.write(
                    "- Differences in keywords suggest varying research focus or framing."
                )
            st.markdown("---")

    # Recommendations
    st.subheader("💡 Recommendations for Improving Citation Potential")
    st.markdown("""
    To increase citation potential based on citation and abstract analysis:
    - **Enhance Abstract Clarity**: Write abstracts with sufficient length (e.g., 100–200 words) to clearly convey contributions.
    - **Use Diverse Keywords**: Incorporate a variety of domain-specific terms to improve discoverability.
    - **Align with Trends**: Frame research to connect with high-cited papers’ themes, as indicated by their keywords.
    - **Monitor Citation Trends**: Publish in years with higher citation activity, if possible.
    """)

# Footer
st.markdown("---")
st.markdown(
    "**Citation Analysis Platform**: Developed to help researchers understand and improve citation impact"
)
