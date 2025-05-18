import streamlit as st

# This MUST be the very first Streamlit command in the script
st.set_page_config(
    page_title="Text Analysis",
    page_icon="📝",
    layout="wide"
)

# Now we can import other libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
import plotly.graph_objects as go
from wordcloud import WordCloud
import base64
from io import BytesIO

# Now import text analysis utilities after ensuring NLTK resources are available
from utils.text_analysis import (
    preprocess_text, 
    extract_text_features, 
    compare_papers_text_features, 
    extract_keyphrases,
    analyze_common_keywords
)
from utils.data_loader import load_paper_text, load_paper_metadata

st.title("Text Analysis of Research Papers")

# Check if required data is available
if 'citation_data' not in st.session_state:
    st.error("No citation data available. Please upload data in the main page.")
    st.stop()

# Get the citation data
citation_df = st.session_state['citation_data']

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "Paper Content Analysis",
    "Comparison of Text Features",
    "Keyword Analysis"
])

with tab1:
    st.header("Paper Content Analysis")
    
    # Check if paper text file is available
    if 'paper_text_file' not in st.session_state:
        st.warning("No paper text file uploaded. Please upload the paper texts in the main page.")
        st.stop()
    
    # Load paper texts
    with st.spinner("Loading paper texts..."):
        try:
            paper_texts = load_paper_text(st.session_state['paper_text_file'])
            st.success(f"Loaded {len(paper_texts)} paper texts.")
        except Exception as e:
            st.error(f"Error loading paper texts: {str(e)}")
            st.stop()
    
    # Select paper to analyze
    paper_ids = list(paper_texts.keys())
    
    if not paper_ids:
        st.error("No paper texts found in the uploaded file.")
        st.stop()
    
    selected_paper = st.selectbox(
        "Select a paper to analyze",
        options=paper_ids
    )
    
    if selected_paper:
        # Get paper text
        paper_text = paper_texts[selected_paper]
        
        # Get paper metadata
        paper_metadata = load_paper_metadata(paper_text)
        
        # Display paper metadata
        st.subheader("Paper Metadata")
        
        # Create an expander for viewing the raw text if needed
        with st.expander("View Raw Paper Content"):
            st.code(paper_text[:2000] + "..." if len(paper_text) > 2000 else paper_text)
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            if paper_metadata['paper']:
                st.markdown(f"**Paper ID:** {paper_metadata['paper']}")
            
            st.markdown(f"**Title:** {paper_metadata['title'] or 'N/A'}")
            st.markdown(f"**Authors:** {paper_metadata['authors'] or 'N/A'}")
            
            if paper_metadata['from']:
                st.markdown(f"**From:** {paper_metadata['from']}")
            
            # Find citation count if available
            citation_count = "N/A"
            if 'Article Id' in citation_df.columns and selected_paper in citation_df['Article Id'].values:
                citation_count = citation_df[citation_df['Article Id'] == selected_paper]['Cited By'].iloc[0]
            st.markdown(f"**Citation Count:** {citation_count}")
        
        with col2:
            st.markdown(f"**Date:** {paper_metadata['date'] or 'N/A'}")
            
            if paper_metadata['comments']:
                st.markdown(f"**Comments:** {paper_metadata['comments']}")
            
            st.markdown(f"**Journal:** {paper_metadata['journal'] or 'N/A'}")
        
        # Display abstract with special formatting
        st.subheader("Abstract")
        if paper_metadata['abstract']:
            st.markdown(f"<div style='background-color:#1e1e1e; color:#ffffff; padding:15px; border-radius:5px'>{paper_metadata['abstract']}</div>", unsafe_allow_html=True)
        else:
            st.info("Abstract not available for this paper.")
        
        # Extract text features
        with st.spinner("Analyzing text features..."):
            text_features = extract_text_features(paper_text)
        
        # Display text statistics
        st.subheader("Text Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Token Count", text_features['token_count'])
            st.metric("Lexical Diversity", f"{text_features['lexical_diversity']:.3f}")
            st.metric("Noun Ratio", f"{text_features['noun_ratio']:.3f}")
        
        with col2:
            st.metric("Sentence Count", text_features['sentence_count'])
            st.metric("Avg Sentence Length", f"{text_features['avg_sentence_length']:.1f}")
            st.metric("Verb Ratio", f"{text_features['verb_ratio']:.3f}")
        
        with col3:
            st.metric("Avg Word Length", f"{text_features['avg_word_length']:.2f}")
            st.metric("Flesch Reading Ease", f"{text_features['flesch_reading_ease']:.1f}")
            st.metric("Adjective Ratio", f"{text_features['adj_ratio']:.3f}")
        
        # Most common words
        st.subheader("Most Frequent Words")
        
        # Create dataframe for most common words
        common_words_df = pd.DataFrame(
            text_features['most_common_words'],
            columns=['Word', 'Frequency']
        )
        
        # Display as bar chart
        fig = px.bar(
            common_words_df.head(15),
            x='Word',
            y='Frequency',
            title="Most Frequent Words",
            color='Frequency',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            xaxis_title="Word",
            yaxis_title="Frequency",
            xaxis={'categoryorder':'total descending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Generate word cloud
        st.subheader("Word Cloud")
        
        # Function to create word cloud
        def create_wordcloud(text):
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                max_words=150,
                contour_width=3,
                contour_color='steelblue'
            ).generate(text)
            
            # Convert to image
            img = wordcloud.to_image()
            
            # Convert to base64 string
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return img_str
        
        # Preprocess text for word cloud
        preprocessed_text, _, _ = preprocess_text(paper_text)
        
        # Create and display word cloud
        wordcloud_img = create_wordcloud(preprocessed_text)
        st.markdown(f'<img src="data:image/png;base64,{wordcloud_img}" style="width:100%">', unsafe_allow_html=True)
        
        # Extract key phrases
        st.subheader("Key Phrases")
        keyphrases = extract_keyphrases(paper_text, n=15)
        
        # Display key phrases
        st.write(", ".join(keyphrases))
        
        # Reading level interpretation
        st.subheader("Readability Analysis")
        
        flesch_reading_ease = text_features['flesch_reading_ease']
        flesch_kincaid_grade = text_features['flesch_kincaid_grade']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Flesch Reading Ease", f"{flesch_reading_ease:.1f}")
            
            # Interpretation of Flesch Reading Ease
            if flesch_reading_ease >= 90:
                readability = "Very Easy (5th grade)"
            elif flesch_reading_ease >= 80:
                readability = "Easy (6th grade)"
            elif flesch_reading_ease >= 70:
                readability = "Fairly Easy (7th grade)"
            elif flesch_reading_ease >= 60:
                readability = "Standard (8th-9th grade)"
            elif flesch_reading_ease >= 50:
                readability = "Fairly Difficult (10th-12th grade)"
            elif flesch_reading_ease >= 30:
                readability = "Difficult (College level)"
            else:
                readability = "Very Difficult (College graduate level)"
                
            st.write(f"Interpretation: {readability}")
        
        with col2:
            st.metric("Flesch-Kincaid Grade Level", f"{flesch_kincaid_grade:.1f}")
            st.write(f"This text is at the reading level of grade {flesch_kincaid_grade:.1f}")
            
            # Compare to average
            if 'paper_text_file' in st.session_state and len(paper_texts) > 1:
                all_grades = []
                for pid, text in paper_texts.items():
                    if pid != selected_paper:  # Exclude current paper
                        features = extract_text_features(text)
                        all_grades.append(features['flesch_kincaid_grade'])
                
                if all_grades:
                    avg_grade = sum(all_grades) / len(all_grades)
                    diff = flesch_kincaid_grade - avg_grade
                    st.write(f"Average grade level of other papers: {avg_grade:.1f}")
                    st.write(f"This paper is {abs(diff):.1f} grade levels {'higher' if diff > 0 else 'lower'} than average")

with tab2:
    st.header("Comparison of Text Features")
    
    # Check if paper text file is available
    if 'paper_text_file' not in st.session_state:
        st.warning("No paper text file uploaded. Please upload the paper texts in the main page.")
        st.stop()
    
    # Load paper texts if not already done
    if 'paper_texts' not in locals():
        with st.spinner("Loading paper texts..."):
            try:
                paper_texts = load_paper_text(st.session_state['paper_text_file'])
                st.success(f"Loaded {len(paper_texts)} paper texts.")
            except Exception as e:
                st.error(f"Error loading paper texts: {str(e)}")
                st.stop()
    
    # Number of papers to compare (with safety checks)
    num_papers_available = len(paper_texts) if paper_texts else 0
    
    if num_papers_available < 2:
        st.warning("Insufficient papers available for comparison. At least 2 papers are needed.")
        st.stop()
    
    max_papers = min(20, num_papers_available)
    default_papers = min(10, num_papers_available)
    
    # Create tabs for different comparison views
    comp_tab1, comp_tab2, comp_tab3 = st.tabs(["Paper Selection", "Feature Comparison", "Keyword Analysis"])
    
    with comp_tab1:
        st.subheader("Select Papers to Compare")
        
        # Add option to select different paper datasets
        dataset_selection = st.radio(
            "Selection Method",
            options=["Use all available papers", "Select by number", "Select specific papers"],
            index=1
        )
        
        # Get paper IDs
        paper_ids = list(paper_texts.keys())
        
        if dataset_selection == "Use all available papers":
            selected_papers = paper_ids[:min(20, len(paper_ids))]
            st.info(f"Using the first {len(selected_papers)} papers from the dataset")
        
        elif dataset_selection == "Select by number":
            num_papers = st.slider(
                "Number of papers to compare",
                min_value=2,
                max_value=max_papers,
                value=default_papers
            )
            selected_papers = paper_ids[:num_papers]
            st.info(f"Using the first {len(selected_papers)} papers from the dataset")
        
        else:  # Select specific papers
            # Add a select all button
            if st.button("Select All Available Papers (max 20)"):
                selected_papers = paper_ids[:min(20, len(paper_ids))]
            else:
                # Add options to filter or search papers
                filter_option = st.checkbox("Enable search/filter by ID", value=False)
                
                if filter_option:
                    search_term = st.text_input("Filter papers by ID (partial match)").strip().lower()
                    if search_term:
                        filtered_ids = [pid for pid in paper_ids if search_term in pid.lower()]
                        if not filtered_ids:
                            st.warning(f"No papers match '{search_term}'")
                            filtered_ids = paper_ids
                    else:
                        filtered_ids = paper_ids
                else:
                    filtered_ids = paper_ids
                
                # Show papers in a multiselect
                selected_papers = st.multiselect(
                    "Select specific papers to compare",
                    options=filtered_ids,
                    default=filtered_ids[:min(5, len(filtered_ids))]
                )
    
        # Display number of selected papers
        st.write(f"Selected {len(selected_papers)} papers for comparison")
        
        # If no papers selected, show a warning
        if not selected_papers:
            st.warning("Please select at least one paper to continue")
            st.stop()
    
    # Since we have papers selected, continue with analysis
    # Extract text features for selected papers
    with st.spinner("Analyzing text features..."):
        papers_features = {}
        for paper_id in selected_papers:
            papers_features[paper_id] = extract_text_features(paper_texts[paper_id])
    
    # Create citation count dictionary if available
    citation_counts = {}
    if 'Article Id' in citation_df.columns:
        for paper_id in selected_papers:
            if paper_id in citation_df['Article Id'].values:
                citation_counts[paper_id] = citation_df[citation_df['Article Id'] == paper_id]['Cited By'].iloc[0]
    
    # Compare text features
    comparison_df = compare_papers_text_features(papers_features, citation_counts)
    
    with comp_tab2:
        # Display comparison table
        st.subheader("Text Features Comparison")
        st.dataframe(comparison_df, use_container_width=True)
        
        # Create scatter plot matrix
        st.subheader("Feature Relationships")
        
        # Extended feature options including the new metrics
        feature_options = [
            'avg_sentence_length', 
            'avg_word_length', 
            'lexical_diversity',
            'flesch_reading_ease',
            'flesch_kincaid_grade',
            'noun_ratio',
            'verb_ratio',
            'adj_ratio',
            'adv_ratio'
        ]
        
        # Add advanced metrics if available
        if 'technical_complexity' in comparison_df.columns:
            feature_options.append('technical_complexity')
        
        if 'formality_score' in comparison_df.columns:
            feature_options.append('formality_score')
        
        # Select features to plot
        plot_features = st.multiselect(
            "Select features to compare",
            options=feature_options,
            default=[
                'avg_sentence_length', 
                'lexical_diversity',
                'flesch_reading_ease',
                'noun_ratio'
            ]
        )
        
        if plot_features:
            # Include citation count in plot if available
            if 'citation_count' in comparison_df.columns:
                plot_features.append('citation_count')
                
            # Create scatter plot matrix
            fig = px.scatter_matrix(
                comparison_df,
                dimensions=plot_features,
                hover_data=['paper_id'],
                title="Relationships Between Text Features"
            )
            
            fig.update_layout(
                height=700,
                width=900
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Analyze correlations with citations if available
            if 'citation_count' in comparison_df.columns:
                st.subheader("Correlations with Citation Count")
                
                # Calculate correlations
                correlations = []
                for feature in plot_features:
                    if feature != 'citation_count' and feature in comparison_df.columns:
                        corr = comparison_df[feature].corr(comparison_df['citation_count'])
                        correlations.append({
                            'Feature': feature.replace('_', ' ').title(),
                            'Correlation': corr
                        })
                
                # Create correlation dataframe
                corr_df = pd.DataFrame(correlations)
                corr_df = corr_df.sort_values('Correlation', ascending=False)
                
                # Display correlation table
                st.dataframe(corr_df)
                
                # Create correlation bar chart
                fig = px.bar(
                    corr_df,
                    x='Feature',
                    y='Correlation',
                    title="Correlation of Text Features with Citation Count",
                    color='Correlation',
                    color_continuous_scale='RdBu_r',
                    labels={'Correlation': 'Pearson Correlation'}
                )
                
                fig.update_layout(
                    xaxis_title="Text Feature",
                    yaxis_title="Correlation with Citation Count",
                    yaxis=dict(range=[-1, 1])
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
    with comp_tab3:
        st.subheader("Keyword Analysis Across Papers")
        
        # Extract common keywords across papers
        try:
            keyword_df, keyword_frequency_df = analyze_common_keywords(papers_features, top_n=15)
            
            # Display top keywords
            st.write("Most Common Keywords Across Selected Papers")
            
            # Create bar chart for keyword frequency
            fig = px.bar(
                keyword_df,
                x='Keyword',
                y='Frequency',
                title="Most Common Keywords",
                color='Frequency',
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(
                xaxis_title="Keyword",
                yaxis_title="Frequency",
                xaxis={'categoryorder':'total descending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display heatmap of keywords by paper
            st.write("Keyword Presence by Paper")
            
            # Prepare data for heatmap
            heatmap_data = keyword_frequency_df.set_index('paper_id')
            
            # Create heatmap
            fig = px.imshow(
                heatmap_data,
                labels=dict(x="Keyword", y="Paper ID", color="Present"),
                x=heatmap_data.columns,
                y=heatmap_data.index,
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(
                height=max(400, len(selected_papers) * 25),
                width=900
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # If citation data is available, analyze keyword-citation relationships
            if citation_counts and len(citation_counts) > 3:
                st.subheader("Keywords vs. Citation Counts")
                
                # Merge keyword data with citation counts
                keyword_citation_data = []
                
                for paper_id in keyword_frequency_df['paper_id']:
                    if paper_id in citation_counts:
                        paper_row = keyword_frequency_df[keyword_frequency_df['paper_id'] == paper_id].iloc[0].to_dict()
                        paper_row['citation_count'] = citation_counts[paper_id]
                        keyword_citation_data.append(paper_row)
                
                if keyword_citation_data:
                    keyword_citation_df = pd.DataFrame(keyword_citation_data)
                    
                    # Calculate average citation count for papers with/without each keyword
                    keyword_impact = []
                    
                    for keyword in keyword_df['Keyword']:
                        with_keyword = keyword_citation_df[keyword_citation_df[keyword] == 1]['citation_count']
                        without_keyword = keyword_citation_df[keyword_citation_df[keyword] == 0]['citation_count']
                        
                        if len(with_keyword) > 0 and len(without_keyword) > 0:
                            avg_with = with_keyword.mean()
                            avg_without = without_keyword.mean()
                            difference = avg_with - avg_without
                            
                            keyword_impact.append({
                                'Keyword': keyword,
                                'Avg Citations With': avg_with,
                                'Avg Citations Without': avg_without,
                                'Difference': difference
                            })
                    
                    if keyword_impact:
                        impact_df = pd.DataFrame(keyword_impact)
                        impact_df = impact_df.sort_values('Difference', ascending=False)
                        
                        # Create bar chart of keyword impact
                        fig = px.bar(
                            impact_df,
                            x='Keyword',
                            y='Difference',
                            title="Difference in Average Citations With vs. Without Keyword",
                            color='Difference',
                            color_continuous_scale='RdBu_r'
                        )
                        
                        fig.update_layout(
                            xaxis_title="Keyword",
                            yaxis_title="Difference in Average Citations",
                            xaxis={'categoryorder':'total descending'}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display table of keyword impact
                        st.write("Impact of Keywords on Citation Counts")
                        st.dataframe(impact_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error analyzing keywords: {str(e)}")
            st.info("This could be due to missing 'keyphrases' in the extracted text features. Try analyzing fewer papers or check the logs for more details.")

with tab3:
    st.header("Keyword Analysis")
    
    # Check if paper text file is available
    if 'paper_text_file' not in st.session_state:
        st.warning("No paper text file uploaded. Please upload the paper texts in the main page.")
        st.stop()
    
    # Load paper texts if not already done
    if 'paper_texts' not in locals():
        with st.spinner("Loading paper texts..."):
            try:
                paper_texts = load_paper_text(st.session_state['paper_text_file'])
                st.success(f"Loaded {len(paper_texts)} paper texts.")
            except Exception as e:
                st.error(f"Error loading paper texts: {str(e)}")
                st.stop()
    
    # Analyze keywords across all papers
    st.subheader("Keyword Analysis Across All Papers")
    
    with st.spinner("Extracting keywords from all papers..."):
        # Extract keywords from all papers
        all_keyphrases = []
        paper_keyphrases = {}
        
        for paper_id, text in paper_texts.items():
            keyphrases = extract_keyphrases(text, n=20)
            paper_keyphrases[paper_id] = keyphrases
            all_keyphrases.extend(keyphrases)
        
        # Count keyphrases
        keyphrase_counts = Counter(all_keyphrases)
        top_keyphrases = keyphrase_counts.most_common(30)
        
        # Create dataframe
        keyphrase_df = pd.DataFrame(top_keyphrases, columns=['Keyword', 'Frequency'])
    
    # Display top keywords
    st.write(f"Top keywords across {len(paper_texts)} papers:")
    
    # Create bar chart
    fig = px.bar(
        keyphrase_df,
        x='Keyword',
        y='Frequency',
        title="Most Common Keywords Across All Papers",
        color='Frequency',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis_title="Keyword",
        yaxis_title="Frequency",
        xaxis=dict(tickangle=45),
        xaxis_tickfont=dict(size=10),
        xaxis_tickmode='array',
        xaxis_tickvals=keyphrase_df['Keyword'].tolist()
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Keyword comparison between high and low cited papers
    st.subheader("Keyword Comparison: High vs. Low Citation")
    
    # Check if citation data is available
    if 'Article Id' in citation_df.columns:
        # Define high and low citation thresholds
        citation_threshold = st.slider(
            "Citation threshold percentile",
            min_value=10,
            max_value=90,
            value=75,
            step=5,
            help="Papers above this percentile are considered 'high cited', below are 'low cited'"
        )
        
        # Calculate threshold values
        high_threshold = np.percentile(citation_df['Cited By'], citation_threshold)
        low_threshold = np.percentile(citation_df['Cited By'], 100 - citation_threshold)
        
        st.write(f"High citation threshold: ≥ {high_threshold} citations")
        st.write(f"Low citation threshold: ≤ {low_threshold} citations")
        
        # Identify high and low cited papers
        high_cited_ids = citation_df[citation_df['Cited By'] >= high_threshold]['Article Id'].tolist()
        low_cited_ids = citation_df[citation_df['Cited By'] <= low_threshold]['Article Id'].tolist()
        
        # Filter to papers that have text available
        high_cited_ids = [pid for pid in high_cited_ids if pid in paper_texts]
        low_cited_ids = [pid for pid in low_cited_ids if pid in paper_texts]
        
        st.write(f"Found {len(high_cited_ids)} high-cited papers and {len(low_cited_ids)} low-cited papers with text available")
        
        if high_cited_ids and low_cited_ids:
            # Extract keywords for high and low cited papers
            high_keyphrases = []
            for paper_id in high_cited_ids:
                if paper_id in paper_keyphrases:
                    high_keyphrases.extend(paper_keyphrases[paper_id])
                else:
                    keyphrases = extract_keyphrases(paper_texts[paper_id], n=20)
                    high_keyphrases.extend(keyphrases)
            
            low_keyphrases = []
            for paper_id in low_cited_ids:
                if paper_id in paper_keyphrases:
                    low_keyphrases.extend(paper_keyphrases[paper_id])
                else:
                    keyphrases = extract_keyphrases(paper_texts[paper_id], n=20)
                    low_keyphrases.extend(keyphrases)
            
            # Count keyphrases
            high_counts = Counter(high_keyphrases)
            low_counts = Counter(low_keyphrases)
            
            # Normalize counts by number of papers
            high_normalized = {k: v / len(high_cited_ids) for k, v in high_counts.items()}
            low_normalized = {k: v / len(low_cited_ids) for k, v in low_counts.items()}
            
            # Get common keywords
            common_keywords = set(high_normalized.keys()) & set(low_normalized.keys())
            
            # Calculate ratio for common keywords
            keyword_ratios = []
            for keyword in common_keywords:
                if low_normalized[keyword] > 0:  # Avoid division by zero
                    ratio = high_normalized[keyword] / low_normalized[keyword]
                    keyword_ratios.append({
                        'Keyword': keyword,
                        'High-Cited Frequency': high_normalized[keyword],
                        'Low-Cited Frequency': low_normalized[keyword],
                        'Ratio (High/Low)': ratio
                    })
            
            # Create dataframe
            ratio_df = pd.DataFrame(keyword_ratios)
            
            # Sort by ratio
            ratio_df = ratio_df.sort_values('Ratio (High/Low)', ascending=False)
            
            # Display table
            st.write("Keywords with largest difference between high and low cited papers:")
            st.dataframe(ratio_df.head(20))
            
            # Create visualization
            top_diff_keywords = ratio_df.head(15)['Keyword'].tolist()
            
            # Create comparison dataframe for visualization
            viz_data = []
            for keyword in top_diff_keywords:
                viz_data.append({
                    'Keyword': keyword,
                    'Citation Group': 'High-Cited',
                    'Frequency': high_normalized[keyword]
                })
                viz_data.append({
                    'Keyword': keyword,
                    'Citation Group': 'Low-Cited',
                    'Frequency': low_normalized[keyword]
                })
            
            viz_df = pd.DataFrame(viz_data)
            
            # Create grouped bar chart
            fig = px.bar(
                viz_df,
                x='Keyword',
                y='Frequency',
                color='Citation Group',
                barmode='group',
                title="Keyword Frequency: High vs. Low Cited Papers",
                color_discrete_map={
                    'High-Cited': '#3366CC',
                    'Low-Cited': '#FF9900'
                }
            )
            
            fig.update_layout(
                xaxis_title="Keyword",
                yaxis_title="Average Frequency per Paper",
                xaxis=dict(tickangle=45)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Visualize keywords unique to high-cited papers
            st.subheader("Keywords Unique to High-Cited Papers")
            
            high_only_keywords = set(high_normalized.keys()) - set(low_normalized.keys())
            high_only_counts = {k: v for k, v in high_normalized.items() if k in high_only_keywords}
            
            # Sort and get top keywords
            high_only_sorted = sorted(high_only_counts.items(), key=lambda x: x[1], reverse=True)
            high_only_top = high_only_sorted[:20]
            
            # Create dataframe
            high_only_df = pd.DataFrame(high_only_top, columns=['Keyword', 'Frequency'])
            
            # Create bar chart
            fig = px.bar(
                high_only_df,
                x='Keyword',
                y='Frequency',
                title="Keywords Found Only in High-Cited Papers",
                color='Frequency',
                color_continuous_scale='Blues'
            )
            
            fig.update_layout(
                xaxis_title="Keyword",
                yaxis_title="Average Frequency per Paper",
                xaxis=dict(tickangle=45)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation
            st.subheader("Interpretation")
            st.markdown("""
            The keyword analysis reveals patterns in language usage between highly-cited and low-cited papers:
            
            1. **Keyword Ratio**: A higher ratio indicates keywords that appear more frequently in highly-cited papers
            2. **Unique Keywords**: Some keywords appear exclusively in highly-cited papers
            
            These differences may indicate topics, approaches, or terminology that correlate with higher citation rates.
            """)
        else:
            st.warning("Not enough papers with both citation data and text content to perform comparison.")
    else:
        st.warning("Citation data does not contain Article IDs for matching with paper texts.")
