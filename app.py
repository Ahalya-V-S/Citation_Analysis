import streamlit as st
import pandas as pd
import os
from utils.data_loader import load_citation_data, load_paper_metadata, load_topic_model_data, load_paper_text
from utils.citation_analysis import calculate_citation_metrics

st.set_page_config(
    page_title="Research Citation Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("Research Citation Analysis Platform")
    
    st.markdown("""
    ## Analyze citation patterns and identify factors affecting citation rates
    
    This platform helps researchers understand why some papers have low citation rates by analyzing:
    - Citation trends over time
    - Paper content and linguistic features
    - Topic modeling results
    - Correlations between paper attributes and citation counts
    """)
    
    st.sidebar.title("Data Upload")
    
    # Upload citation data
    citation_file = st.sidebar.file_uploader(
        "Upload citation data (CSV with Article ID, Title, Author, yearly citations)", 
        type=['csv']
    )
    
    # Upload topic model data
    topic_model_file = st.sidebar.file_uploader(
        "Upload topic model data (CSV with topic distributions)", 
        type=['csv']
    )
    
    # Upload paper text folder (zip file)
    paper_text_file = st.sidebar.file_uploader(
        "Upload paper texts (ZIP with folders named by Article ID)", 
        type=['zip']
    )
    
    # Check if all necessary files are uploaded
    if citation_file and topic_model_file:
        # Load the datasets
        citation_df = load_citation_data(citation_file)
        topic_model_df = load_topic_model_data(topic_model_file)
        
        # Store the data in session state for use across pages
        st.session_state['citation_data'] = citation_df
        st.session_state['topic_model_data'] = topic_model_df
        
        # Display data summary
        st.header("Data Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Citation Dataset")
            st.write(f"Number of papers: {len(citation_df)}")
            st.write(f"Year range: 1992-2023")
            st.write(f"Average citations per paper: {citation_df['Cited By'].mean():.1f}")
            
            # Calculate some basic metrics
            citation_metrics = calculate_citation_metrics(citation_df)
            st.write(f"Median citations: {citation_metrics['median_citations']}")
            st.write(f"Papers with zero citations: {citation_metrics['zero_citation_count']} ({citation_metrics['zero_citation_percent']:.1f}%)")
            
        with col2:
            st.subheader("Topic Model Dataset")
            st.write(f"Number of papers: {len(topic_model_df)}")
            st.write(f"Topic models available: LDA, HDP, CTM")
            st.write(f"Models with different topic counts: 5, 10")
            
        # If paper texts are provided, show info about them
        if paper_text_file:
            st.session_state['paper_text_file'] = paper_text_file
            st.subheader("Paper Text Files")
            st.write("Paper text files uploaded. You can analyze text content in the Text Analysis page.")
            
        # Display preview of citation data
        st.subheader("Citation Data Preview")
        st.dataframe(citation_df.head())
        
        # Display preview of topic model data
        st.subheader("Topic Model Data Preview")
        st.dataframe(topic_model_df.head())
        
        st.success("Data loaded successfully! Navigate to the specific analysis pages using the sidebar.")
        
        # Instructions for using the platform
        st.header("How to Use This Platform")
        st.markdown("""
        1. **Citation Trends**: Analyze how citations evolve over time
        2. **Text Analysis**: Examine linguistic features of papers 
        3. **Topic Modeling**: Explore topic distributions across papers
        4. **Correlation Analysis**: Identify factors correlated with citation counts
        5. **Predictive Analysis**: Use machine learning to predict citation potential
        """)
    else:
        st.info("Please upload the required datasets to begin analysis.")
        
        # Example of expected data formats
        st.header("Expected Data Formats")
        
        st.subheader("Citation Data Format")
        example_citation = pd.DataFrame({
            'Article Id': ['hep-th/9201002', 'hep-th/9201003'],
            'Title': ['Inomogeneous Quantum Groups as Symmetries of Phonons', 'Some Research Paper Title'],
            'Author': ['F.Bonechi et al.', 'J. Smith et al.'],
            'Cited By': [45, 12],
            '1992': [2, 0],
            '1993': [5, 1],
            '2023': [0, 0]
        })
        st.dataframe(example_citation)
        
        st.subheader("Topic Model Data Format")
        example_topic = pd.DataFrame({
            'ArticleID': ['hep-th/9201002', 'hep-th/9201003'],
            'LDA5': [0.2, 0.3],
            'LDA10': [0.15, 0.25],
            'HDP5': [0.25, 0.3],
            'CITATIONCOUNT': [45, 12]
        })
        st.dataframe(example_topic)
        
        st.subheader("Paper Text Format")
        st.code("""
Paper: hep-th/9201002
From: TARLINI%FI.INFN.IT@ICINECA.CINECA.IT
Date: Thu, 2 JAN 92 12:17 N   (6kb)

Title: Inomogeneous Quantum Groups as Symmetries of Phonons
Authors: F.Bonechi, E.Celeghini, R.Giachetti, E.Sorace and M.Tarlini
Comments: 5 pags. 0 figs
Journal-ref: Phys.Rev.Lett. 68 (1992) 3718-3720
\\
  The quantum deformed (1+1) Poincare' algebra is shown to be the kinematical
symmetry of the harmonic chain, whose spacing is given by the deformation
parameter. Phonons with their symmetries as well as multiphonon processes are
derived from the quantum group structure. Inhomogeneous quantum groups are thus
proposed as kinematical invariance of discrete systems.
\\
        """)

if __name__ == "__main__":
    main()
