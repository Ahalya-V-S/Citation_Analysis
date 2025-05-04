import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from utils.visualization import plot_topic_distributions
import seaborn as sns

st.set_page_config(
    page_title="Topic Model Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("Topic Model Analysis")

# Check if required data is available
if 'citation_data' not in st.session_state or 'topic_model_data' not in st.session_state:
    st.error("Missing required datasets. Please upload both citation and topic model data in the main page.")
    st.stop()

# Get the data
citation_df = st.session_state['citation_data']
topic_model_df = st.session_state['topic_model_data']

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "Topic Distribution",
    "Topics and Citations",
    "Paper Topic Explorer"
])

with tab1:
    st.header("Topic Distribution Analysis")
    
    # Identify available topic models
    model_types = []
    topic_counts = []
    
    for col in topic_model_df.columns:
        # Check for LDA, HDP, CTM models with topic counts 5 or 10
        if any(prefix in col for prefix in ['LDA', 'HDP', 'CTM']):
            model_type = ''.join([c for c in col if not c.isdigit()])
            if model_type not in model_types:
                model_types.append(model_type)
            
            # Extract topic count
            topic_count = ''.join([c for c in col if c.isdigit()])
            if topic_count and int(topic_count) not in topic_counts:
                topic_counts.append(int(topic_count))
    
    if not model_types or not topic_counts:
        st.error("No topic model data found in the uploaded dataset.")
        st.stop()
    
    # Select model type and topic count
    col1, col2 = st.columns(2)
    
    with col1:
        selected_model = st.selectbox(
            "Select topic model type",
            options=model_types
        )
    
    with col2:
        selected_topic_count = st.selectbox(
            "Select number of topics",
            options=topic_counts
        )
    
    # Construct model name
    model_name = f"{selected_model}{selected_topic_count}"
    
    # Display model information
    st.subheader(f"Topic Model: {model_name}")
    
    # Get topic columns
    topic_cols = []
    for i in range(1, selected_topic_count + 1):
        col_name = f"{model_name}_{i}"
        
        # Check different possible column naming patterns
        if col_name in topic_model_df.columns:
            topic_cols.append(col_name)
        elif f"{selected_model}{i}" in topic_model_df.columns:
            topic_cols.append(f"{selected_model}{i}")
        elif f"{selected_model}_{i}" in topic_model_df.columns:
            topic_cols.append(f"{selected_model}_{i}")
    
    if not topic_cols:
        st.error(f"Could not find topic columns for model {model_name}.")
        st.stop()
    
    # Calculate topic distribution
    topic_dist = topic_model_df[topic_cols].mean()
    
    # Create topic names
    topic_names = [f"Topic {i+1}" for i in range(len(topic_cols))]
    
    # Create dataframe for plotting
    topic_dist_df = pd.DataFrame({
        'Topic': topic_names,
        'Average Weight': topic_dist.values
    })
    
    # Sort by weight
    topic_dist_df = topic_dist_df.sort_values('Average Weight', ascending=False)
    
    # Plot topic distribution
    fig = px.bar(
        topic_dist_df,
        x='Topic',
        y='Average Weight',
        title=f"Average Topic Distribution ({model_name})",
        color='Average Weight',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis_title="Topic",
        yaxis_title="Average Weight",
        xaxis={'categoryorder':'total descending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display topic distribution statistics
    st.subheader("Topic Distribution Statistics")
    
    # Calculate topic dominance for each paper
    dominant_topics = []
    for _, row in topic_model_df.iterrows():
        topic_values = [row[col] for col in topic_cols if col in row.index]
        if topic_values:
            dominant_topic = topic_names[np.argmax(topic_values)]
            dominant_topics.append(dominant_topic)
    
    # Count papers per dominant topic
    topic_counts = pd.Series(dominant_topics).value_counts()
    
    # Create dataframe for plotting
    topic_count_df = pd.DataFrame({
        'Topic': topic_counts.index,
        'Paper Count': topic_counts.values
    })
    
    # Plot dominant topics
    fig = px.pie(
        topic_count_df,
        values='Paper Count',
        names='Topic',
        title="Papers by Dominant Topic",
        hole=0.3
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate topic diversity
    st.subheader("Topic Diversity")
    
    # Get entropy of topic distribution
    entropies = []
    for _, row in topic_model_df.iterrows():
        topic_values = [row[col] for col in topic_cols if col in row.index]
        if topic_values:
            # Normalize to ensure they sum to 1
            topic_values = np.array(topic_values) / sum(topic_values)
            
            # Calculate entropy
            entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in topic_values)
            
            # Normalize by maximum possible entropy
            max_entropy = np.log2(len(topic_values))
            if max_entropy > 0:
                normalized_entropy = entropy / max_entropy
            else:
                normalized_entropy = 0
                
            entropies.append(normalized_entropy)
    
    # Create histogram of entropies
    fig = px.histogram(
        x=entropies,
        nbins=30,
        title="Distribution of Topic Diversity (Normalized Entropy)",
        labels={"x": "Normalized Entropy", "y": "Number of Papers"},
        color_discrete_sequence=['#3366CC']
    )
    
    fig.update_layout(
        xaxis_title="Topic Diversity (Normalized Entropy)",
        yaxis_title="Number of Papers",
        xaxis=dict(range=[0, 1])
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    st.markdown("""
    **Interpretation:**
    
    - **Topic Distribution**: Shows the average weight of each topic across all papers
    - **Dominant Topics**: Shows how many papers have each topic as their primary topic
    - **Topic Diversity**: Measures how evenly topics are distributed within each paper
      - Higher values (closer to 1) indicate papers with more even topic distributions
      - Lower values (closer to 0) indicate papers dominated by a single topic
    """)

with tab2:
    st.header("Topics and Citations")
    
    # Select model type and topic count (same as tab 1)
    if 'selected_model' not in locals() or 'selected_topic_count' not in locals():
        col1, col2 = st.columns(2)
        
        with col1:
            selected_model = st.selectbox(
                "Select topic model type",
                options=model_types,
                key="tab2_model"
            )
        
        with col2:
            selected_topic_count = st.selectbox(
                "Select number of topics",
                options=topic_counts,
                key="tab2_topics"
            )
        
    # Construct model name
    model_name = f"{selected_model}{selected_topic_count}"
    
    # Get topic columns
    topic_cols = []
    for i in range(1, selected_topic_count + 1):
        col_name = f"{model_name}_{i}"
        
        # Check different possible column naming patterns
        if col_name in topic_model_df.columns:
            topic_cols.append(col_name)
        elif f"{selected_model}{i}" in topic_model_df.columns:
            topic_cols.append(f"{selected_model}{i}")
        elif f"{selected_model}_{i}" in topic_model_df.columns:
            topic_cols.append(f"{selected_model}_{i}")
    
    if not topic_cols:
        st.error(f"Could not find topic columns for model {model_name}.")
        st.stop()
    
    # Prepare data for correlation analysis
    if 'CITATIONCOUNT' in topic_model_df.columns:
        # Use citation count from topic model data
        topic_citation_df = topic_model_df.copy()
    elif 'ArticleID' in topic_model_df.columns and 'Article Id' in citation_df.columns:
        # Merge topic data with citation data
        topic_citation_df = pd.merge(
            topic_model_df,
            citation_df[['Article Id', 'Cited By']],
            left_on='ArticleID',
            right_on='Article Id',
            how='inner'
        )
        # Add citation count column
        topic_citation_df['CITATIONCOUNT'] = topic_citation_df['Cited By']
    else:
        st.error("Cannot link topic model data with citation counts.")
        st.stop()
    
    # Calculate correlation between topics and citations
    topic_correlations = []
    for col in topic_cols:
        corr = topic_citation_df[col].corr(topic_citation_df['CITATIONCOUNT'])
        topic_correlations.append({
            'Topic': col.replace(f"{model_name}_", "Topic ").replace(f"{model_name}", "Topic "),
            'Correlation': corr
        })
    
    # Create dataframe for plotting
    corr_df = pd.DataFrame(topic_correlations)
    
    # Sort by absolute correlation value
    corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
    corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
    corr_df = corr_df.drop('Abs_Correlation', axis=1)
    
    # Plot correlation
    fig = px.bar(
        corr_df,
        x='Topic',
        y='Correlation',
        title=f"Correlation Between Topics and Citation Count ({model_name})",
        color='Correlation',
        color_continuous_scale='RdBu_r',
        labels={'Correlation': 'Pearson Correlation'}
    )
    
    fig.update_layout(
        xaxis_title="Topic",
        yaxis_title="Correlation with Citation Count",
        yaxis=dict(range=[-1, 1])
    )
    
    # Add a horizontal line at zero
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=0,
        x1=len(topic_correlations) - 0.5,
        y1=0,
        line=dict(
            color="black",
            width=1,
            dash="dash",
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyze topics of high vs. low cited papers
    st.subheader("Topic Distribution in High vs. Low Cited Papers")
    
    # Define high and low citation thresholds
    citation_percentile = st.slider(
        "Citation threshold percentile",
        min_value=10,
        max_value=90,
        value=75,
        step=5,
        help="Papers above this percentile are considered 'high cited', below are 'low cited'"
    )
    
    # Calculate threshold values
    high_threshold = np.percentile(topic_citation_df['CITATIONCOUNT'], citation_percentile)
    low_threshold = np.percentile(topic_citation_df['CITATIONCOUNT'], 100 - citation_percentile)
    
    st.write(f"High citation threshold: ≥ {high_threshold} citations")
    st.write(f"Low citation threshold: ≤ {low_threshold} citations")
    
    # Create high and low citation dataframes
    high_cited_df = topic_citation_df[topic_citation_df['CITATIONCOUNT'] >= high_threshold]
    low_cited_df = topic_citation_df[topic_citation_df['CITATIONCOUNT'] <= low_threshold]
    
    st.write(f"Number of high-cited papers: {len(high_cited_df)}")
    st.write(f"Number of low-cited papers: {len(low_cited_df)}")
    
    # Calculate mean topic distribution for each group
    high_topic_dist = high_cited_df[topic_cols].mean()
    low_topic_dist = low_cited_df[topic_cols].mean()
    
    # Create dataframe for comparison
    topic_names = [col.replace(f"{model_name}_", "Topic ").replace(f"{model_name}", "Topic ") for col in topic_cols]
    
    comparison_data = []
    for i, topic in enumerate(topic_cols):
        topic_name = topic_names[i]
        high_weight = high_topic_dist[topic]
        low_weight = low_topic_dist[topic]
        diff = high_weight - low_weight
        
        comparison_data.append({
            'Topic': topic_name,
            'High-Cited Weight': high_weight,
            'Low-Cited Weight': low_weight,
            'Difference': diff
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by absolute difference
    comparison_df['Abs_Difference'] = comparison_df['Difference'].abs()
    comparison_df = comparison_df.sort_values('Abs_Difference', ascending=False)
    comparison_df = comparison_df.drop('Abs_Difference', axis=1)
    
    # Display comparison table
    st.write("Topic weight comparison between high and low cited papers:")
    st.dataframe(comparison_df)
    
    # Create grouped bar chart
    fig = go.Figure()
    
    # Add high cited bars
    fig.add_trace(go.Bar(
        x=comparison_df['Topic'],
        y=comparison_df['High-Cited Weight'],
        name='High-Cited Papers',
        marker_color='#3366CC'
    ))
    
    # Add low cited bars
    fig.add_trace(go.Bar(
        x=comparison_df['Topic'],
        y=comparison_df['Low-Cited Weight'],
        name='Low-Cited Papers',
        marker_color='#FF9900'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Topic Distribution: High vs. Low Cited Papers ({model_name})",
        xaxis_title="Topic",
        yaxis_title="Average Topic Weight",
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create difference chart
    fig = px.bar(
        comparison_df,
        x='Topic',
        y='Difference',
        title=f"Topic Weight Difference (High-Cited minus Low-Cited)",
        color='Difference',
        color_continuous_scale='RdBu',
        labels={'Difference': 'Weight Difference'}
    )
    
    fig.update_layout(
        xaxis_title="Topic",
        yaxis_title="Difference in Topic Weight"
    )
    
    # Add a horizontal line at zero
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=0,
        x1=len(comparison_df) - 0.5,
        y1=0,
        line=dict(
            color="black",
            width=1,
            dash="dash",
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    st.markdown("""
    **Interpretation:**
    
    - **Topic Correlation**: Shows how each topic correlates with citation count
      - Positive values indicate topics associated with higher citations
      - Negative values indicate topics associated with lower citations
      
    - **Topic Distribution Comparison**: Shows differences in topic weights between high and low cited papers
      - Positive differences (high > low) suggest topics more prevalent in highly-cited work
      - Negative differences (high < low) suggest topics more prevalent in less-cited work
    """)

with tab3:
    st.header("Paper Topic Explorer")
    
    # Select model type and topic count (same as previous tabs)
    if 'selected_model' not in locals() or 'selected_topic_count' not in locals():
        col1, col2 = st.columns(2)
        
        with col1:
            selected_model = st.selectbox(
                "Select topic model type",
                options=model_types,
                key="tab3_model"
            )
        
        with col2:
            selected_topic_count = st.selectbox(
                "Select number of topics",
                options=topic_counts,
                key="tab3_topics"
            )
        
    # Construct model name
    model_name = f"{selected_model}{selected_topic_count}"
    
    # Get topic columns
    topic_cols = []
    for i in range(1, selected_topic_count + 1):
        col_name = f"{model_name}_{i}"
        
        # Check different possible column naming patterns
        if col_name in topic_model_df.columns:
            topic_cols.append(col_name)
        elif f"{selected_model}{i}" in topic_model_df.columns:
            topic_cols.append(f"{selected_model}{i}")
        elif f"{selected_model}_{i}" in topic_model_df.columns:
            topic_cols.append(f"{selected_model}_{i}")
    
    if not topic_cols:
        st.error(f"Could not find topic columns for model {model_name}.")
        st.stop()
    
    # Get article IDs
    if 'ArticleID' in topic_model_df.columns:
        article_ids = topic_model_df['ArticleID'].tolist()
    else:
        article_ids = topic_model_df.index.tolist()
    
    # Select paper to explore
    selected_paper = st.selectbox(
        "Select a paper to explore",
        options=article_ids
    )
    
    if selected_paper:
        # Get paper data
        if 'ArticleID' in topic_model_df.columns:
            paper_data = topic_model_df[topic_model_df['ArticleID'] == selected_paper]
        else:
            paper_data = topic_model_df.loc[topic_model_df.index == selected_paper]
        
        if paper_data.empty:
            st.error(f"Paper {selected_paper} not found in topic model data.")
            st.stop()
        
        # Get citation count if available
        citation_count = "N/A"
        if 'CITATIONCOUNT' in paper_data.columns:
            citation_count = paper_data['CITATIONCOUNT'].iloc[0]
        elif 'Article Id' in citation_df.columns and selected_paper in citation_df['Article Id'].values:
            citation_count = citation_df[citation_df['Article Id'] == selected_paper]['Cited By'].iloc[0]
        
        # Display paper information
        st.subheader("Paper Information")
        
        # Get paper title if available
        paper_title = "N/A"
        if 'Article Id' in citation_df.columns and selected_paper in citation_df['Article Id'].values:
            paper_title = citation_df[citation_df['Article Id'] == selected_paper]['Title'].iloc[0]
        
        st.write(f"**Title:** {paper_title}")
        st.write(f"**ID:** {selected_paper}")
        st.write(f"**Citation Count:** {citation_count}")
        
        # Extract topic weights
        topic_weights = []
        for col in topic_cols:
            if col in paper_data.columns:
                topic_name = col.replace(f"{model_name}_", "Topic ").replace(f"{model_name}", "Topic ")
                topic_weights.append({
                    'Topic': topic_name,
                    'Weight': paper_data[col].iloc[0]
                })
        
        # Create dataframe
        topic_df = pd.DataFrame(topic_weights)
        
        # Sort by weight
        topic_df = topic_df.sort_values('Weight', ascending=False)
        
        # Display topic distribution
        st.subheader("Topic Distribution")
        
        # Create bar chart
        fig = px.bar(
            topic_df,
            x='Topic',
            y='Weight',
            title=f"Topic Distribution for Paper {selected_paper}",
            color='Weight',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            xaxis_title="Topic",
            yaxis_title="Weight",
            xaxis={'categoryorder':'total descending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Find similar papers based on topic distribution
        st.subheader("Similar Papers")
        
        # Number of similar papers to find
        num_similar = st.slider(
            "Number of similar papers to find",
            min_value=1,
            max_value=20,
            value=5
        )
        
        # Calculate similarity based on topic distribution
        similarities = []
        
        paper_vector = paper_data[topic_cols].iloc[0].values
        
        for idx, row in topic_model_df.iterrows():
            # Skip the selected paper
            paper_id = row.get('ArticleID', idx)
            if paper_id == selected_paper:
                continue
                
            # Get topic vector
            other_vector = row[topic_cols].values
            
            # Calculate cosine similarity
            dot_product = np.dot(paper_vector, other_vector)
            norm_paper = np.linalg.norm(paper_vector)
            norm_other = np.linalg.norm(other_vector)
            
            if norm_paper > 0 and norm_other > 0:
                similarity = dot_product / (norm_paper * norm_other)
            else:
                similarity = 0
                
            # Get citation count if available
            other_citations = "N/A"
            if 'CITATIONCOUNT' in row.index:
                other_citations = row['CITATIONCOUNT']
            elif 'Article Id' in citation_df.columns and paper_id in citation_df['Article Id'].values:
                other_citations = citation_df[citation_df['Article Id'] == paper_id]['Cited By'].iloc[0]
                
            # Get title if available
            other_title = "N/A"
            if 'Article Id' in citation_df.columns and paper_id in citation_df['Article Id'].values:
                other_title = citation_df[citation_df['Article Id'] == paper_id]['Title'].iloc[0]
                
            similarities.append({
                'Paper ID': paper_id,
                'Title': other_title,
                'Similarity': similarity,
                'Citations': other_citations
            })
        
        # Create dataframe
        similar_df = pd.DataFrame(similarities)
        
        # Sort by similarity
        similar_df = similar_df.sort_values('Similarity', ascending=False)
        
        # Display top similar papers
        st.write(f"Top {num_similar} similar papers based on topic distribution:")
        st.dataframe(similar_df.head(num_similar))
        
        # Compare citation counts
        st.subheader("Citation Comparison with Similar Papers")
        
        # Create comparison data
        comparison_data = []
        
        # Add selected paper
        comparison_data.append({
            'Paper': f"Selected: {selected_paper}",
            'Citations': float(citation_count) if citation_count != "N/A" else 0
        })
        
        # Add similar papers
        for i, (_, row) in enumerate(similar_df.head(num_similar).iterrows()):
            paper_id = row['Paper ID']
            citations = row['Citations']
            
            comparison_data.append({
                'Paper': f"Similar {i+1}: {paper_id}",
                'Citations': float(citations) if citations != "N/A" else 0
            })
        
        # Create dataframe
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create bar chart
        fig = px.bar(
            comparison_df,
            x='Paper',
            y='Citations',
            title="Citation Comparison with Similar Papers",
            color='Citations',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            xaxis_title="Paper",
            yaxis_title="Citation Count"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        st.markdown("""
        **Interpretation:**
        
        - **Topic Distribution**: Shows the weight of each topic in the selected paper
        - **Similar Papers**: Papers with the most similar topic distributions
        - **Citation Comparison**: Compare citation counts between the selected paper and similar papers
          - If similar papers have significantly higher citations, this may suggest potential for increased impact
          - Differences in citation counts among topically similar papers may indicate other factors affecting citation rates
        """)
