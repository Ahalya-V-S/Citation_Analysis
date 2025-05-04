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
    
    # Extract model prefixes and topic counts from column names
    model_prefixes = ['LDA', 'HDP', 'CTM', 'DLDA', 'DHDP', 'DCTM']
    
    for col in topic_model_df.columns:
        # Skip the ArticleID and CITATIONCOUNT columns
        if col in ['ArticleID', 'CITATIONCOUNT', 'Article Id']:
            continue
            
        # Check for model prefixes (LDA, HDP, CTM, DLDA, DHDP, DCTM)
        for prefix in model_prefixes:
            if prefix in col:
                # Extract model type
                if prefix not in model_types:
                    model_types.append(prefix)
                
                # Extract topic count (5 or 10)
                if '5' in col and 5 not in topic_counts:
                    topic_counts.append(5)
                if '10' in col and 10 not in topic_counts:
                    topic_counts.append(10)
    
    if not model_types or not topic_counts:
        st.error("No topic model data found in the uploaded dataset.")
        st.stop()
        
    # Sort model types and topic counts
    model_types.sort()
    topic_counts.sort()
    
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
    
    # Look for columns that contain the model name and topic count
    for col in topic_model_df.columns:
        # Skip non-topic columns
        if col in ['ArticleID', 'CITATIONCOUNT', 'Article Id']:
            continue
            
        # Check if the column contains the model name and topic count
        if selected_model in col and str(selected_topic_count) in col:
            # Make sure it's not just the model name column itself
            if len(col) > len(f"{selected_model}{selected_topic_count}"):
                topic_cols.append(col)
    
    # If we can't find columns by looking at the whole name, try more specific patterns
    if not topic_cols:
        for i in range(1, selected_topic_count + 1):
            # Try various naming patterns
            patterns = [
                f"{model_name}_{i}",     # LDA5_1
                f"{model_name}-{i}",      # LDA5-1 
                f"{model_name}{i}",       # LDA51
                f"{selected_model}{selected_topic_count}_{i}", # DLDA5_1
                f"{selected_model}{selected_topic_count}{i}"   # DLDA51
            ]
            
            for pattern in patterns:
                if pattern in topic_model_df.columns:
                    topic_cols.append(pattern)
                    break
    
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
    
    # Look for columns that contain the model name and topic count
    for col in topic_model_df.columns:
        # Skip non-topic columns
        if col in ['ArticleID', 'CITATIONCOUNT', 'Article Id']:
            continue
            
        # Check if the column contains the model name and topic count
        if selected_model in col and str(selected_topic_count) in col:
            # Make sure it's not just the model name column itself
            if len(col) > len(f"{selected_model}{selected_topic_count}"):
                topic_cols.append(col)
    
    # If we can't find columns by looking at the whole name, try more specific patterns
    if not topic_cols:
        for i in range(1, selected_topic_count + 1):
            # Try various naming patterns
            patterns = [
                f"{model_name}_{i}",     # LDA5_1
                f"{model_name}-{i}",      # LDA5-1 
                f"{model_name}{i}",       # LDA51
                f"{selected_model}{selected_topic_count}_{i}", # DLDA5_1
                f"{selected_model}{selected_topic_count}{i}"   # DLDA51
            ]
            
            for pattern in patterns:
                if pattern in topic_model_df.columns:
                    topic_cols.append(pattern)
                    break
    
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
    
    # Look for columns that contain the model name and topic count
    for col in topic_model_df.columns:
        # Skip non-topic columns
        if col in ['ArticleID', 'CITATIONCOUNT', 'Article Id']:
            continue
            
        # Check if the column contains the model name and topic count
        if selected_model in col and str(selected_topic_count) in col:
            # Make sure it's not just the model name column itself
            if len(col) > len(f"{selected_model}{selected_topic_count}"):
                topic_cols.append(col)
    
    # If we can't find columns by looking at the whole name, try more specific patterns
    if not topic_cols:
        for i in range(1, selected_topic_count + 1):
            # Try various naming patterns
            patterns = [
                f"{model_name}_{i}",     # LDA5_1
                f"{model_name}-{i}",      # LDA5-1 
                f"{model_name}{i}",       # LDA51
                f"{selected_model}{selected_topic_count}_{i}", # DLDA5_1
                f"{selected_model}{selected_topic_count}{i}"   # DLDA51
            ]
            
            for pattern in patterns:
                if pattern in topic_model_df.columns:
                    topic_cols.append(pattern)
                    break
    
    if not topic_cols:
        st.error(f"Could not find topic columns for model {model_name}.")
        st.stop()
    
    # Get paper IDs for selection
    if 'ArticleID' in topic_model_df.columns:
        paper_ids = topic_model_df['ArticleID'].tolist()
    else:
        paper_ids = list(range(len(topic_model_df)))
    
    # Add citation info if available
    paper_options = []
    for paper_id in paper_ids[:100]:  # Limit to first 100 for performance
        if 'CITATIONCOUNT' in topic_model_df.columns:
            citations = topic_model_df.loc[topic_model_df['ArticleID'] == paper_id, 'CITATIONCOUNT'].values[0]
            paper_options.append(f"Paper {paper_id} ({citations} citations)")
        else:
            paper_options.append(f"Paper {paper_id}")
    
    # Select a paper
    selected_paper = st.selectbox(
        "Select a paper to analyze:",
        options=paper_options
    )
    
    # Get paper ID from selection
    selected_paper_id = int(selected_paper.split(" ")[1].split(" ")[0])
    
    # Get paper data
    paper_data = topic_model_df[topic_model_df['ArticleID'] == selected_paper_id]
    
    if len(paper_data) == 0:
        st.error(f"Paper ID {selected_paper_id} not found in dataset.")
        st.stop()
    
    # Get topic distribution for this paper
    paper_topic_dist = paper_data[topic_cols].values[0]
    
    # Create dataframe for plotting
    paper_topic_df = pd.DataFrame({
        'Topic': [f"Topic {i+1}" for i in range(len(topic_cols))],
        'Weight': paper_topic_dist
    })
    
    # Sort by weight
    paper_topic_df = paper_topic_df.sort_values('Weight', ascending=False)
    
    # Plot topic distribution
    fig = px.bar(
        paper_topic_df,
        x='Topic',
        y='Weight',
        title=f"Topic Distribution for Paper {selected_paper_id}",
        color='Weight',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis_title="Topic",
        yaxis_title="Weight",
        xaxis={'categoryorder':'total descending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate topic diversity for this paper
    topic_values = paper_topic_dist
    topic_values = np.array(topic_values) / sum(topic_values)
    
    # Calculate entropy
    entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in topic_values)
    
    # Normalize by maximum possible entropy
    max_entropy = np.log2(len(topic_values))
    if max_entropy > 0:
        normalized_entropy = entropy / max_entropy
    else:
        normalized_entropy = 0
    
    # Display paper information
    st.subheader("Paper Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Paper ID", selected_paper_id)
        
        if 'CITATIONCOUNT' in topic_model_df.columns:
            citations = paper_data['CITATIONCOUNT'].values[0]
            st.metric("Citations", citations)
    
    with col2:
        st.metric("Topic Diversity", f"{normalized_entropy:.2f}")
        
        # Dominant topic
        dominant_topic_idx = np.argmax(paper_topic_dist)
        dominant_topic = f"Topic {dominant_topic_idx + 1}"
        dominant_weight = paper_topic_dist[dominant_topic_idx]
        
        st.metric("Dominant Topic", f"{dominant_topic} ({dominant_weight:.3f})")
    
    # Compare with similar papers
    st.subheader("Similar Papers")
    
    # Define similarity function (cosine similarity)
    def cosine_similarity(v1, v2):
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        return dot_product / (norm_v1 * norm_v2) if norm_v1 > 0 and norm_v2 > 0 else 0
    
    # Calculate similarity to all other papers
    similarities = []
    for idx, row in topic_model_df.iterrows():
        paper_id = row.get('ArticleID', idx)
        
        if paper_id != selected_paper_id:
            sim = cosine_similarity(paper_topic_dist, row[topic_cols].values)
            
            similarity_info = {
                'Paper ID': paper_id,
                'Similarity': sim
            }
            
            if 'CITATIONCOUNT' in topic_model_df.columns:
                similarity_info['Citations'] = row['CITATIONCOUNT']
            
            similarities.append(similarity_info)
    
    # Create dataframe and sort by similarity
    sim_df = pd.DataFrame(similarities).sort_values('Similarity', ascending=False)
    
    # Display top similar papers
    st.write("Papers with similar topic distributions:")
    st.dataframe(sim_df.head(10))
    
    # Plot similarity vs citations for similar papers
    if 'Citations' in sim_df.columns:
        # Get top similar papers
        top_similar = sim_df.head(50)
        
        # Create scatter plot
        fig = px.scatter(
            top_similar,
            x='Similarity',
            y='Citations',
            hover_name='Paper ID',
            title="Citations vs. Topic Similarity (50 most similar papers)",
            labels={
                'Similarity': 'Topic Distribution Similarity',
                'Citations': 'Citation Count'
            }
        )
        
        # Add a point for the selected paper
        selected_citations = paper_data['CITATIONCOUNT'].values[0] if 'CITATIONCOUNT' in paper_data.columns else 0
        
        fig.add_trace(go.Scatter(
            x=[1.0],  # Perfect similarity with itself
            y=[selected_citations],
            mode='markers',
            marker=dict(
                color='red',
                size=12,
                symbol='star'
            ),
            name=f"Selected Paper ({selected_paper_id})"
        ))
        
        # Calculate trend line
        if len(top_similar) > 1:
            x = top_similar['Similarity']
            y = top_similar['Citations']
            
            # Add trend line
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                line=dict(color='rgba(0,0,0,0.3)'),
                name='Trend'
            ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate correlation between similarity and citations
        similarity_citation_corr = top_similar['Similarity'].corr(top_similar['Citations'])
        
        st.write(f"Correlation between topic similarity and citations: {similarity_citation_corr:.3f}")
        
        if similarity_citation_corr > 0.2:
            st.success("Positive correlation: Papers with similar topic distributions tend to have more citations.")
        elif similarity_citation_corr < -0.2:
            st.warning("Negative correlation: Papers with similar topic distributions tend to have fewer citations.")
        else:
            st.info("No strong correlation between topic similarity and citation counts.")
    
    # Paper comparison
    st.subheader("Compare with Another Paper")
    
    # Select another paper
    comparison_paper = st.selectbox(
        "Select a paper to compare with:",
        options=paper_options,
        key="compare_paper"
    )
    
    # Get comparison paper ID
    comparison_paper_id = int(comparison_paper.split(" ")[1].split(" ")[0])
    
    # Make sure we're not comparing with the same paper
    if comparison_paper_id == selected_paper_id:
        st.warning("Please select a different paper for comparison.")
    else:
        # Get comparison paper data
        comparison_data = topic_model_df[topic_model_df['ArticleID'] == comparison_paper_id]
        
        if len(comparison_data) == 0:
            st.error(f"Paper ID {comparison_paper_id} not found in dataset.")
        else:
            # Get topic distribution for comparison paper
            comparison_topic_dist = comparison_data[topic_cols].values[0]
            
            # Calculate similarity between papers
            similarity = cosine_similarity(paper_topic_dist, comparison_topic_dist)
            
            st.write(f"Topic distribution similarity: {similarity:.3f}")
            
            # Create comparison dataframe
            comparison_df = pd.DataFrame({
                'Topic': [f"Topic {i+1}" for i in range(len(topic_cols))],
                f"Paper {selected_paper_id}": paper_topic_dist,
                f"Paper {comparison_paper_id}": comparison_topic_dist,
                'Difference': comparison_topic_dist - paper_topic_dist
            })
            
            # Sort by absolute difference
            comparison_df['Abs_Difference'] = abs(comparison_df['Difference'])
            comparison_df = comparison_df.sort_values('Abs_Difference', ascending=False)
            comparison_df = comparison_df.drop('Abs_Difference', axis=1)
            
            # Create grouped bar chart
            fig = go.Figure()
            
            # Add selected paper bars
            fig.add_trace(go.Bar(
                x=comparison_df['Topic'],
                y=comparison_df[f"Paper {selected_paper_id}"],
                name=f"Paper {selected_paper_id}",
                marker_color='#3366CC'
            ))
            
            # Add comparison paper bars
            fig.add_trace(go.Bar(
                x=comparison_df['Topic'],
                y=comparison_df[f"Paper {comparison_paper_id}"],
                name=f"Paper {comparison_paper_id}",
                marker_color='#FF9900'
            ))
            
            # Update layout
            fig.update_layout(
                title=f"Topic Distribution Comparison",
                xaxis_title="Topic",
                yaxis_title="Weight",
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
            
            # Comparison table
            st.write("Topic weight comparison:")
            st.dataframe(comparison_df)
            
            # Citation comparison if available
            if 'CITATIONCOUNT' in topic_model_df.columns:
                selected_citations = paper_data['CITATIONCOUNT'].values[0]
                comparison_citations = comparison_data['CITATIONCOUNT'].values[0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(f"Paper {selected_paper_id} Citations", selected_citations)
                
                with col2:
                    st.metric(f"Paper {comparison_paper_id} Citations", comparison_citations, 
                             delta=comparison_citations - selected_citations)
                
                # Suggest possible reasons for citation difference
                if abs(comparison_citations - selected_citations) > 5:
                    st.subheader("Possible Reasons for Citation Difference")
                    
                    # Get top differences in topics
                    top_diffs = comparison_df.head(3)
                    
                    if comparison_citations > selected_citations:
                        st.write(f"Paper {comparison_paper_id} has more citations than Paper {selected_paper_id}. Possible reasons:")
                        
                        for _, row in top_diffs.iterrows():
                            if row['Difference'] > 0.05:
                                st.write(f"- Higher weight in {row['Topic']}: {row['Difference']:.3f}")
                            elif row['Difference'] < -0.05:
                                st.write(f"- Lower weight in {row['Topic']}: {row['Difference']:.3f}")
                    else:
                        st.write(f"Paper {selected_paper_id} has more citations than Paper {comparison_paper_id}. Possible reasons:")
                        
                        for _, row in top_diffs.iterrows():
                            if row['Difference'] < -0.05:
                                st.write(f"- Higher weight in {row['Topic']}: {-row['Difference']:.3f}")
                            elif row['Difference'] > 0.05:
                                st.write(f"- Lower weight in {row['Topic']}: {-row['Difference']:.3f}")