import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Low Citation Analysis",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Low Citation Analysis")
st.write("""
This tool performs in-depth analysis of factors contributing to low citation rates, 
focusing on topic distribution patterns and thematic positioning of papers.
""")

# Check if required data is available
if 'citation_data' not in st.session_state or 'topic_model_data' not in st.session_state:
    st.error("Missing required datasets. Please upload both citation and topic model data in the main page.")
    st.stop()

# Get the data
citation_df = st.session_state['citation_data']
topic_model_df = st.session_state['topic_model_data']

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "Topic-Based Analysis", 
    "Citation Threshold Analysis",
    "Comparative Case Studies"
])

# ------ Tab 1: Topic-Based Analysis ------
with tab1:
    st.header("🔍 Topic-Based Analysis")
    
    # Identify available topic models
    model_prefixes = ['LDA', 'HDP', 'CTM', 'DLDA', 'DHDP', 'DCTM']
    topic_numbers = [5, 10]
    
    model_columns = {}
    for prefix in model_prefixes:
        for num in topic_numbers:
            column_pattern = f"{prefix}{num}"
            # Get all columns that match the pattern
            matching_cols = [col for col in topic_model_df.columns if column_pattern in col and col != column_pattern]
            if matching_cols:
                model_columns[column_pattern] = matching_cols
    
    if not model_columns:
        st.error("No topic model columns found in the uploaded dataset.")
        st.stop()
    
    # Model selection
    col1, col2 = st.columns(2)
    with col1:
        model_key = st.selectbox(
            "Select topic model",
            options=list(model_columns.keys()),
            help="Choose the topic model to analyze"
        )
    
    with col2:
        # Check if CITATIONCOUNT exists in the data
        if 'CITATIONCOUNT' not in topic_model_df.columns:
            # Try to merge with citation data
            if 'ArticleID' in topic_model_df.columns and 'Article Id' in citation_df.columns:
                # Merge to get citation counts
                merged_df = pd.merge(
                    topic_model_df,
                    citation_df[['Article Id', 'Cited By']],
                    left_on='ArticleID',
                    right_on='Article Id',
                    how='inner'
                )
                if 'Cited By' in merged_df.columns:
                    topic_model_df['CITATIONCOUNT'] = merged_df['Cited By']
                    st.success("Successfully merged citation counts")
                else:
                    st.error("Failed to merge citation counts")
            else:
                st.error("No citation count information available")
                st.stop()
    
    # Get the selected topic columns
    topic_cols = model_columns[model_key]
    
    # Correlation analysis between topics and citation counts
    st.subheader("Topic-Citation Correlation Analysis")
    
    # Calculate correlations
    correlations = []
    for col in topic_cols:
        pearson_corr, p_value = pearsonr(topic_model_df[col], topic_model_df['CITATIONCOUNT'])
        spearman_corr, sp_p_value = spearmanr(topic_model_df[col], topic_model_df['CITATIONCOUNT'])
        
        topic_num = col.replace(model_key, "").replace("_", "")
        if not topic_num:  # Handle case where column name doesn't follow expected pattern
            topic_num = col.split("_")[-1] if "_" in col else "Unknown"
            
        correlations.append({
            'Topic': f"Topic {topic_num}",
            'Pearson Correlation': pearson_corr,
            'Pearson p-value': p_value,
            'Spearman Correlation': spearman_corr,
            'Spearman p-value': sp_p_value,
            'Abs Correlation': abs(pearson_corr)
        })
    
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('Abs Correlation', ascending=False)
    
    # Show correlation table
    with st.expander("View complete correlation data"):
        st.dataframe(corr_df.drop('Abs Correlation', axis=1))
    
    # Plot correlations
    fig = px.bar(
        corr_df,
        x='Topic',
        y='Pearson Correlation',
        color='Pearson Correlation',
        color_continuous_scale='RdBu_r',
        title=f"Topic-Citation Correlation Analysis ({model_key})",
        labels={'Pearson Correlation': 'Correlation Coefficient'}
    )
    
    # Add zero line
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=0,
        x1=len(corr_df)-0.5,
        y1=0,
        line=dict(color="black", width=1, dash="dash")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Identify problematic topics (negatively correlated with citations)
    negative_corr = corr_df[corr_df['Pearson Correlation'] < -0.1]
    if not negative_corr.empty:
        st.subheader("Potentially Problematic Topics")
        st.write("These topics show negative correlation with citation counts:")
        
        for _, row in negative_corr.iterrows():
            significance = "statistically significant" if row['Pearson p-value'] < 0.05 else "not statistically significant"
            st.markdown(f"- **{row['Topic']}**: Correlation = {row['Pearson Correlation']:.3f} ({significance}, p={row['Pearson p-value']:.3f})")
    
    # Identify beneficial topics (positively correlated with citations)
    positive_corr = corr_df[corr_df['Pearson Correlation'] > 0.1]
    if not positive_corr.empty:
        st.subheader("Beneficial Topics")
        st.write("These topics show positive correlation with citation counts:")
        
        for _, row in positive_corr.iterrows():
            significance = "statistically significant" if row['Pearson p-value'] < 0.05 else "not statistically significant"
            st.markdown(f"- **{row['Topic']}**: Correlation = {row['Pearson Correlation']:.3f} ({significance}, p={row['Pearson p-value']:.3f})")
    
    # Topic distribution in low vs. high cited papers
    st.subheader("Topic Distribution in Low vs. High Cited Papers")
    
    # Define citation threshold
    threshold_option = st.radio(
        "Define citation threshold using:",
        ["Percentile", "Absolute Value"],
        horizontal=True
    )
    
    if threshold_option == "Percentile":
        percentile = st.slider("Citation percentile threshold", 10, 50, 25, step=5)
        citation_threshold = np.percentile(topic_model_df['CITATIONCOUNT'], percentile)
    else:
        max_citation = int(topic_model_df['CITATIONCOUNT'].max())
        citation_threshold = st.slider("Maximum citation count for low-cited papers", 0, max_citation, min(10, max_citation))
    
    # Create low and high citation groups
    low_cited = topic_model_df[topic_model_df['CITATIONCOUNT'] <= citation_threshold]
    high_cited = topic_model_df[topic_model_df['CITATIONCOUNT'] > citation_threshold]
    
    # Display threshold and group sizes
    st.write(f"Low citation threshold: ≤ {citation_threshold} citations")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Low-cited Papers", f"{len(low_cited)} ({len(low_cited)/len(topic_model_df)*100:.1f}%)")
    with col2:
        st.metric("High-cited Papers", f"{len(high_cited)} ({len(high_cited)/len(topic_model_df)*100:.1f}%)")
    
    # Calculate average topic weights for each group
    low_topic_weights = low_cited[topic_cols].mean()
    high_topic_weights = high_cited[topic_cols].mean()
    
    # Create comparison dataframe
    topic_names = [f"Topic {i+1}" for i in range(len(topic_cols))]
    comparison_df = pd.DataFrame({
        'Topic': topic_names,
        'Low-cited Papers': low_topic_weights.values,
        'High-cited Papers': high_topic_weights.values
    })
    
    # Calculate difference
    comparison_df['Difference'] = comparison_df['High-cited Papers'] - comparison_df['Low-cited Papers']
    comparison_df['Abs Difference'] = abs(comparison_df['Difference'])
    
    # Sort by absolute difference
    comparison_df = comparison_df.sort_values('Abs Difference', ascending=False)
    
    # Plot comparison
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=comparison_df['Topic'],
        y=comparison_df['Low-cited Papers'],
        name='Low-cited Papers',
        marker_color='#FF6B6B'
    ))
    fig.add_trace(go.Bar(
        x=comparison_df['Topic'],
        y=comparison_df['High-cited Papers'],
        name='High-cited Papers',
        marker_color='#4ECDC4'
    ))
    
    fig.update_layout(
        barmode='group',
        title="Topic Distribution: Low vs High Cited Papers",
        xaxis_title="Topics",
        yaxis_title="Average Weight"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Plot differences
    fig = px.bar(
        comparison_df,
        x='Topic',
        y='Difference',
        color='Difference',
        color_continuous_scale='RdBu',
        title="Topic Weight Difference (High-cited minus Low-cited)",
    )
    
    # Add zero line
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=0,
        x1=len(comparison_df)-0.5,
        y1=0,
        line=dict(color="black", width=1, dash="dash")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Topic impact analysis
    st.subheader("Feature Importance Analysis")
    st.write("Analyzing which topics have the greatest impact on citation counts.")
    
    if st.button("Run Feature Importance Analysis"):
        with st.spinner("Training model and calculating feature importance..."):
            # Prepare data
            X = topic_model_df[topic_cols]
            y = topic_model_df['CITATIONCOUNT']
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train random forest model
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_scaled, y)
            
            # Get feature importances
            importances = rf_model.feature_importances_
            
            # Create dataframe
            importance_df = pd.DataFrame({
                'Topic': topic_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Plot importance
            fig = px.bar(
                importance_df,
                x='Topic',
                y='Importance',
                color='Importance',
                color_continuous_scale='viridis',
                title="Topic Importance for Citation Prediction"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation of results
            st.subheader("Key Findings: Topic Impact on Citations")
            
            # List top influential topics
            top_topics = importance_df.head(3)['Topic'].tolist()
            bottom_topics = importance_df.tail(3)['Topic'].tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Most influential topics:**")
                for topic in top_topics:
                    corr_value = corr_df.loc[corr_df['Topic'] == topic, 'Pearson Correlation'].values[0]
                    direction = "positive" if corr_value > 0 else "negative"
                    st.write(f"- {topic} ({direction} impact)")
            
            with col2:
                st.markdown("**Least influential topics:**")
                for topic in bottom_topics:
                    st.write(f"- {topic}")

# ------ Tab 2: Citation Threshold Analysis ------
with tab2:
    st.header("🔢 Citation Threshold Analysis")
    st.write("Analyze how papers cluster based on citation counts and topic distributions.")
    
    # Define citation thresholds
    threshold_method = st.radio(
        "Threshold determination method:",
        ["Natural Clustering", "Custom Ranges", "Percentiles"],
        horizontal=True
    )
    
    if threshold_method == "Natural Clustering":
        # Perform k-means clustering on citation counts
        n_clusters = st.slider("Number of citation clusters", 2, 8, 4)
        
        # Get citation counts
        citation_values = topic_model_df['CITATIONCOUNT'].values.reshape(-1, 1)
        
        # Apply log transformation for better clustering
        log_citations = np.log1p(citation_values)
        
        # Run k-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(log_citations)
        
        # Add cluster information to dataframe
        topic_model_df['citation_cluster'] = clusters
        
        # Get cluster centers and sort them
        centers = np.expm1(kmeans.cluster_centers_).flatten()
        sorted_clusters = np.argsort(centers)
        
        # Create lookup table
        cluster_names = {sorted_clusters[i]: f"Cluster {i+1}" for i in range(n_clusters)}
        cluster_centers = {sorted_clusters[i]: centers[sorted_clusters[i]] for i in range(n_clusters)}
        
        # Display cluster information
        st.subheader("Citation Clusters")
        
        cluster_info = []
        for i in range(n_clusters):
            cluster_idx = sorted_clusters[i]
            count = (topic_model_df['citation_cluster'] == cluster_idx).sum()
            percentage = count / len(topic_model_df) * 100
            
            cluster_info.append({
                'Cluster': f"Cluster {i+1}",
                'Center': cluster_centers[cluster_idx],
                'Papers': count,
                'Percentage': percentage
            })
        
        cluster_df = pd.DataFrame(cluster_info)
        st.dataframe(cluster_df)
        
        # Plot distribution
        fig = px.histogram(
            topic_model_df,
            x='CITATIONCOUNT',
            color='citation_cluster',
            color_discrete_sequence=px.colors.qualitative.G10,
            nbins=50,
            opacity=0.7,
            title="Citation Distribution by Cluster",
            labels={'citation_cluster': 'Cluster'}
        )
        
        # Add vertical lines for cluster centers
        for i, center in enumerate(centers):
            fig.add_vline(
                x=center,
                line_dash="dash",
                line_color="black",
                annotation_text=f"Cluster {sorted_clusters.tolist().index(i)+1} center",
                annotation_position="top right"
            )
        
        fig.update_layout(
            xaxis_title="Citation Count",
            yaxis_title="Number of Papers",
            bargap=0.1
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Analyze topic distribution across clusters
        st.subheader("Topic Distribution Across Citation Clusters")
        
        # Calculate average topic weights for each cluster
        cluster_topic_weights = []
        for i in range(n_clusters):
            cluster_idx = sorted_clusters[i]
            cluster_papers = topic_model_df[topic_model_df['citation_cluster'] == cluster_idx]
            
            avg_weights = cluster_papers[topic_cols].mean()
            avg_weights_dict = {f"Topic {j+1}": avg_weights[topic_cols[j]] for j in range(len(topic_cols))}
            avg_weights_dict['Cluster'] = f"Cluster {i+1}"
            avg_weights_dict['Avg Citations'] = cluster_papers['CITATIONCOUNT'].mean()
            
            cluster_topic_weights.append(avg_weights_dict)
        
        cluster_topic_df = pd.DataFrame(cluster_topic_weights)
        
        # Create heatmap
        topic_matrix = cluster_topic_df[[f"Topic {i+1}" for i in range(len(topic_cols))]].values
        
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.heatmap(
            topic_matrix,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            yticklabels=cluster_topic_df['Cluster'],
            xticklabels=[f"Topic {i+1}" for i in range(len(topic_cols))],
            ax=ax
        )
        ax.set_title("Topic Distribution by Citation Cluster")
        ax.set_ylabel("Citation Cluster")
        ax.set_xlabel("Topic")
        
        st.pyplot(fig)
        
        # Create radar chart for comparison
        st.subheader("Topic Profile by Citation Cluster")
        
        # Prepare data for radar chart
        categories = [f"Topic {i+1}" for i in range(len(topic_cols))]
        
        fig = go.Figure()
        
        # Add traces for each cluster
        for i in range(n_clusters):
            values = cluster_topic_df[[f"Topic {i+1}" for i in range(len(topic_cols))]].iloc[i].values.tolist()
            # Close the loop
            values.append(values[0])
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=f"Cluster {i+1} (Avg: {cluster_topic_df['Avg Citations'].iloc[i]:.1f} citations)"
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(topic_matrix.max(), 0.5)]
                )),
            showlegend=True,
            title="Topic Profile by Citation Cluster"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    elif threshold_method == "Custom Ranges":
        # Define custom citation ranges
        range_input = st.text_area(
            "Enter citation ranges (one range per line, e.g., '0-10')",
            value="0-5\n6-25\n26-100\n100+"
        )
        
        try:
            # Parse ranges
            ranges = []
            for line in range_input.strip().split('\n'):
                if '-' in line:
                    parts = line.split('-')
                    if len(parts) == 2:
                        start = int(parts[0].strip())
                        if parts[1].strip().endswith('+'):
                            end = float('inf')
                            ranges.append((start, end))
                        else:
                            end = int(parts[1].strip())
                            ranges.append((start, end))
            
            # Sort ranges
            ranges.sort(key=lambda x: x[0])
            
            # Check for range overlap
            for i in range(len(ranges)-1):
                if ranges[i][1] >= ranges[i+1][0]:
                    st.warning(f"Ranges overlap: {ranges[i]} and {ranges[i+1]}")
            
            # Create range categories
            range_categories = []
            for i, (start, end) in enumerate(ranges):
                if end == float('inf'):
                    category = f"{start}+"
                else:
                    category = f"{start}-{end}"
                range_categories.append(category)
            
            # Assign papers to categories
            topic_model_df['citation_category'] = None
            for i, (start, end) in enumerate(ranges):
                mask = (topic_model_df['CITATIONCOUNT'] >= start) & (topic_model_df['CITATIONCOUNT'] <= end)
                topic_model_df.loc[mask, 'citation_category'] = range_categories[i]
            
            # Display category information
            st.subheader("Citation Categories")
            
            category_info = []
            for category in range_categories:
                count = (topic_model_df['citation_category'] == category).sum()
                percentage = count / len(topic_model_df) * 100
                avg_citations = topic_model_df[topic_model_df['citation_category'] == category]['CITATIONCOUNT'].mean()
                
                category_info.append({
                    'Category': category,
                    'Papers': count,
                    'Percentage': percentage,
                    'Avg Citations': avg_citations
                })
            
            category_df = pd.DataFrame(category_info)
            st.dataframe(category_df)
            
            # Plot distribution
            fig = px.histogram(
                topic_model_df,
                x='CITATIONCOUNT',
                color='citation_category',
                color_discrete_sequence=px.colors.qualitative.G10,
                nbins=50,
                opacity=0.7,
                title="Citation Distribution by Category",
                labels={'citation_category': 'Category'}
            )
            
            fig.update_layout(
                xaxis_title="Citation Count",
                yaxis_title="Number of Papers",
                bargap=0.1
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Analyze topic distribution across categories
            st.subheader("Topic Distribution Across Citation Categories")
            
            # Calculate average topic weights for each category
            category_topic_weights = []
            for category in range_categories:
                category_papers = topic_model_df[topic_model_df['citation_category'] == category]
                
                avg_weights = category_papers[topic_cols].mean()
                avg_weights_dict = {f"Topic {j+1}": avg_weights[topic_cols[j]] for j in range(len(topic_cols))}
                avg_weights_dict['Category'] = category
                avg_weights_dict['Avg Citations'] = category_papers['CITATIONCOUNT'].mean()
                
                category_topic_weights.append(avg_weights_dict)
            
            category_topic_df = pd.DataFrame(category_topic_weights)
            
            # Create heatmap
            topic_matrix = category_topic_df[[f"Topic {i+1}" for i in range(len(topic_cols))]].values
            
            fig, ax = plt.subplots(figsize=(12, 4))
            sns.heatmap(
                topic_matrix,
                annot=True,
                fmt=".2f",
                cmap="viridis",
                yticklabels=category_topic_df['Category'],
                xticklabels=[f"Topic {i+1}" for i in range(len(topic_cols))],
                ax=ax
            )
            ax.set_title("Topic Distribution by Citation Category")
            ax.set_ylabel("Citation Category")
            ax.set_xlabel("Topic")
            
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error parsing ranges: {str(e)}")
    
    else:  # Percentiles
        # Define percentile thresholds
        percentiles = st.multiselect(
            "Select percentile thresholds",
            options=[10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90],
            default=[25, 50, 75]
        )
        
        if not percentiles:
            st.warning("Please select at least one percentile threshold")
        else:
            # Sort percentiles
            percentiles.sort()
            
            # Calculate threshold values
            thresholds = [np.percentile(topic_model_df['CITATIONCOUNT'], p) for p in percentiles]
            
            # Create category boundaries
            boundaries = [0] + thresholds + [float('inf')]
            
            # Create category names
            categories = []
            for i in range(len(boundaries) - 1):
                if i == 0:
                    categories.append(f"Bottom {percentiles[0]}%")
                elif i == len(boundaries) - 2:
                    categories.append(f"Top {100 - percentiles[-1]}%")
                else:
                    categories.append(f"{percentiles[i-1]}-{percentiles[i]}%")
            
            # Assign papers to categories
            topic_model_df['percentile_category'] = None
            for i in range(len(categories)):
                mask = (topic_model_df['CITATIONCOUNT'] >= boundaries[i]) & (topic_model_df['CITATIONCOUNT'] < boundaries[i+1])
                topic_model_df.loc[mask, 'percentile_category'] = categories[i]
            
            # Display category information
            st.subheader("Citation Percentile Categories")
            
            category_info = []
            for category in categories:
                count = (topic_model_df['percentile_category'] == category).sum()
                percentage = count / len(topic_model_df) * 100
                avg_citations = topic_model_df[topic_model_df['percentile_category'] == category]['CITATIONCOUNT'].mean()
                
                category_info.append({
                    'Category': category,
                    'Papers': count,
                    'Percentage': percentage,
                    'Avg Citations': avg_citations
                })
            
            category_df = pd.DataFrame(category_info)
            st.dataframe(category_df)
            
            # Plot distribution
            fig = px.histogram(
                topic_model_df,
                x='CITATIONCOUNT',
                color='percentile_category',
                color_discrete_sequence=px.colors.qualitative.G10,
                nbins=50,
                opacity=0.7,
                title="Citation Distribution by Percentile Category",
                labels={'percentile_category': 'Category'}
            )
            
            # Add vertical lines for thresholds
            for i, threshold in enumerate(thresholds):
                fig.add_vline(
                    x=threshold,
                    line_dash="dash",
                    line_color="black",
                    annotation_text=f"{percentiles[i]}th percentile",
                    annotation_position="top right"
                )
            
            fig.update_layout(
                xaxis_title="Citation Count",
                yaxis_title="Number of Papers",
                bargap=0.1
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Analyze lowest percentile in detail
            lowest_category = categories[0]
            st.subheader(f"Analysis of {lowest_category} Papers")
            
            low_cited_papers = topic_model_df[topic_model_df['percentile_category'] == lowest_category]
            
            # Topic distribution in low-cited papers
            topic_weights = low_cited_papers[topic_cols].mean()
            topic_weights_df = pd.DataFrame({
                'Topic': [f"Topic {i+1}" for i in range(len(topic_cols))],
                'Weight': topic_weights.values
            }).sort_values('Weight', ascending=False)
            
            fig = px.bar(
                topic_weights_df,
                x='Topic',
                y='Weight',
                color='Weight',
                color_continuous_scale='viridis',
                title=f"Topic Distribution in {lowest_category} Papers"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Compare with highest percentile
            highest_category = categories[-1]
            high_cited_papers = topic_model_df[topic_model_df['percentile_category'] == highest_category]
            
            high_topic_weights = high_cited_papers[topic_cols].mean()
            
            comparison_df = pd.DataFrame({
                'Topic': [f"Topic {i+1}" for i in range(len(topic_cols))],
                f"{lowest_category}": topic_weights.values,
                f"{highest_category}": high_topic_weights.values
            })
            
            comparison_df['Difference'] = comparison_df[f"{highest_category}"] - comparison_df[f"{lowest_category}"]
            comparison_df['Abs Difference'] = abs(comparison_df['Difference'])
            comparison_df = comparison_df.sort_values('Abs Difference', ascending=False)
            
            # Plot comparison
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=comparison_df['Topic'],
                y=comparison_df[f"{lowest_category}"],
                name=f"{lowest_category}",
                marker_color='#FF6B6B'
            ))
            
            fig.add_trace(go.Bar(
                x=comparison_df['Topic'],
                y=comparison_df[f"{highest_category}"],
                name=f"{highest_category}",
                marker_color='#4ECDC4'
            ))
            
            fig.update_layout(
                barmode='group',
                title=f"Topic Comparison: {lowest_category} vs {highest_category}",
                xaxis_title="Topic",
                yaxis_title="Average Weight"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Common factors analysis
    st.subheader("Characteristics of Low-Cited Papers")
    
    # Define low citation threshold based on previous tab choices
    if 'citation_cluster' in topic_model_df.columns:
        lowest_cluster = sorted_clusters[0]
        low_cited_papers = topic_model_df[topic_model_df['citation_cluster'] == lowest_cluster]
        low_label = f"Cluster {sorted_clusters.tolist().index(lowest_cluster)+1}"
    elif 'citation_category' in topic_model_df.columns:
        low_cited_papers = topic_model_df[topic_model_df['citation_category'] == range_categories[0]]
        low_label = range_categories[0]
    elif 'percentile_category' in topic_model_df.columns:
        low_cited_papers = topic_model_df[topic_model_df['percentile_category'] == categories[0]]
        low_label = categories[0]
    else:
        # Default to bottom 25% percentile
        threshold = np.percentile(topic_model_df['CITATIONCOUNT'], 25)
        low_cited_papers = topic_model_df[topic_model_df['CITATIONCOUNT'] <= threshold]
        low_label = f"Bottom 25% (≤ {threshold} citations)"
    
    st.write(f"Analyzing {len(low_cited_papers)} papers in {low_label}")
    
    # Topic pattern analysis
    low_topic_weights = low_cited_papers[topic_cols].mean()
    
    # Create sorted topic weights
    topic_weights_df = pd.DataFrame({
        'Topic': [f"Topic {i+1}" for i in range(len(topic_cols))],
        'Weight': low_topic_weights.values
    }).sort_values('Weight', ascending=False)
    
    # Characteristic topics
    st.write("**Dominant Topics in Low-Cited Papers:**")
    
    for i, row in topic_weights_df.head(3).iterrows():
        st.markdown(f"- **{row['Topic']}**: Weight = {row['Weight']:.3f}")
    
    # Find topic concentration metrics
    topic_concentrations = []
    for _, row in low_cited_papers.iterrows():
        # Calculate Gini coefficient
        values = np.array(row[topic_cols].values)
        values = values / values.sum() if values.sum() > 0 else values
        values.sort()
        
        n = len(values)
        index = np.arange(1, n+1)
        gini = (np.sum((2 * index - n - 1) * values)) / (n * np.sum(values))
        
        topic_concentrations.append({
            'ArticleID': row.get('ArticleID', 'Unknown'),
            'Citations': row.get('CITATIONCOUNT', 0),
            'Max Topic Weight': np.max(values),
            'Topic Gini': gini
        })
    
    concentration_df = pd.DataFrame(topic_concentrations)
    
    # Plot concentration metrics
    fig = px.scatter(
        concentration_df,
        x='Max Topic Weight',
        y='Topic Gini',
        color='Citations',
        hover_name='ArticleID',
        color_continuous_scale='viridis',
        title="Topic Concentration in Low-Cited Papers",
        labels={
            'Max Topic Weight': 'Maximum Topic Weight',
            'Topic Gini': 'Topic Concentration (Gini)',
            'Citations': 'Citation Count'
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Average topic concentration metrics
    avg_max_weight = concentration_df['Max Topic Weight'].mean()
    avg_gini = concentration_df['Topic Gini'].mean()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Avg Max Topic Weight", f"{avg_max_weight:.3f}")
    with col2:
        st.metric("Avg Topic Concentration (Gini)", f"{avg_gini:.3f}")
    
    # Recommendations based on analysis
    st.subheader("🔍 Potential Reasons for Low Citations")
    
    reasons = []
    
    # Check topic concentration
    if avg_max_weight > 0.4:
        reasons.append("**High topic concentration**: Papers focus too narrowly on specific topics, limiting broader appeal")
    
    # Check problem topics
    if not negative_corr.empty:
        problem_topics = negative_corr['Topic'].tolist()
        topic_overlaps = set(problem_topics) & set(topic_weights_df.head(3)['Topic'].tolist())
        if topic_overlaps:
            reasons.append(f"**Focus on low-impact topics**: Dominant topics ({', '.join(topic_overlaps)}) have negative correlation with citations")
    
    # Check missing beneficial topics
    if not positive_corr.empty:
        beneficial_topics = positive_corr['Topic'].tolist()
        missing_beneficial = set(beneficial_topics) - set(topic_weights_df.head(3)['Topic'].tolist())
        if missing_beneficial:
            reasons.append(f"**Lack of high-impact topics**: Insufficient focus on topics with positive citation correlation ({', '.join(list(missing_beneficial)[:3])})")
    
    # Add general reasons if nothing specific found
    if not reasons:
        reasons = [
            "**Topic timeliness**: Papers may focus on topics that are not currently trending",
            "**Topic positioning**: Research may not clearly connect to broader research themes",
            "**Topic framing**: Work may be framed in ways that limit visibility to relevant audiences"
        ]
    
    for reason in reasons:
        st.markdown(f"- {reason}")

# ------ Tab 3: Comparative Case Studies ------
with tab3:
    st.header("📋 Comparative Case Studies")
    st.write("Analyze specific examples of low-cited papers compared to similar but highly-cited papers.")
    
    # Sample low-cited papers
    low_citation_threshold = np.percentile(topic_model_df['CITATIONCOUNT'], 25)
    low_cited_papers = topic_model_df[topic_model_df['CITATIONCOUNT'] <= low_citation_threshold]
    
    if len(low_cited_papers) == 0:
        st.warning("No low-cited papers found for analysis.")
    else:
        # Sample a few papers for case studies
        sample_size = min(5, len(low_cited_papers))
        paper_samples = low_cited_papers.sample(sample_size)
        
        # Define topic columns for similarity calculation
        topic_cols = model_columns[model_key]
        
        # Create cosine similarity function
        def cosine_similarity(v1, v2):
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            return dot_product / (norm_v1 * norm_v2) if norm_v1 > 0 and norm_v2 > 0 else 0
        
        # Function to find similar but highly cited papers
        def find_similar_high_cited(paper_id, paper_vector, low_threshold, high_threshold=None):
            if high_threshold is None:
                high_threshold = np.percentile(topic_model_df['CITATIONCOUNT'], 75)
                
            high_cited = topic_model_df[topic_model_df['CITATIONCOUNT'] >= high_threshold]
            
            if len(high_cited) == 0:
                return pd.DataFrame()
                
            similarities = []
            for idx, row in high_cited.iterrows():
                if row.get('ArticleID', idx) != paper_id:
                    sim = cosine_similarity(paper_vector, row[topic_cols].values)
                    similarities.append({
                        'ArticleID': row.get('ArticleID', idx),
                        'Citations': row.get('CITATIONCOUNT', 0),
                        'Similarity': sim
                    })
            
            if not similarities:
                return pd.DataFrame()
                
            sim_df = pd.DataFrame(similarities).sort_values('Similarity', ascending=False)
            return sim_df.head(3)  # Return top 3 similar papers
        
        # Loop through sample papers
        for i, (idx, row) in enumerate(paper_samples.iterrows()):
            paper_id = row.get('ArticleID', idx)
            citations = row.get('CITATIONCOUNT', 0)
            
            # Get the paper title if available
            if 'Article Id' in citation_df.columns:
                title_row = citation_df[citation_df['Article Id'] == paper_id]
                title = title_row['Title'].values[0] if len(title_row) > 0 else f"Paper {paper_id}"
            else:
                title = f"Paper {paper_id}"
            
            st.subheader(f"Case Study {i+1}: {title}")
            st.write(f"**Paper ID:** {paper_id}")
            st.write(f"**Citations:** {citations}")
            
            # Get topic vector
            paper_vector = row[topic_cols].values
            
            # Find similar but highly cited papers
            similar_high_cited = find_similar_high_cited(paper_id, paper_vector, low_citation_threshold)
            
            if len(similar_high_cited) == 0:
                st.warning("No similar high-cited papers found.")
                continue
            
            # Display similar papers
            st.write("**Similar but Highly-Cited Papers:**")
            
            for j, (sim_idx, sim_row) in enumerate(similar_high_cited.iterrows()):
                sim_paper_id = sim_row['ArticleID']
                sim_citations = sim_row['Citations']
                similarity = sim_row['Similarity'] * 100  # Convert to percentage
                
                # Get the paper title if available
                if 'Article Id' in citation_df.columns:
                    sim_title_row = citation_df[citation_df['Article Id'] == sim_paper_id]
                    sim_title = sim_title_row['Title'].values[0] if len(sim_title_row) > 0 else f"Paper {sim_paper_id}"
                else:
                    sim_title = f"Paper {sim_paper_id}"
                
                st.markdown(f"**{j+1}. {sim_title}**")
                st.write(f"Citations: {sim_citations} (compared to {citations})")
                st.write(f"Topic Similarity: {similarity:.1f}%")
                
                # Get topic vectors for comparison
                sim_paper_row = topic_model_df[topic_model_df['ArticleID'] == sim_paper_id]
                if len(sim_paper_row) > 0:
                    sim_paper_vector = sim_paper_row[topic_cols].values[0]
                    
                    # Create comparison dataframe
                    topic_comparison = pd.DataFrame({
                        'Topic': [f"Topic {i+1}" for i in range(len(topic_cols))],
                        'Low-cited Paper': paper_vector,
                        'High-cited Paper': sim_paper_vector,
                        'Difference': sim_paper_vector - paper_vector
                    })
                    
                    # Sort by absolute difference
                    topic_comparison['Abs Difference'] = abs(topic_comparison['Difference'])
                    topic_comparison = topic_comparison.sort_values('Abs Difference', ascending=False)
                    
                    # Show only top differences
                    top_diffs = topic_comparison.head(3)
                    
                    st.write("**Key Topic Differences:**")
                    for _, diff_row in top_diffs.iterrows():
                        diff_value = diff_row['Difference']
                        if diff_value > 0:
                            st.write(f"- {diff_row['Topic']}: **Higher** in highly-cited paper ({diff_value:.3f})")
                        else:
                            st.write(f"- {diff_row['Topic']}: **Lower** in highly-cited paper ({diff_value:.3f})")
                
                # Add visual separator
                st.markdown("---")
            
            # Analysis of differences
            st.write("**Analysis of Key Differences:**")
            
            # Identify potential reasons for citation gap
            if len(similar_high_cited) > 0:
                topic_gaps = []
                for _, sim_row in similar_high_cited.iterrows():
                    sim_paper_id = sim_row['ArticleID']
                    sim_paper_row = topic_model_df[topic_model_df['ArticleID'] == sim_paper_id]
                    
                    if len(sim_paper_row) > 0:
                        sim_paper_vector = sim_paper_row[topic_cols].values[0]
                        diffs = sim_paper_vector - paper_vector
                        
                        # Count positive and negative differences
                        pos_diffs = [(i, diff) for i, diff in enumerate(diffs) if diff > 0.05]
                        neg_diffs = [(i, diff) for i, diff in enumerate(diffs) if diff < -0.05]
                        
                        for idx, diff in pos_diffs:
                            topic_gaps.append((f"Topic {idx+1}", diff))
                        
                # Count topic frequency across similar papers
                topic_freq = {}
                for topic, diff in topic_gaps:
                    if topic in topic_freq:
                        topic_freq[topic] += 1
                    else:
                        topic_freq[topic] = 1
                
                # Sort by frequency
                sorted_gaps = sorted(topic_freq.items(), key=lambda x: x[1], reverse=True)
                
                # List key topic differences
                for topic, freq in sorted_gaps[:3]:
                    if freq > 1:  # Mention only if appears in multiple papers
                        st.write(f"- **{topic}**: Present in {freq}/{len(similar_high_cited)} highly-cited similar papers, but weaker in this paper")
                
                # Check correlation with citation
                for topic, _ in sorted_gaps[:3]:
                    topic_idx = int(topic.split()[1]) - 1
                    topic_col = topic_cols[topic_idx]
                    corr = corr_df.loc[corr_df['Topic'] == f"Topic {topic_idx+1}", 'Pearson Correlation'].values
                    if len(corr) > 0 and corr[0] > 0:
                        st.write(f"- **{topic}** has positive correlation ({corr[0]:.3f}) with citations")
            
            # Add visual separator
            st.markdown("---")
    
    # General recommendations for improving citation counts
    st.subheader("💡 Recommendations for Improving Citation Potential")
    
    # Get positive correlation topics
    positive_corr_topics = corr_df[corr_df['Pearson Correlation'] > 0].sort_values('Pearson Correlation', ascending=False)
    
    st.markdown("""
    Based on the analysis of low-cited papers compared to similar highly-cited papers, here are general recommendations for improving citation potential:
    """)
    
    st.markdown("**1. Topic Selection and Framing:**")
    if not positive_corr_topics.empty:
        st.write("Consider incorporating these high-impact topics in your research:")
        for i, row in positive_corr_topics.head(3).iterrows():
            st.markdown(f"   - **{row['Topic']}** (correlation with citations: {row['Pearson Correlation']:.3f})")
    
    st.markdown("""
    **2. Topic Balance:**
    - Aim for a balance between specialization and breadth
    - Very high concentration in a single topic may limit audience
    - Very low concentration across many topics may dilute contribution
    
    **3. Strategic Positioning:**
    - Connect your work to broader research themes
    - Clearly position relative to trending research areas
    - Frame findings in terms of wider implications
    
    **4. Topic Combination:**
    - Consider combining specialized topics with more broadly appealing ones
    - Create novel combinations of topics that bridge research communities
    """)


# Footer
st.markdown("---")
st.markdown("**Citation Analysis Platform**: Developed to help researchers understand and improve citation impact")