import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.visualization import plot_correlation_matrix
from scipy import stats

st.set_page_config(
    page_title="Correlation Analysis",
    page_icon="🔄",
    layout="wide"
)

st.title("Correlation Analysis")

# Check if required data is available
if 'citation_data' not in st.session_state:
    st.error("No citation data available. Please upload data in the main page.")
    st.stop()

# Get the data
citation_df = st.session_state['citation_data']

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "Citation Correlations",
    "Multi-factor Analysis",
    "Statistical Tests"
])

with tab1:
    st.header("Correlation Between Features and Citations")
    
    # Extract year columns
    year_cols = [col for col in citation_df.columns if col.isdigit() and 1992 <= int(col) <= 2023]
    
    # Create metadata features
    st.subheader("Paper Metadata Analysis")
    
    # Create additional features from available data
    features_df = pd.DataFrame()
    
    # Add paper ID
    if 'Article Id' in citation_df.columns:
        features_df['Article Id'] = citation_df['Article Id']
    
    # Add citation count
    features_df['Citation Count'] = citation_df['Cited By']
    
    # Add author count (if Author column exists)
    if 'Author' in citation_df.columns:
        features_df['Author Count'] = citation_df['Author'].apply(
            lambda x: len(str(x).split(',')) if pd.notnull(x) else 0
        )
    
    # Add title length (if Title column exists)
    if 'Title' in citation_df.columns:
        features_df['Title Length'] = citation_df['Title'].apply(
            lambda x: len(str(x)) if pd.notnull(x) else 0
        )
        
        # Add whether title has colon
        features_df['Title Has Colon'] = citation_df['Title'].apply(
            lambda x: 1 if ':' in str(x) else 0
        )
        
        # Add whether title has question mark
        features_df['Title Has Question'] = citation_df['Title'].apply(
            lambda x: 1 if '?' in str(x) else 0
        )
    
    # Add years since publication (from first year in dataset)
    if year_cols:
        first_year = int(year_cols[0])
        current_year = 2023
        features_df['Years Since Publication'] = current_year - first_year
    
    # Add early citation metrics (citations in first 3 years)
    if len(year_cols) >= 3:
        first_three_years = year_cols[:3]
        features_df['Early Citations'] = citation_df[first_three_years].sum(axis=1)
    
    # Add citation in first year
    if year_cols:
        first_year_col = year_cols[0]
        features_df['First Year Citations'] = citation_df[first_year_col]
    
    # Calculate correlation with citation count
    numeric_cols = features_df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'Citation Count']
    
    correlations = []
    for col in numeric_cols:
        corr = features_df[col].corr(features_df['Citation Count'])
        correlations.append({
            'Feature': col,
            'Correlation': corr
        })
    
    # Create correlation dataframe
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('Correlation', ascending=False)
    
    # Display table
    st.write("Correlation with Citation Count:")
    st.dataframe(corr_df)
    
    # Create bar chart
    fig = px.bar(
        corr_df,
        x='Feature',
        y='Correlation',
        title="Feature Correlation with Citation Count",
        color='Correlation',
        color_continuous_scale='RdBu_r',
        labels={'Correlation': 'Pearson Correlation'}
    )
    
    fig.update_layout(
        xaxis_title="Feature",
        yaxis_title="Correlation with Citation Count",
        xaxis=dict(tickangle=45),
        yaxis=dict(range=[-1, 1])
    )
    
    # Add a horizontal line at zero
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=0,
        x1=len(numeric_cols) - 0.5,
        y1=0,
        line=dict(
            color="black",
            width=1,
            dash="dash",
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plots for top correlated features
    st.subheader("Scatter Plots for Top Correlated Features")
    
    # Get top correlated features
    top_features = corr_df.head(4)['Feature'].tolist()
    
    for feature in top_features:
        fig = px.scatter(
            features_df,
            x=feature,
            y='Citation Count',
            title=f"{feature} vs. Citation Count",
            trendline="ols",
            labels={feature: feature, 'Citation Count': 'Citation Count'}
        )
        
        # Add correlation coefficient
        corr = features_df[feature].corr(features_df['Citation Count'])
        fig.add_annotation(
            x=0.95, y=0.95,
            xref="paper", yref="paper",
            text=f"Correlation: {corr:.2f}",
            showarrow=False,
            align="right",
            bgcolor="rgba(255,255,255,0.8)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Citation pattern analysis
    st.subheader("Citation Pattern Analysis")
    
    # Calculate ratio of early to total citations
    if 'Early Citations' in features_df.columns:
        features_df['Early Citation Ratio'] = features_df['Early Citations'] / features_df['Citation Count'].replace(0, np.nan)
        features_df['Early Citation Ratio'] = features_df['Early Citation Ratio'].fillna(0)
        
        # Create scatter plot
        fig = px.scatter(
            features_df,
            x='Early Citation Ratio',
            y='Citation Count',
            title="Early Citation Ratio vs. Total Citations",
            labels={'Early Citation Ratio': 'Early Citations / Total Citations', 'Citation Count': 'Citation Count'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create histogram of early citation ratio
        fig = px.histogram(
            features_df,
            x='Early Citation Ratio',
            nbins=30,
            title="Distribution of Early Citation Ratio",
            labels={'Early Citation Ratio': 'Early Citations / Total Citations', 'count': 'Number of Papers'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        st.markdown("""
        **Early Citation Ratio Interpretation:**
        
        - Higher values (closer to 1) indicate papers that received most of their citations early
        - Lower values indicate papers that accumulated citations more gradually or later
        - Papers with high total citations but low early citation ratio may be "sleeping beauties"
        """)

with tab2:
    st.header("Multi-factor Analysis")
    
    # Combine data from different sources
    st.subheader("Combined Feature Analysis")
    
    # Check if topic model data is available
    has_topic_data = 'topic_model_data' in st.session_state
    
    # Create combined dataframe
    combined_df = features_df.copy()
    
    # Add topic model features if available
    if has_topic_data:
        topic_model_df = st.session_state['topic_model_data']
        
        # Check if ArticleID column exists in topic model data
        if 'ArticleID' in topic_model_df.columns and 'Article Id' in combined_df.columns:
            # Create temporary dataframes with renamed ID columns for merging
            topic_temp = topic_model_df.rename(columns={'ArticleID': 'Article Id'})
            
            # Get topic model columns (exclude ID and citation count)
            topic_cols = [col for col in topic_temp.columns 
                         if col != 'Article Id' and col != 'CITATIONCOUNT']
            
            # Merge topic data with combined data
            if topic_cols:
                # Select only necessary columns to avoid duplicates
                topic_subset = topic_temp[['Article Id'] + topic_cols]
                combined_df = pd.merge(combined_df, topic_subset, on='Article Id', how='left')
                
                st.success(f"Added {len(topic_cols)} topic model features to combined analysis.")
    
    # Check for text features in session state
    has_text_features = 'paper_text_features' in st.session_state
    
    if has_text_features:
        text_features_df = st.session_state['paper_text_features']
        
        # Merge text features if article IDs match
        if 'Article Id' in combined_df.columns and 'paper_id' in text_features_df.columns:
            text_features_df = text_features_df.rename(columns={'paper_id': 'Article Id'})
            
            # Get text feature columns
            text_cols = [col for col in text_features_df.columns if col != 'Article Id']
            
            if text_cols:
                # Select only necessary columns
                text_subset = text_features_df[['Article Id'] + text_cols]
                combined_df = pd.merge(combined_df, text_subset, on='Article Id', how='left')
                
                st.success(f"Added {len(text_cols)} text analysis features to combined analysis.")
    
    # Display feature counts
    metadata_features = [col for col in features_df.columns if col != 'Article Id' and col != 'Citation Count']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Metadata Features", len(metadata_features))
    
    with col2:
        topic_features = [col for col in combined_df.columns 
                         if col in topic_model_df.columns 
                         and col != 'ArticleID' 
                         and col != 'CITATIONCOUNT'] if has_topic_data else []
        st.metric("Topic Features", len(topic_features))
    
    with col3:
        text_features = [col for col in combined_df.columns 
                        if has_text_features and col in text_features_df.columns 
                        and col != 'paper_id'] if has_text_features else []
        st.metric("Text Features", len(text_features))
    
    # Create correlation matrix
    st.subheader("Feature Correlation Matrix")
    
    # Get numeric columns (excluding ID)
    numeric_cols = combined_df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'Article Id']
    
    # Allow user to select features for the correlation matrix
    if len(numeric_cols) > 15:
        st.warning(f"There are {len(numeric_cols)} numeric features. Please select a subset for the correlation matrix.")
        
        # Default selected features
        default_features = ['Citation Count']
        
        if 'Early Citations' in numeric_cols:
            default_features.append('Early Citations')
        if 'Author Count' in numeric_cols:
            default_features.append('Author Count')
        if 'Title Length' in numeric_cols:
            default_features.append('Title Length')
        
        # Add a few more features if available
        if has_topic_data and topic_features:
            default_features.extend(topic_features[:3])
        
        if has_text_features and text_features:
            # Select text features likely to be useful
            useful_text_features = [col for col in text_features if any(term in col.lower() for term in 
                                   ['diversity', 'reading', 'sentence', 'word', 'ratio'])]
            default_features.extend(useful_text_features[:3])
        
        # Ensure we don't have too many default features
        default_features = list(set(default_features))[:10]
        
        # Let user select features
        selected_features = st.multiselect(
            "Select features for correlation matrix",
            options=numeric_cols,
            default=default_features
        )
        
        if selected_features:
            fig = plot_correlation_matrix(combined_df, selected_features, "Correlation Matrix of Selected Features")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please select at least one feature for the correlation matrix.")
    else:
        # If not too many features, show the full correlation matrix
        fig = plot_correlation_matrix(combined_df, numeric_cols, "Correlation Matrix of All Features")
        st.plotly_chart(fig, use_container_width=True)
    
    # Key feature pairs analysis
    st.subheader("Key Feature Relationships")
    
    # Find most correlated pairs
    if len(numeric_cols) > 1 and 'Citation Count' in numeric_cols:
        citation_correlations = []
        
        for col in numeric_cols:
            if col != 'Citation Count':
                corr = combined_df[col].corr(combined_df['Citation Count'])
                citation_correlations.append({
                    'Feature': col,
                    'Correlation': corr,
                    'Abs_Correlation': abs(corr)
                })
        
        correlation_df = pd.DataFrame(citation_correlations)
        correlation_df = correlation_df.sort_values('Abs_Correlation', ascending=False)
        
        # Get top correlated features
        top_correlated = correlation_df.head(6)['Feature'].tolist()
        
        if top_correlated:
            st.write("Scatter plots for features most correlated with citation count:")
            
            # Create scatter plots for top 2 features at a time
            for i in range(0, len(top_correlated), 2):
                col1, col2 = st.columns(2)
                
                with col1:
                    if i < len(top_correlated):
                        feature = top_correlated[i]
                        fig = px.scatter(
                            combined_df,
                            x=feature,
                            y='Citation Count',
                            title=f"{feature} vs. Citation Count",
                            trendline="ols",
                            labels={feature: feature, 'Citation Count': 'Citation Count'}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if i+1 < len(top_correlated):
                        feature = top_correlated[i+1]
                        fig = px.scatter(
                            combined_df,
                            x=feature,
                            y='Citation Count',
                            title=f"{feature} vs. Citation Count",
                            trendline="ols",
                            labels={feature: feature, 'Citation Count': 'Citation Count'}
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
    
    # 3D scatter plot
    st.subheader("3D Feature Visualization")
    
    if len(numeric_cols) >= 3:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_feature = st.selectbox(
                "X-axis feature",
                options=[col for col in numeric_cols if col != 'Citation Count'],
                index=0
            )
        
        with col2:
            y_feature = st.selectbox(
                "Y-axis feature",
                options=[col for col in numeric_cols if col != 'Citation Count' and col != x_feature],
                index=0
            )
        
        with col3:
            z_feature = st.selectbox(
                "Z-axis feature (or color)",
                options=['Citation Count'] + [col for col in numeric_cols if col != 'Citation Count' and col != x_feature and col != y_feature],
                index=0
            )
        
        # Create 3D scatter plot
        fig = px.scatter_3d(
            combined_df,
            x=x_feature,
            y=y_feature,
            z=z_feature,
            color='Citation Count',
            title=f"3D Relationship: {x_feature} vs {y_feature} vs {z_feature}",
            labels={
                x_feature: x_feature,
                y_feature: y_feature,
                z_feature: z_feature,
                'Citation Count': 'Citation Count'
            }
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title=x_feature,
                yaxis_title=y_feature,
                zaxis_title=z_feature,
            ),
            height=700
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Statistical Tests")
    
    # Run statistical tests
    st.subheader("Feature Significance Tests")
    
    # Select test type
    test_type = st.radio(
        "Select statistical test",
        ["Correlation Significance", "Group Comparison"],
        horizontal=True
    )
    
    if test_type == "Correlation Significance":
        # Get feature options
        feature_options = [col for col in features_df.columns 
                          if col != 'Article Id' and col != 'Citation Count' 
                          and pd.api.types.is_numeric_dtype(features_df[col])]
        
        # Add topic and text features if available
        if has_topic_data:
            topic_features = [col for col in topic_model_df.columns 
                             if col != 'ArticleID' and col != 'CITATIONCOUNT'
                             and pd.api.types.is_numeric_dtype(topic_model_df[col])]
            feature_options.extend(topic_features)
        
        if has_text_features:
            text_features = [col for col in text_features_df.columns 
                            if col != 'paper_id'
                            and pd.api.types.is_numeric_dtype(text_features_df[col])]
            feature_options.extend(text_features)
        
        # Remove duplicates
        feature_options = list(set(feature_options))
        
        # Let user select features
        selected_features = st.multiselect(
            "Select features to test",
            options=feature_options,
            default=feature_options[:min(5, len(feature_options))]
        )
        
        if selected_features:
            # Create results table
            results = []
            
            for feature in selected_features:
                # Find the feature in the appropriate dataframe
                if feature in features_df.columns:
                    data = features_df
                    id_col = 'Article Id' if 'Article Id' in features_df.columns else None
                elif has_topic_data and feature in topic_model_df.columns:
                    data = topic_model_df
                    id_col = 'ArticleID'
                elif has_text_features and feature in text_features_df.columns:
                    data = text_features_df
                    id_col = 'paper_id'
                else:
                    # Skip if feature not found
                    continue
                
                # Get citation counts for this feature
                if 'Citation Count' in data.columns:
                    citations = data['Citation Count']
                elif 'Cited By' in data.columns:
                    citations = data['Cited By']
                elif 'CITATIONCOUNT' in data.columns:
                    citations = data['CITATIONCOUNT']
                else:
                    # Try to merge with citation data
                    if id_col and 'Article Id' in citation_df.columns:
                        merged = pd.merge(
                            data[[id_col, feature]], 
                            citation_df[['Article Id', 'Cited By']], 
                            left_on=id_col, 
                            right_on='Article Id',
                            how='inner'
                        )
                        
                        if not merged.empty:
                            feature_data = merged[feature]
                            citations = merged['Cited By']
                        else:
                            # Skip if no matches
                            continue
                    else:
                        # Skip if no way to get citations
                        continue
                
                # Calculate Pearson correlation
                if feature in data.columns:
                    feature_data = data[feature]
                else:
                    # Skip if feature not found
                    continue
                
                # Remove rows with NaN values
                valid_data = pd.DataFrame({
                    'feature': feature_data,
                    'citations': citations
                }).dropna()
                
                if len(valid_data) < 2:
                    # Skip if not enough data
                    continue
                
                # Calculate correlation and p-value
                corr, p_value = stats.pearsonr(valid_data['feature'], valid_data['citations'])
                
                # Determine significance
                alpha = 0.05
                is_significant = p_value < alpha
                
                # Add to results
                results.append({
                    'Feature': feature,
                    'Correlation': corr,
                    'P-value': p_value,
                    'Significant': 'Yes' if is_significant else 'No'
                })
            
            # Create dataframe
            results_df = pd.DataFrame(results)
            
            # Sort by significance, then by absolute correlation
            results_df['Abs_Correlation'] = results_df['Correlation'].abs()
            results_df = results_df.sort_values(['Significant', 'Abs_Correlation'], ascending=[False, False])
            results_df = results_df.drop('Abs_Correlation', axis=1)
            
            # Display results
            st.write("Correlation Significance Test Results:")
            st.dataframe(results_df)
            
            # Visualize significant features
            significant_features = results_df[results_df['Significant'] == 'Yes']['Feature'].tolist()
            
            if significant_features:
                st.subheader("Significant Features")
                
                # Create bar chart of correlations for significant features
                sig_results = results_df[results_df['Significant'] == 'Yes']
                
                fig = px.bar(
                    sig_results,
                    x='Feature',
                    y='Correlation',
                    title="Correlations of Significant Features with Citation Count",
                    color='Correlation',
                    color_continuous_scale='RdBu_r',
                    labels={'Correlation': 'Pearson Correlation'}
                )
                
                fig.update_layout(
                    xaxis_title="Feature",
                    yaxis_title="Correlation with Citation Count",
                    xaxis=dict(tickangle=45),
                    yaxis=dict(range=[-1, 1])
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation
                st.markdown("""
                **Statistical Significance Interpretation:**
                
                - **Correlation**: Measures the strength and direction of the relationship between a feature and citation count
                - **P-value**: Probability that the observed correlation occurred by chance
                - **Significant**: Features with p-value < 0.05 are statistically significant
                
                Statistically significant features are those most likely to have a real relationship with citation counts.
                """)
            else:
                st.info("No statistically significant features found.")
        else:
            st.info("Please select at least one feature to test.")
    
    elif test_type == "Group Comparison":
        st.subheader("High vs. Low Citation Group Comparison")
        
        # Define groups
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
        
        # Create high and low citation dataframes
        high_cited = citation_df[citation_df['Cited By'] >= high_threshold]
        low_cited = citation_df[citation_df['Cited By'] <= low_threshold]
        
        st.write(f"Number of high-cited papers: {len(high_cited)}")
        st.write(f"Number of low-cited papers: {len(low_cited)}")
        
        # Select features for comparison
        feature_options = []
        
        # Add metadata features
        metadata_features = [col for col in features_df.columns 
                            if col != 'Article Id' and col != 'Citation Count' 
                            and pd.api.types.is_numeric_dtype(features_df[col])]
        feature_options.extend(metadata_features)
        
        # Add topic features if available
        if has_topic_data:
            topic_features = [col for col in topic_model_df.columns 
                             if col != 'ArticleID' and col != 'CITATIONCOUNT'
                             and pd.api.types.is_numeric_dtype(topic_model_df[col])]
            feature_options.extend(topic_features)
        
        # Add text features if available
        if has_text_features:
            text_features = [col for col in text_features_df.columns 
                            if col != 'paper_id'
                            and pd.api.types.is_numeric_dtype(text_features_df[col])]
            feature_options.extend(text_features)
        
        # Remove duplicates
        feature_options = list(set(feature_options))
        
        # Let user select features
        selected_features = st.multiselect(
            "Select features to compare",
            options=feature_options,
            default=feature_options[:min(5, len(feature_options))]
        )
        
        if selected_features:
            # Create results table
            results = []
            
            for feature in selected_features:
                # Find the feature in the appropriate dataframe
                if feature in features_df.columns:
                    data = features_df
                    id_col = 'Article Id' if 'Article Id' in features_df.columns else None
                elif has_topic_data and feature in topic_model_df.columns:
                    data = topic_model_df
                    id_col = 'ArticleID'
                elif has_text_features and feature in text_features_df.columns:
                    data = text_features_df
                    id_col = 'paper_id'
                else:
                    # Skip if feature not found
                    continue
                
                # Get high and low cited paper IDs
                if 'Article Id' in high_cited.columns and id_col:
                    high_ids = high_cited['Article Id'].tolist()
                    low_ids = low_cited['Article Id'].tolist()
                    
                    # Filter data to high and low cited papers
                    high_data = data[data[id_col].isin(high_ids)][feature]
                    low_data = data[data[id_col].isin(low_ids)][feature]
                    
                    # Run t-test if enough data
                    if len(high_data) > 1 and len(low_data) > 1:
                        # Remove NaN values
                        high_data = high_data.dropna()
                        low_data = low_data.dropna()
                        
                        if len(high_data) > 1 and len(low_data) > 1:
                            # Calculate mean values
                            high_mean = high_data.mean()
                            low_mean = low_data.mean()
                            
                            # Calculate difference
                            difference = high_mean - low_mean
                            percent_diff = (difference / low_mean) * 100 if low_mean != 0 else float('inf')
                            
                            # Run t-test
                            t_stat, p_value = stats.ttest_ind(high_data, low_data, equal_var=False)
                            
                            # Determine significance
                            alpha = 0.05
                            is_significant = p_value < alpha
                            
                            # Add to results
                            results.append({
                                'Feature': feature,
                                'High Cited Mean': high_mean,
                                'Low Cited Mean': low_mean,
                                'Difference': difference,
                                'Percent Difference': percent_diff,
                                'P-value': p_value,
                                'Significant': 'Yes' if is_significant else 'No'
                            })
            
            if results:
                # Create dataframe
                results_df = pd.DataFrame(results)
                
                # Sort by significance, then by absolute difference
                results_df['Abs_Diff'] = results_df['Difference'].abs()
                results_df = results_df.sort_values(['Significant', 'Abs_Diff'], ascending=[False, False])
                results_df = results_df.drop('Abs_Diff', axis=1)
                
                # Format percent difference
                results_df['Percent Difference'] = results_df['Percent Difference'].apply(
                    lambda x: f"{x:.1f}%" if not pd.isna(x) and abs(x) != float('inf') else "N/A"
                )
                
                # Display results
                st.write("Group Comparison Test Results:")
                st.dataframe(results_df)
                
                # Visualize significant differences
                significant_features = results_df[results_df['Significant'] == 'Yes']['Feature'].tolist()
                
                if significant_features:
                    st.subheader("Significant Differences")
                    
                    # Create grouped bar chart for comparison
                    compare_data = []
                    
                    for _, row in results_df[results_df['Significant'] == 'Yes'].iterrows():
                        compare_data.append({
                            'Feature': row['Feature'],
                            'Group': 'High Cited',
                            'Value': row['High Cited Mean']
                        })
                        compare_data.append({
                            'Feature': row['Feature'],
                            'Group': 'Low Cited',
                            'Value': row['Low Cited Mean']
                        })
                    
                    compare_df = pd.DataFrame(compare_data)
                    
                    # Create grouped bar chart
                    fig = px.bar(
                        compare_df,
                        x='Feature',
                        y='Value',
                        color='Group',
                        barmode='group',
                        title="Feature Comparison: High vs. Low Cited Papers",
                        color_discrete_map={
                            'High Cited': '#3366CC',
                            'Low Cited': '#FF9900'
                        }
                    )
                    
                    fig.update_layout(
                        xaxis_title="Feature",
                        yaxis_title="Average Value",
                        xaxis=dict(tickangle=45)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interpretation
                    st.markdown("""
                    **Group Comparison Interpretation:**
                    
                    - **High/Low Cited Mean**: Average feature value for papers in each group
                    - **Difference**: Absolute difference between group means
                    - **Percent Difference**: Relative difference as a percentage
                    - **P-value**: Probability that the observed difference occurred by chance
                    - **Significant**: Features with p-value < 0.05 show statistically significant differences
                    
                    Features with significant differences may be important factors affecting citation rates.
                    """)
                else:
                    st.info("No statistically significant differences found between high and low cited papers.")
            else:
                st.warning("Could not perform comparison. Check that feature data can be linked to citation data.")
        else:
            st.info("Please select at least one feature to compare.")

    # Summary of findings
    st.subheader("Summary of Findings")
    
    # Create expandable section
    with st.expander("View summary of key factors affecting citation rates"):
        st.markdown("""
        Based on the correlation and statistical analyses, these factors appear to influence citation rates:
        
        1. **Early Citations**: Papers that receive citations soon after publication tend to accumulate more citations over time
        2. **Author Count**: Papers with more authors often receive more citations, possibly due to wider networks
        3. **Publication Age**: Older papers have had more time to accumulate citations
        4. **Title Properties**: Certain title characteristics may affect citation rates
        
        When comparing high and low cited papers:
        
        - **Topic Differences**: Certain research topics attract more citations
        - **Text Complexity**: Readability and linguistic features show some correlation with citation counts
        - **Citation Patterns**: Papers follow different citation trajectories over time
        
        **Note**: Correlation does not imply causation. These factors are associated with citation rates but may not directly cause higher citations.
        """)

