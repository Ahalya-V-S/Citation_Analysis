import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt

def plot_citation_distribution(citation_df, title="Distribution of Citation Counts"):
    """
    Plot the distribution of citation counts
    
    Parameters:
    -----------
    citation_df : DataFrame
        DataFrame containing citation data
    title : str
        Plot title
    
    Returns:
    --------
    fig
        Plotly figure object
    """
    # Create histogram
    fig = px.histogram(
        citation_df, 
        x="Cited By",
        nbins=50,
        title=title,
        labels={"Cited By": "Citation Count", "count": "Number of Papers"},
        opacity=0.7,
        color_discrete_sequence=['#3366CC']
    )
    
    # Add median line
    median = citation_df["Cited By"].median()
    fig.add_vline(x=median, line_dash="dash", line_color="red", 
                annotation_text=f"Median: {median:.1f}", 
                annotation_position="top right")
    
    # Add mean line
    mean = citation_df["Cited By"].mean()
    fig.add_vline(x=mean, line_dash="dash", line_color="green", 
                annotation_text=f"Mean: {mean:.1f}", 
                annotation_position="top right")
    
    # Update layout
    fig.update_layout(
        xaxis_title="Citation Count",
        yaxis_title="Number of Papers",
        bargap=0.2,
        plot_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgrey'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgrey'
        )
    )
    
    return fig

def plot_citation_trends(citation_df, x_scale='linear', y_scale='linear'):
    """
    Plot citation trends over the years
    
    Parameters:
    -----------
    citation_df : DataFrame
        DataFrame containing citation data with yearly columns
    x_scale : str
        Scale for x-axis ('linear' or 'log')
    y_scale : str
        Scale for y-axis ('linear' or 'log')
    
    Returns:
    --------
    fig
        Plotly figure object
    """
    # Extract year columns
    year_cols = [col for col in citation_df.columns if col.isdigit() and 1992 <= int(col) <= 2023]
    
    # Calculate yearly citation totals and average per paper
    yearly_totals = citation_df[year_cols].sum()
    yearly_averages = citation_df[year_cols].mean()
    
    # Create a DataFrame for plotting
    trend_df = pd.DataFrame({
        'Year': [int(year) for year in year_cols],
        'Total Citations': yearly_totals.values,
        'Average Citations': yearly_averages.values
    })
    
    # Create subplots
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=("Total Citations by Year", "Average Citations per Paper by Year"),
                        x_title="Year",
                        horizontal_spacing=0.15)
    
    # Add total citations trace
    fig.add_trace(
        go.Scatter(
            x=trend_df['Year'], 
            y=trend_df['Total Citations'],
            mode='lines+markers',
            name='Total Citations',
            marker=dict(color='#3366CC'),
            line=dict(width=2)
        ),
        row=1, col=1
    )
    
    # Add average citations trace
    fig.add_trace(
        go.Scatter(
            x=trend_df['Year'], 
            y=trend_df['Average Citations'],
            mode='lines+markers',
            name='Average Citations',
            marker=dict(color='#FF9900'),
            line=dict(width=2)
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        plot_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            type=x_scale,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgrey'
        ),
        yaxis=dict(
            title="Total Citations",
            type=y_scale,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgrey'
        ),
        xaxis2=dict(
            type=x_scale,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgrey'
        ),
        yaxis2=dict(
            title="Average Citations",
            type=y_scale,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgrey'
        )
    )
    
    return fig

def plot_citation_heatmap(citation_df, num_papers=30):
    """
    Create a heatmap of citations over time for top cited papers
    
    Parameters:
    -----------
    citation_df : DataFrame
        DataFrame containing citation data with yearly columns
    num_papers : int
        Number of top cited papers to include
    
    Returns:
    --------
    fig
        Plotly figure object
    """
    # Extract year columns
    year_cols = [col for col in citation_df.columns if col.isdigit() and 1992 <= int(col) <= 2023]
    
    # Get top cited papers
    top_papers = citation_df.nlargest(num_papers, 'Cited By')
    
    # Create paper labels
    paper_labels = []
    for _, row in top_papers.iterrows():
        author = row['Author'].split(',')[0] if ',' in row['Author'] else row['Author']
        title = row['Title'][:30] + '...' if len(row['Title']) > 30 else row['Title']
        paper_labels.append(f"{author}: {title}")
    
    # Create heatmap data
    heatmap_data = top_papers[year_cols].values
    
    # Normalize data for better visualization (log scale)
    heatmap_data_norm = np.log1p(heatmap_data)  # log(1+x) to handle zeros
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data_norm,
        x=[int(year) for year in year_cols],
        y=paper_labels,
        colorscale='Viridis',
        colorbar=dict(
            title="Log(Citations + 1)",
        )
    ))
    
    # Update layout
    fig.update_layout(
        title="Citation Patterns Over Time (Top Cited Papers)",
        xaxis_title="Year",
        yaxis_title="Paper",
        height=700,
        plot_bgcolor='white',
        xaxis=dict(
            tickmode='linear',
            tick0=int(year_cols[0]),
            dtick=5
        )
    )
    
    return fig

def plot_comparative_metrics(papers_features_df, citation_column=None):
    """
    Create plots comparing linguistic features with citation counts
    
    Parameters:
    -----------
    papers_features_df : DataFrame
        DataFrame containing paper features
    citation_column : str, optional
        Name of the column containing citation counts
    
    Returns:
    --------
    list
        List of Plotly figure objects
    """
    # Features to plot
    feature_cols = [
        'avg_sentence_length', 
        'avg_word_length', 
        'lexical_diversity',
        'flesch_reading_ease',
        'flesch_kincaid_grade',
        'noun_ratio',
        'verb_ratio',
        'adj_ratio'
    ]
    
    figures = []
    
    # If citation data is available, create scatter plots
    if citation_column and citation_column in papers_features_df.columns:
        for feature in feature_cols:
            if feature in papers_features_df.columns:
                fig = px.scatter(
                    papers_features_df,
                    x=feature,
                    y=citation_column,
                    hover_data=['paper_id'],
                    title=f"{feature.replace('_', ' ').title()} vs. Citations",
                    labels={
                        feature: feature.replace('_', ' ').title(),
                        citation_column: "Citation Count"
                    },
                    trendline="ols"
                )
                
                # Add correlation coefficient
                corr = papers_features_df[feature].corr(papers_features_df[citation_column])
                fig.add_annotation(
                    x=0.95, y=0.95,
                    xref="paper", yref="paper",
                    text=f"Correlation: {corr:.2f}",
                    showarrow=False,
                    align="right",
                    bgcolor="rgba(255,255,255,0.8)"
                )
                
                figures.append(fig)
    else:
        # Create box plots for feature distribution
        for feature in feature_cols:
            if feature in papers_features_df.columns:
                fig = px.box(
                    papers_features_df,
                    y=feature,
                    title=f"Distribution of {feature.replace('_', ' ').title()}",
                    labels={feature: feature.replace('_', ' ').title()},
                    points="all",
                    notched=True
                )
                
                figures.append(fig)
    
    return figures

def plot_topic_distributions(topic_model_df, citation_df=None):
    """
    Plot topic distributions and their relationship with citations
    
    Parameters:
    -----------
    topic_model_df : DataFrame
        DataFrame containing topic model data
    citation_df : DataFrame, optional
        DataFrame containing citation data
    
    Returns:
    --------
    tuple
        (topic_dist_fig, topic_citation_fig) Plotly figure objects
    """
    # Identify topic model columns
    topic_prefixes = ['LDA', 'HDP', 'CTM']
    topic_counts = [5, 10]
    
    # Choose one model type for visualization
    available_models = []
    for prefix in topic_prefixes:
        for count in topic_counts:
            model_col = f"{prefix}{count}"
            if model_col in topic_model_df.columns:
                available_models.append(model_col)
    
    if not available_models:
        # No topic model columns found
        return None, None
    
    # Select first available model
    model_name = available_models[0]
    model_prefix = ''.join([c for c in model_name if not c.isdigit()])
    model_topics = int(''.join([c for c in model_name if c.isdigit()]))
    
    # Get topic columns for the selected model
    topic_cols = []
    for i in range(model_topics):
        col_name = f"{model_prefix}{i+1}"
        if col_name in topic_model_df.columns:
            topic_cols.append(col_name)
        else:
            # Try alternative column naming pattern
            col_name = f"{model_name}_{i+1}"
            if col_name in topic_model_df.columns:
                topic_cols.append(col_name)
    
    # Create topic distribution figure
    topic_dist_data = topic_model_df[topic_cols].mean()
    topic_dist_fig = go.Figure(data=[
        go.Bar(
            x=[f"Topic {i+1}" for i in range(len(topic_dist_data))],
            y=topic_dist_data,
            marker_color='#3366CC'
        )
    ])
    
    topic_dist_fig.update_layout(
        title=f"Average Topic Distribution ({model_name})",
        xaxis_title="Topics",
        yaxis_title="Average Weight",
        plot_bgcolor='white',
        xaxis=dict(
            tickmode='linear',
            tickangle=45,
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgrey'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgrey'
        )
    )
    
    # If citation data is available, create topic-citation correlation figure
    topic_citation_fig = None
    if citation_df is not None and 'CITATIONCOUNT' in topic_model_df.columns:
        # Merge topic data with citation data if needed
        if 'ArticleID' in topic_model_df.columns and 'Article Id' in citation_df.columns:
            merged_df = pd.merge(
                topic_model_df, 
                citation_df[['Article Id', 'Cited By']], 
                left_on='ArticleID', 
                right_on='Article Id',
                how='inner'
            )
        else:
            # Use citation count from topic model data
            merged_df = topic_model_df
            merged_df['Cited By'] = merged_df['CITATIONCOUNT']
        
        # Calculate correlation between topics and citations
        topic_correlations = []
        for col in topic_cols:
            corr = merged_df[col].corr(merged_df['Cited By'])
            topic_correlations.append(corr)
        
        # Create correlation figure
        topic_citation_fig = go.Figure(data=[
            go.Bar(
                x=[f"Topic {i+1}" for i in range(len(topic_correlations))],
                y=topic_correlations,
                marker_color=['red' if c < 0 else 'green' for c in topic_correlations]
            )
        ])
        
        topic_citation_fig.update_layout(
            title=f"Correlation Between Topics and Citation Count ({model_name})",
            xaxis_title="Topics",
            yaxis_title="Correlation Coefficient",
            plot_bgcolor='white',
            xaxis=dict(
                tickmode='linear',
                tickangle=45,
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgrey'
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgrey',
                range=[-1, 1]
            )
        )
        
        # Add a horizontal line at zero
        topic_citation_fig.add_shape(
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
    
    return topic_dist_fig, topic_citation_fig

def plot_correlation_matrix(df, numeric_columns=None, title="Correlation Matrix"):
    """
    Create a correlation matrix heatmap
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame containing the data
    numeric_columns : list, optional
        List of numeric columns to include in correlation matrix
    title : str
        Plot title
    
    Returns:
    --------
    fig
        Plotly figure object
    """
    # Select numeric columns if not specified
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_columns].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title=title
    )
    
    # Add correlation values as text
    for i, row in enumerate(corr_matrix.values):
        for j, val in enumerate(row):
            fig.add_annotation(
                x=j,
                y=i,
                text=f"{val:.2f}",
                showarrow=False,
                font=dict(
                    color='white' if abs(val) > 0.5 else 'black'
                )
            )
    
    # Update layout
    fig.update_layout(
        height=700,
        plot_bgcolor='white',
        xaxis=dict(
            tickangle=45,
            side='bottom'
        )
    )
    
    return fig
