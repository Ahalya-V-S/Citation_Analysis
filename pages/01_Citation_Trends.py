import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from utils.citation_analysis import calculate_citation_metrics, analyze_citation_trends, identify_citation_patterns
from utils.visualization import plot_citation_distribution, plot_citation_trends, plot_citation_heatmap

st.set_page_config(
    page_title="Citation Trends Analysis",
    page_icon="📈",
    layout="wide"
)

st.title("Citation Trends Analysis")

# Check if citation data is available in session state
if 'citation_data' not in st.session_state:
    st.error("No citation data available. Please upload data in the main page.")
    st.stop()

# Get the citation data
citation_df = st.session_state['citation_data']

# Create tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs([
    "Citation Distribution", 
    "Citation Trends Over Time", 
    "Paper Comparison",
    "Citation Patterns"
])

with tab1:
    st.header("Citation Distribution Analysis")
    
    # Calculate citation metrics
    citation_metrics = calculate_citation_metrics(citation_df)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Papers", citation_metrics['total_papers'])
        st.metric("Mean Citations", f"{citation_metrics['mean_citations']:.2f}")
        st.metric("Zero Citations", f"{citation_metrics['zero_citation_percent']:.1f}%")
    
    with col2:
        st.metric("Total Citations", citation_metrics['total_citations'])
        st.metric("Median Citations", citation_metrics['median_citations'])
        st.metric("H-index", citation_metrics['h_index'])
    
    with col3:
        st.metric("Max Citations", citation_metrics['max_citations'])
        st.metric("95th Percentile", citation_metrics['percentile_95'])
        st.metric("Gini Coefficient", f"{citation_metrics['gini_coefficient']:.3f}")
    
    # Plot citation distribution
    st.subheader("Citation Count Distribution")
    
    # Allow choosing between normal and log scale
    scale_type = st.radio(
        "X-axis Scale:",
        ["Linear", "Logarithmic"],
        horizontal=True
    )
    
    if scale_type == "Logarithmic":
        # Use log scale for citation counts (add 1 to handle zeros)
        log_df = citation_df.copy()
        log_df['Log Citations'] = np.log1p(log_df['Cited By'])
        fig = plot_citation_distribution(log_df, title="Distribution of Citation Counts (Log Scale)")
        fig.update_layout(xaxis_title="Log(Citations + 1)")
    else:
        # Use normal scale
        fig = plot_citation_distribution(citation_df)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Citation Percentiles")
    
    # Create percentile table
    percentiles = pd.DataFrame({
        'Percentile': ['10%', '25%', '50% (Median)', '75%', '90%', '95%', '99%'],
        'Citation Count': [
            citation_metrics['percentile_10'],
            citation_metrics['percentile_25'],
            citation_metrics['percentile_50'],
            citation_metrics['percentile_75'],
            citation_metrics['percentile_90'],
            citation_metrics['percentile_95'],
            citation_metrics['percentile_99']
        ]
    })
    
    st.table(percentiles)
    
    # Interpretation
    st.subheader("Interpretation")
    st.markdown("""
    - **Gini Coefficient** measures inequality in citation distribution (0 = perfect equality, 1 = perfect inequality)
    - **H-index** is the number h such that h papers have at least h citations
    - **Zero citation percentage** shows what portion of papers have never been cited
    """)

with tab2:
    st.header("Citation Trends Over Time")
    
    # Analyze citation trends
    trends = analyze_citation_trends(citation_df)
    
    # Display key trend metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Median Time to First Citation", 
            f"{trends['median_time_to_first_citation']} years" if trends['median_time_to_first_citation'] is not None else "N/A"
        )
    
    with col2:
        st.metric(
            "Median Citation Half-Life", 
            f"{trends['median_citation_half_life']}" if trends['median_citation_half_life'] is not None else "N/A"
        )
    
    with col3:
        st.metric(
            "Most Common Peak Citation Year", 
            f"{trends['most_common_peak_year']}" if trends['most_common_peak_year'] is not None else "N/A"
        )
    
    # Plot options
    col1, col2 = st.columns(2)
    
    with col1:
        x_scale = st.radio(
            "X-axis Scale:",
            ["linear", "log"],
            horizontal=True,
            index=0
        )
    
    with col2:
        y_scale = st.radio(
            "Y-axis Scale:",
            ["linear", "log"],
            horizontal=True,
            index=0
        )
    
    # Plot citation trends
    fig = plot_citation_trends(citation_df, x_scale, y_scale)
    st.plotly_chart(fig, use_container_width=True)
    
    # Citation heatmap
    st.subheader("Citation Heatmap by Year")
    
    # Number of papers to include
    num_papers = st.slider(
        "Number of papers to display",
        min_value=10,
        max_value=50,
        value=20,
        step=5
    )
    
    # Plot heatmap
    heatmap_fig = plot_citation_heatmap(citation_df, num_papers)
    st.plotly_chart(heatmap_fig, use_container_width=True)
    
    # Interpretation
    st.subheader("Interpretation")
    st.markdown("""
    - **Time to First Citation**: How long papers typically wait for their first citation
    - **Citation Half-Life**: Year by which papers receive half of their total citations
    - **Peak Citation Year**: The year when papers typically receive the most citations
    - The **Citation Heatmap** shows the citation pattern for each paper across years
    """)

with tab3:
    st.header("Paper Comparison")
    
    # Allow selection of papers to compare
    paper_ids = citation_df['Article Id'].tolist() if 'Article Id' in citation_df.columns else citation_df.index.tolist()
    
    selected_papers = st.multiselect(
        "Select papers to compare",
        options=paper_ids,
        default=paper_ids[:5] if len(paper_ids) >= 5 else paper_ids
    )
    
    if selected_papers:
        # Filter to selected papers
        if 'Article Id' in citation_df.columns:
            selected_df = citation_df[citation_df['Article Id'].isin(selected_papers)]
        else:
            selected_df = citation_df.loc[selected_df.index.isin(selected_papers)]
        
        # Display paper details
        st.subheader("Selected Papers")
        paper_details = selected_df[['Title', 'Author', 'Cited By']] if 'Title' in selected_df.columns else selected_df[['Cited By']]
        st.dataframe(paper_details)
        
        # Extract year columns
        year_cols = [col for col in citation_df.columns if col.isdigit() and 1992 <= int(col) <= 2023]
        
        # Create yearly citation chart
        st.subheader("Citation Trends for Selected Papers")
        
        # Create dataframe for plotting
        plot_data = []
        for idx, row in selected_df.iterrows():
            paper_id = row.get('Article Id', idx)
            title = row.get('Title', paper_id)
            title_short = title[:30] + "..." if len(title) > 30 else title
            
            for year in year_cols:
                plot_data.append({
                    'Paper': title_short,
                    'Year': int(year),
                    'Citations': row[year]
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create line chart
        fig = px.line(
            plot_df,
            x='Year',
            y='Citations',
            color='Paper',
            title="Yearly Citation Trends",
            labels={'Citations': 'Citations', 'Year': 'Year'}
        )
        
        fig.update_layout(
            xaxis=dict(tickmode='linear', tick0=int(year_cols[0]), dtick=5),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create cumulative citation chart
        st.subheader("Cumulative Citations for Selected Papers")
        
        # Calculate cumulative citations
        cumulative_data = []
        for idx, row in selected_df.iterrows():
            paper_id = row.get('Article Id', idx)
            title = row.get('Title', paper_id)
            title_short = title[:30] + "..." if len(title) > 30 else title
            
            cumulative = 0
            for year in year_cols:
                cumulative += row[year]
                cumulative_data.append({
                    'Paper': title_short,
                    'Year': int(year),
                    'Cumulative Citations': cumulative
                })
        
        cumulative_df = pd.DataFrame(cumulative_data)
        
        # Create line chart
        fig = px.line(
            cumulative_df,
            x='Year',
            y='Cumulative Citations',
            color='Paper',
            title="Cumulative Citation Trends",
            labels={'Cumulative Citations': 'Total Citations', 'Year': 'Year'}
        )
        
        fig.update_layout(
            xaxis=dict(tickmode='linear', tick0=int(year_cols[0]), dtick=5),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select at least one paper to compare.")

with tab4:
    st.header("Citation Pattern Analysis")
    
    # Identify citation patterns
    patterns = identify_citation_patterns(citation_df)
    
    # Display pattern counts
    st.subheader("Distribution of Citation Patterns")
    
    pattern_counts = {
        'Pattern': ['Flash in the Pan', 'Sleeping Beauty', 'Steady Riser', 'Constant Performer'],
        'Count': [
            len(patterns['flash_in_pan']),
            len(patterns['sleeping_beauty']),
            len(patterns['steady_riser']),
            len(patterns['constant_performer'])
        ]
    }
    
    pattern_df = pd.DataFrame(pattern_counts)
    
    # Create bar chart
    fig = px.bar(
        pattern_df,
        x='Pattern',
        y='Count',
        title="Number of Papers by Citation Pattern",
        color='Pattern',
        color_discrete_sequence=['#FF9900', '#3366CC', '#66AA00', '#9370DB']
    )
    
    fig.update_layout(
        xaxis_title="Citation Pattern",
        yaxis_title="Number of Papers",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Explanation of patterns
    st.subheader("Pattern Descriptions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Flash in the Pan**
        - High initial citations
        - Quick drop in citations afterward
        - Short-lived impact
        
        **Sleeping Beauty**
        - Few initial citations
        - Significant rise in citations later
        - Delayed recognition
        """)
    
    with col2:
        st.markdown("""
        **Steady Riser**
        - Gradual increase in citations over time
        - Consistent growth in recognition
        - Long-term impact
        
        **Constant Performer**
        - Relatively steady citation rate
        - Consistent recognition over time
        - Sustained impact
        """)
    
    # Allow exploration of papers by pattern
    st.subheader("Explore Papers by Pattern")
    
    selected_pattern = st.selectbox(
        "Select a citation pattern",
        ["Flash in the Pan", "Sleeping Beauty", "Steady Riser", "Constant Performer"]
    )
    
    # Map selection to pattern key
    pattern_key_map = {
        "Flash in the Pan": 'flash_in_pan',
        "Sleeping Beauty": 'sleeping_beauty',
        "Steady Riser": 'steady_riser',
        "Constant Performer": 'constant_performer'
    }
    
    pattern_key = pattern_key_map[selected_pattern]
    pattern_papers = patterns[pattern_key]
    
    if pattern_papers:
        # Filter citation data to pattern papers
        if 'Article Id' in citation_df.columns:
            pattern_df = citation_df[citation_df['Article Id'].isin(pattern_papers)]
        else:
            pattern_df = citation_df.loc[citation_df.index.isin(pattern_papers)]
        
        # Display papers with this pattern
        st.write(f"Papers with {selected_pattern} pattern ({len(pattern_papers)} papers):")
        if 'Title' in pattern_df.columns:
            st.dataframe(pattern_df[['Article Id', 'Title', 'Author', 'Cited By']])
        else:
            st.dataframe(pattern_df[['Cited By']])
        
        # Plot average trend for this pattern
        year_cols = [col for col in citation_df.columns if col.isdigit() and 1992 <= int(col) <= 2023]
        
        avg_pattern = pattern_df[year_cols].mean()
        
        # Create DataFrame for plotting
        avg_data = pd.DataFrame({
            'Year': [int(year) for year in year_cols],
            'Average Citations': avg_pattern.values
        })
        
        # Create line chart
        fig = px.line(
            avg_data,
            x='Year',
            y='Average Citations',
            title=f"Average Citation Trend for {selected_pattern} Pattern",
            labels={'Average Citations': 'Average Citations', 'Year': 'Year'}
        )
        
        fig.update_layout(
            xaxis=dict(tickmode='linear', tick0=int(year_cols[0]), dtick=5)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No papers with {selected_pattern} pattern found.")
