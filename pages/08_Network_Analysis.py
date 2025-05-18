import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from utils.network_analysis import (create_citation_network, analyze_network,
                                    plot_network,
                                    create_interactive_network_graph,
                                    analyze_community_citation_patterns,
                                    identify_bridge_papers,
                                    analyze_low_cited_papers_network_position)

st.set_page_config(page_title="Citation Network Analysis",
                   page_icon="🕸️",
                   layout="wide")

st.title("🕸️ Citation Network Analysis for Low-Cited Papers")
st.write("""
This analysis tool identifies factors that contribute to low citation rates by examining 
the position and relationships of papers within the citation network. By comparing network 
centrality metrics with citation counts, we can discover undervalued papers and understand 
structural reasons for low citation rates.
""")

# Check if required data is available
if 'citation_data' not in st.session_state:
    st.error(
        "Missing citation dataset. Please upload citation data in the main page."
    )
    st.stop()

# Get the data
citation_df = st.session_state['citation_data']

# Define the column names to use (with fallbacks)
article_id_col = 'ArticleID' if 'ArticleID' in citation_df.columns else 'Article Id'
citation_col = 'CITATIONCOUNT' if 'CITATIONCOUNT' in citation_df.columns else 'Cited By'

st.write(
    f"Using column '{article_id_col}' for paper identifiers and '{citation_col}' for citation counts."
)

# Validate input data
if article_id_col not in citation_df.columns:
    st.error(
        f"Citation data must include a column for paper IDs: ArticleID or Article Id."
    )
    st.stop()
if citation_col not in citation_df.columns:
    st.error(
        f"Citation data must include a column for citation counts: CITATIONCOUNT or Cited By."
    )
    st.stop()

# Additional validation for citation_df
if citation_df.empty:
    st.error("Citation data is empty. Please upload valid data.")
    st.stop()
if citation_df[article_id_col].isnull().any():
    st.error(
        f"Article ID column contains missing values. Please ensure all rows have valid Article ID values."
    )
    st.stop()
if citation_df[article_id_col].duplicated().any():
    st.error(
        f"Duplicate Article ID values found. Please ensure Article IDs are unique."
    )
    st.stop()
if citation_df[citation_col].isnull().any():
    st.warning(
        f"Citation count column contains missing values. Filling with zeros.")
    citation_df[citation_col] = citation_df[citation_col].fillna(0)

# Display sample data for debugging
if st.checkbox("Show citation data sample"):
    st.write("Sample of citation data:")
    st.dataframe(citation_df.head())

# Create tabs - focusing only on centrality and community detection as requested
tab1, tab2 = st.tabs(["Centrality Analysis", "Community Detection"])

# Only use citation data for analysis as requested
st.sidebar.header("Network Parameters")
st.sidebar.info(
    "Using citation data only to analyze reasons for low citation rates.")

# Create citation network
with st.spinner("Building citation network..."):
    try:
        # Prepare citation_df with expected column names
        # Check if Title column exists, otherwise use article_id_col as title
        title_col = 'Title' if 'Title' in citation_df.columns else article_id_col

        # Create input dataframe with consistent column names expected by create_citation_network
        citation_input_df = pd.DataFrame()
        citation_input_df['paper_id'] = citation_df[article_id_col].astype(str)
        citation_input_df['title'] = citation_df[title_col].astype(str)
        citation_input_df['citation_count'] = citation_df[citation_col].astype(
            float)

        # Create the network
        G = create_citation_network(citation_input_df)

        # Log success
        st.success(
            f"Successfully created network with {G.number_of_nodes()} papers")
    except Exception as e:
        st.error(f"Failed to create citation network: {str(e)}")
        st.write("Debug info:")
        st.write(f"- Available columns: {citation_df.columns.tolist()}")
        st.write(f"- Used ID column: {article_id_col}")
        st.write(f"- Used citation column: {citation_col}")
        st.stop()

    # Validate graph
    if G.number_of_nodes() == 0:
        st.error("No papers in the citation network. Check citation data.")
        st.stop()

    # Calculate network metrics
    try:
        network_metrics = analyze_network(G)
    except Exception as e:
        st.error(f"Failed to analyze network: {str(e)}")
        st.stop()

    st.sidebar.write(f"Network has {G.number_of_nodes()} papers (nodes).")

# Tab 1: Centrality Analysis
with tab1:
    st.header("Paper Centrality Analysis")
    st.write("""
    ### Understanding Low Citation Rates Through Network Position
    
    This analysis identifies influential papers using centrality metrics:
    
    - **Katz Centrality**: Measures influence via connections to other influential papers.
    - **Harmonic Centrality**: Measures closeness to other papers via shortest paths.
    
    High centrality with low citations may indicate underappreciated papers that are 
    structurally important in the research landscape but haven't received due recognition.
    """)

    try:
        # Validate centrality metrics
        required_metrics = [
            'katz_centrality', 'harmonic_centrality', 'degree_centrality'
        ]
        if not all(metric in network_metrics for metric in required_metrics):
            st.error(
                "Missing centrality metrics. Ensure analyze_network returns proper metrics."
            )
            st.stop()

        katz_centrality = network_metrics['katz_centrality']
        harmonic_centrality = network_metrics['harmonic_centrality']
        degree_centrality = network_metrics['degree_centrality']

        # Create centrality dataframe
        paper_centrality = []
        for node in G.nodes():
            try:
                paper_centrality.append({
                    'Paper ID':
                    str(node),
                    'Title':
                    G.nodes[node].get('title', 'Unknown'),
                    'Citations':
                    int(G.nodes[node].get('citation_count', 0)),
                    'Katz Centrality':
                    float(katz_centrality.get(node, 0)),
                    'Harmonic Centrality':
                    float(harmonic_centrality.get(node, 0)),
                    'Degree Centrality':
                    float(degree_centrality.get(node, 0))
                })
            except Exception as e:
                continue

        if not paper_centrality:
            st.error("No valid centrality data for papers.")
            st.stop()

        centrality_df = pd.DataFrame(paper_centrality)

        # Top papers by Katz Centrality
        st.subheader("Top Papers by Network Influence (Katz Centrality)")
        st.write(
            "These papers are the most influential in the network, connected to other important papers."
        )
        katz_top = centrality_df.sort_values('Katz Centrality',
                                             ascending=False).head(10)
        st.dataframe(
            katz_top[['Paper ID', 'Title', 'Citations', 'Katz Centrality']])

        # Top papers by Harmonic Centrality
        st.subheader("Top Papers by Network Position (Harmonic Centrality)")
        st.write(
            "These papers have strategic positions with short paths to other papers, acting as knowledge bridges."
        )
        harmonic_top = centrality_df.sort_values('Harmonic Centrality',
                                                 ascending=False).head(10)
        st.dataframe(harmonic_top[[
            'Paper ID', 'Title', 'Citations', 'Harmonic Centrality'
        ]])

        # Compare Katz and Harmonic centrality
        st.subheader("Network Position vs. Citation Count")
        st.write("""
        This scatter plot reveals the relationship between network position and citation counts.
        Papers in the upper left are influential in the network but have few citations.
        """)

        # Create scatter plot for centrality vs. citations
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=centrality_df['Katz Centrality'],
                       y=centrality_df['Citations'],
                       mode='markers',
                       name='Papers',
                       text=centrality_df['Title'],
                       marker=dict(
                           size=8,
                           color=centrality_df['Harmonic Centrality'],
                           colorscale='Viridis',
                           showscale=True,
                           colorbar=dict(title='Harmonic Centrality'))))

        fig.update_layout(title="Network Influence vs. Citation Count",
                          xaxis_title="Katz Centrality (Network Influence)",
                          yaxis_title="Citation Count",
                          hoverlabel=dict(bgcolor="white", font_size=12),
                          hovermode="closest")

        st.plotly_chart(fig, use_container_width=True)

        # Identify potential "hidden gems" - high centrality but low citations
        st.subheader("Potentially Underrecognized Papers")
        st.write("""
        These papers have high network centrality but relatively low citation counts.
        They may be underappreciated but structurally important to the research field.
        """)

        # Define ratio of centrality to citations
        centrality_df['Katz_per_Citation'] = centrality_df[
            'Katz Centrality'] / (centrality_df['Citations'] + 1)
        centrality_df['Harmonic_per_Citation'] = centrality_df[
            'Harmonic Centrality'] / (centrality_df['Citations'] + 1)

        # Filter for papers with significant centrality
        min_centrality = np.percentile(centrality_df['Katz Centrality'], 50)
        potential_gems = centrality_df[centrality_df['Katz Centrality'] >=
                                       min_centrality]

        # Sort by Katz per citation ratio
        # Create a copy of the dataframe and sort it
        potential_gems_sorted = potential_gems.copy()

        # Sort by Katz_per_Citation in descending order using a safer approach
        try:
            katz_gems = potential_gems_sorted.sort_values(
                by='Katz_per_Citation', ascending=False).head(10)
        except Exception as e:
            st.warning(f"Sorting error: {str(e)}. Using unsorted data.")
            katz_gems = potential_gems_sorted.head(10)
        st.dataframe(katz_gems[[
            'Paper ID', 'Title', 'Citations', 'Katz Centrality',
            'Katz_per_Citation'
        ]])

        # Analyze low-cited papers in the network
        st.subheader("Low Citation Analysis")

        citation_threshold = st.slider(
            "Low citation threshold:",
            min_value=0,
            max_value=50,
            value=5,
            step=1,
            help=
            "Papers with citations at or below this threshold are considered low-cited."
        )

        low_cited_analysis = analyze_low_cited_papers_network_position(
            G, network_metrics, low_threshold=citation_threshold)

        if 'error' not in low_cited_analysis:
            st.write(
                f"Found {low_cited_analysis['low_cited_count']} papers with {citation_threshold} or fewer citations."
            )

            # Compare centrality metrics
            col1, col2 = st.columns(2)
            with col1:
                # Ensure values are Python native types for Streamlit metrics
                avg_katz_low = float(low_cited_analysis['avg_katz_low'])
                katz_diff_pct = float(low_cited_analysis['katz_diff_pct'])

                st.metric("Avg. Katz Centrality (Low-cited)",
                          f"{avg_katz_low:.4f}",
                          delta=f"{-katz_diff_pct:.1f}%"
                          if katz_diff_pct > 0 else None,
                          delta_color="inverse")
            with col2:
                # Convert to Python native float
                avg_katz_high = float(low_cited_analysis['avg_katz_high'])
                st.metric("Avg. Katz Centrality (Others)",
                          f"{avg_katz_high:.4f}")

            st.write("### Low-Cited Papers with High Network Influence")
            st.write("""
            These papers have few citations but high centrality, suggesting they are more influential 
            than their citation count indicates. They represent potentially valuable but overlooked research.
            """)

            # Display top low-cited papers by Katz centrality
            hidden_gems = low_cited_analysis['hidden_gems_katz']
            hidden_gems_data = []

            for paper_id, centrality in hidden_gems:
                title = G.nodes[paper_id].get('title', 'Unknown')
                citations = G.nodes[paper_id].get('citation_count', 0)
                hidden_gems_data.append({
                    'Paper ID': paper_id,
                    'Title': title,
                    'Citations': citations,
                    'Katz Centrality': centrality
                })

            st.dataframe(pd.DataFrame(hidden_gems_data))
        else:
            st.info(low_cited_analysis['error'])
    except Exception as e:
        st.error(f"Error in centrality analysis: {str(e)}")

# Tab 2: Community Detection
with tab2:
    st.header("Research Community Analysis")

    st.write("""
    ### Understanding Low Citation Through Community Structure
    
    This analysis identifies communities of papers that form distinct research clusters.
    Papers in smaller or isolated communities may receive fewer citations due to limited
    visibility within the broader research landscape.
    
    Community detection helps understand:
    - Which research clusters receive the most attention
    - How isolated low-cited papers are from high-impact communities
    - Whether low citation rates correlate with community membership
    """)

    if 'communities' in network_metrics:
        # Get community data
        communities = network_metrics['communities']
        community_sizes = network_metrics['community_sizes']

        # Display community stats
        st.subheader("Community Statistics")

        col1, col2 = st.columns(2)
        with col1:
            # Ensure num_communities is a Python native type
            num_communities = int(
                network_metrics['num_communities']) if isinstance(
                    network_metrics['num_communities'],
                    (np.integer,
                     np.floating)) else network_metrics['num_communities']
            st.metric("Number of Communities", num_communities)
        with col2:
            largest_community = max(community_sizes.items(),
                                    key=lambda x: x[1])
            # Convert potential numpy types to native Python types
            community_size = int(largest_community[1]) if isinstance(
                largest_community[1],
                (np.integer, np.floating)) else largest_community[1]
            node_count = int(G.number_of_nodes()) if isinstance(
                G.number_of_nodes(),
                (np.integer, np.floating)) else G.number_of_nodes()

            st.metric("Largest Community Size", community_size,
                      f"{community_size/node_count:.1%} of all papers")

        # Create dataframe for community sizes
        community_df = pd.DataFrame({
            'Community': list(community_sizes.keys()),
            'Size': list(community_sizes.values())
        }).sort_values('Size', ascending=False)

        # Two-column layout for community size analysis
        cols = st.columns([3, 2])

        with cols[0]:
            # Bar chart of community sizes
            st.write("### Community Size Distribution")
            fig = go.Figure(
                go.Bar(x=community_df['Community'],
                       y=community_df['Size'],
                       marker_color='royalblue',
                       text=community_df['Size'],
                       textposition='auto'))

            fig.update_layout(title="Number of Papers per Community",
                              xaxis_title="Community ID",
                              yaxis_title="Number of Papers")

            st.plotly_chart(fig, use_container_width=True)

        with cols[1]:
            # Community size pie chart
            st.write("### Relative Community Sizes")
            fig = px.pie(community_df,
                         values='Size',
                         names='Community',
                         hole=0.4,
                         title="Distribution of Papers Across Communities")
            fig.update_traces(textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

        # Community Network Visualization
        st.subheader("Community Network Visualization")
        st.write("""
        The visualization below shows how papers are grouped into communities. 
        Each color represents a different community, and the size of each node represents its citation count.
        This helps identify if low-cited papers are concentrated in specific communities or spread across the network.
        """)

        # Create an interactive community-colored network visualization
        with st.spinner("Generating network visualization..."):
            try:
                # Create positions for nodes using spring layout
                pos = nx.spring_layout(G, seed=42)

                # Get node positions
                node_x = [pos[node][0] for node in G.nodes()]
                node_y = [pos[node][1] for node in G.nodes()]

                # Get community colors
                colors = []
                for node in G.nodes():
                    comm = communities.get(node, 0)
                    colors.append(comm)

                # Get node sizes based on citation count
                node_sizes = []
                for node in G.nodes():
                    cite_count = G.nodes[node].get('citation_count', 1)
                    # Use log scale for better visualization, with minimum size
                    node_sizes.append(max(5, min(50,
                                                 5 * np.log1p(cite_count))))

                # Create hover text
                hover_texts = []
                for node in G.nodes():
                    title = G.nodes[node].get('title', 'Unknown')
                    cite_count = G.nodes[node].get('citation_count', 0)
                    comm = communities.get(node, 'Unknown')
                    hover_texts.append(
                        f"ID: {node}<br>Title: {title[:30]}...<br>Citations: {cite_count}<br>Community: {comm}"
                    )

                # Create node trace
                node_trace = go.Scatter(x=node_x,
                                        y=node_y,
                                        mode='markers',
                                        hoverinfo='text',
                                        text=hover_texts,
                                        marker=dict(size=node_sizes,
                                                    color=colors,
                                                    colorscale='Viridis',
                                                    colorbar=dict(
                                                        thickness=15,
                                                        title='Community',
                                                        xanchor='left',
                                                        titleside='right'),
                                                    line=dict(width=0.5,
                                                              color='white')))

                # Create the figure
                fig = go.Figure(data=[node_trace],
                                layout=go.Layout(
                                    title='Paper Communities Network',
                                    titlefont=dict(size=16),
                                    showlegend=False,
                                    width=800,
                                    height=600,
                                    margin=dict(b=20, l=5, r=5, t=40),
                                    xaxis=dict(showgrid=False,
                                               zeroline=False,
                                               showticklabels=False),
                                    yaxis=dict(showgrid=False,
                                               zeroline=False,
                                               showticklabels=False),
                                    hovermode='closest'))

                st.plotly_chart(fig, use_container_width=True)

                # Add insights about the visualization
                st.info("""
                **How to interpret this visualization:**
                
                - **Clusters**: Closely positioned nodes likely represent papers with similar research topics
                - **Node Size**: Larger nodes have more citations
                - **Node Color**: Nodes with the same color belong to the same community
                - **Isolation**: Papers that are far from other communities may suffer from reduced visibility
                - **Bridge Papers**: Nodes at the boundaries between communities often connect different research areas
                """)

            except Exception as e:
                st.error(f"Error generating network visualization: {str(e)}")
                st.write(
                    "This might be due to the network structure or size limitations."
                )

        # Analyze citation patterns by community
        st.subheader("Citation Patterns by Community")
        st.write("""
        This analysis examines how citation counts vary across different communities.
        Communities with lower average citations may represent emerging or niche research areas.
        """)

        community_citation_df = analyze_community_citation_patterns(
            G, network_metrics)

        if not community_citation_df.empty:
            # Sort by average citations
            community_citation_df = community_citation_df.sort_values(
                'Avg Citations', ascending=False)

            # Display table
            st.dataframe(community_citation_df)

            # Create bar chart for citations by community
            fig = go.Figure()

            fig.add_trace(
                go.Bar(x=community_citation_df['Community'],
                       y=community_citation_df['Avg Citations'],
                       name='Average Citations',
                       marker_color='royalblue'))

            fig.add_trace(
                go.Bar(x=community_citation_df['Community'],
                       y=community_citation_df['Median Citations'],
                       name='Median Citations',
                       marker_color='lightseagreen'))

            fig.update_layout(title="Citation Metrics by Community",
                              xaxis_title="Community",
                              yaxis_title="Citations",
                              barmode='group',
                              legend=dict(orientation="h",
                                          yanchor="bottom",
                                          y=1.02,
                                          xanchor="right",
                                          x=1))

            st.plotly_chart(fig, use_container_width=True)

            # Low citation analysis by community
            st.subheader("Low Citation Analysis by Community")

            low_threshold = st.slider("Define low citation threshold:",
                                      min_value=0,
                                      max_value=30,
                                      value=5)

            # Calculate percentage of low-cited papers in each community
            low_cited_by_community = {}
            for comm_id, papers in community_sizes.items():
                comm_papers = [
                    node for node in G.nodes()
                    if node in communities and communities[node] == comm_id
                ]
                low_cited_count = sum(
                    1 for node in comm_papers
                    if G.nodes[node].get('citation_count', 0) <= low_threshold)
                if comm_papers:
                    low_cited_by_community[comm_id] = (low_cited_count /
                                                       len(comm_papers)) * 100

            # Create dataframe
            if low_cited_by_community:
                low_cited_df = pd.DataFrame({
                    'Community':
                    list(low_cited_by_community.keys()),
                    'Low Cited %':
                    list(low_cited_by_community.values()),
                    'Community Size': [
                        community_sizes.get(comm_id, 0)
                        for comm_id in low_cited_by_community.keys()
                    ]
                }).sort_values('Low Cited %', ascending=False)

                st.write(
                    f"Distribution of papers with {low_threshold} or fewer citations across communities:"
                )
                st.dataframe(low_cited_df)

                # Create visualization
                fig = px.scatter(
                    low_cited_df,
                    x='Community Size',
                    y='Low Cited %',
                    size='Community Size',
                    color='Low Cited %',
                    hover_name='Community',
                    color_continuous_scale='RdBu_r',
                    title=
                    f"Relationship Between Community Size and Low Citation Percentage (≤{low_threshold} citations)"
                )

                fig.update_layout(
                    xaxis_title="Community Size (Number of Papers)",
                    yaxis_title="Percentage of Low-Cited Papers")

                st.plotly_chart(fig, use_container_width=True)

                # Advanced analysis of low citation communities
                st.subheader("Detailed Community Analysis for Low Citation")

                # Create a two-column layout for the insights
                insight_cols = st.columns(2)

                with insight_cols[0]:
                    st.markdown("### General Patterns")
                    st.markdown("""
                    1. **Community Size Effect**: Smaller communities often have higher percentages of low-cited papers,
                       possibly due to limited visibility and smaller audience.
                    
                    2. **Isolation Factor**: Papers in communities with fewer connections to other communities tend
                       to have lower citation rates.
                    
                    3. **Emerging Research**: Communities with high percentages of low-cited papers may represent
                       emerging research areas that haven't yet gained mainstream attention.
                    """)

                with insight_cols[1]:
                    # Identify the community with the highest percentage of low-cited papers
                    highest_low_cited_community = low_cited_df.iloc[
                        0] if not low_cited_df.empty else None

                    if highest_low_cited_community is not None:
                        problem_community = highest_low_cited_community[
                            'Community']
                        problem_pct = highest_low_cited_community[
                            'Low Cited %']
                        st.markdown("### Focus Community Analysis")
                        st.markdown(f"""
                        Community **{problem_community}** has **{problem_pct:.1f}%** of papers with ≤{low_threshold} citations
                        
                        **Potential Reasons:**
                        
                        - **Specialized Terminology**: May use niche terms that reduce discoverability
                        - **Limited Audience**: Research area might appeal to a smaller audience
                        - **Research Direction**: Could represent an emerging or declining research direction
                        - **Network Position**: May lack strong connections to mainstream research topics
                        """)

                # Create visualization comparing community stats
                st.subheader("Community Comparison")

                # Create more advanced comparative metrics
                try:
                    # Calculate additional metrics for each community for comparison
                    community_metrics = []

                    for comm_id in low_cited_df['Community']:
                        # Get papers in this community
                        comm_papers = [
                            node for node in G.nodes() if node in communities
                            and communities[node] == comm_id
                        ]

                        # Calculate percentage of papers with different citation levels
                        zero_cite = sum(
                            1 for node in comm_papers
                            if G.nodes[node].get('citation_count', 0) == 0)
                        low_cite = sum(
                            1 for node in comm_papers if 0 < G.nodes[node].get(
                                'citation_count', 0) <= low_threshold)

                        # Calculate percentages
                        total_papers = len(comm_papers)
                        if total_papers > 0:
                            zero_pct = (zero_cite / total_papers) * 100
                            low_pct = (low_cite / total_papers) * 100
                            high_pct = 100 - zero_pct - low_pct

                            community_metrics.append({
                                'Community':
                                comm_id,
                                'Size':
                                community_sizes.get(comm_id, 0),
                                'Zero Citations %':
                                zero_pct,
                                'Low Citations %':
                                low_pct,
                                'Higher Citations %':
                                high_pct
                            })

                    if community_metrics:
                        metrics_df = pd.DataFrame(community_metrics)

                        # Create a stacked bar chart showing citation distribution by community
                        metrics_df = metrics_df.sort_values(
                            'Higher Citations %',
                            ascending=True)  # Sort for better visualization

                        # Create dataframe in format needed for stacked bar chart
                        plot_data = pd.DataFrame()
                        for index, row in metrics_df.iterrows():
                            plot_data = pd.concat([
                                plot_data,
                                pd.DataFrame({
                                    'Community': [row['Community']] * 3,
                                    'Citation Category': [
                                        'Zero Citations', 'Low Citations',
                                        'Higher Citations'
                                    ],
                                    'Percentage': [
                                        row['Zero Citations %'],
                                        row['Low Citations %'],
                                        row['Higher Citations %']
                                    ],
                                    'Size': [row['Size']] * 3
                                })
                            ])

                        # Create bar chart
                        fig = px.bar(
                            plot_data,
                            x='Community',
                            y='Percentage',
                            color='Citation Category',
                            color_discrete_map={
                                'Zero Citations': 'red',
                                'Low Citations': 'orange',
                                'Higher Citations': 'green'
                            },
                            title=
                            f"Citation Distribution by Community (Low Citation Threshold: {low_threshold})",
                            hover_data=['Size'])

                        fig.update_layout(xaxis_title="Community ID",
                                          yaxis_title="Percentage of Papers",
                                          legend_title="Citation Category",
                                          barmode='stack')

                        st.plotly_chart(fig, use_container_width=True)

                        # Add interpretation
                        st.info("""
                        **How to interpret this chart:**
                        
                        - **Red sections** show papers with zero citations - these need the most attention
                        - **Orange sections** show papers with low citations (but not zero)
                        - **Green sections** show papers with higher citations
                        
                        Communities with large red/orange sections may benefit the most from improved connectivity
                        to other research areas and better positioning in the citation network.
                        """)

                except Exception as e:
                    st.warning(
                        f"Could not generate comparative visualization: {str(e)}"
                    )

                # Add actionable recommendations
                st.subheader("Recommendations for Improving Citation Impact")

                st.markdown("""
                Based on community analysis, here are targeted strategies to improve citation rates:
                
                #### For Authors in Low-Cited Communities:
                
                1. **Increase Connectivity**
                   - Cite influential papers from larger, well-connected communities
                   - Collaborate with researchers from high-impact communities
                   - Frame research in terms relevant to broader research areas
                
                2. **Improve Discoverability**
                   - Use more standardized terminology alongside specialized terms
                   - Include keywords common in related high-citation communities
                   - Create bridge papers that connect specialized topics to mainstream research
                
                3. **Strategic Positioning**
                   - Highlight interdisciplinary aspects of research
                   - Position papers as extending or complementing high-impact work
                   - Create review papers that synthesize specialized and mainstream research
                """)

        # Bridge papers analysis
        st.subheader("Bridge Papers Analysis")
        st.write("""
        Bridge papers connect different research communities and can play critical roles
        in knowledge transfer. Papers that fail to act as bridges may receive fewer citations
        due to their limited reach across research areas.
        """)

        # Get bridge papers
        bridge_papers = identify_bridge_papers(G, network_metrics, top_n=10)

        if not bridge_papers.empty:
            st.dataframe(bridge_papers)

            st.write("""
            **Papers with Potentially Low Citations Due to Positioning:**
            
            1. **Community Isolation**: Papers that are deeply embedded within small, isolated communities
               and lack connections to other communities may receive fewer citations.
               
            2. **Missing Bridge Position**: Papers that could potentially bridge communities but fail to do so
               might miss citation opportunities from adjacent research areas.
               
            3. **Peripheral Position**: Papers positioned at the periphery of major communities rather than
               at the center may receive fewer citations despite their relevance.
            """)
    else:
        st.info(
            "Community detection requires a connected network with sufficient data."
        )

# Footer with guidance
st.markdown("---")
st.markdown("""
### How to Interpret Network Analysis Results for Low Citation

**Centrality vs. Citations Gap:**
- Papers with high centrality but low citations may be undervalued in the field
- These papers often represent important foundational or methodological contributions

**Community Positioning:**
- Papers in smaller or isolated communities tend to receive fewer citations
- Papers that bridge communities can overcome this limitation

**Recommendations for Addressing Low Citation:**
- Position papers to connect multiple research communities
- Identify and cite underappreciated but central papers in your field
- Consider both citation count and network position when evaluating research impact

The network position of a paper can be as important as its content in determining citation rates.
""")
