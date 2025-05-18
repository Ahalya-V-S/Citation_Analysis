import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go

# Handle different package names for the Louvain community detection
try:
    import community as community_louvain
except ImportError:
    try:
        import community.community_louvain as community_louvain
    except ImportError:
        # Fallback implementation
        community_louvain = None

def create_citation_network(citation_df, column_type='yearly'):
    """
    Create a directed graph network from citation data
    
    Parameters:
    -----------
    citation_df : DataFrame
        DataFrame containing citation data with columns:
        - paper_id: Paper identifier
        - title: Paper title
        - citation_count: Number of citations
    column_type : str
        Type of citation columns to use ('yearly' or 'total')
    
    Returns:
    --------
    networkx.DiGraph
        Directed graph representing the citation network
    """
    G = nx.DiGraph()
    
    # Ensure the expected columns exist
    expected_cols = ['paper_id', 'title', 'citation_count']
    if not all(col in citation_df.columns for col in expected_cols):
        raise ValueError(f"Citation dataframe must contain columns: {expected_cols}")
    
    # Add nodes (papers)
    paper_ids = citation_df['paper_id'].tolist()
    
    # Add node attributes
    for _, row in citation_df.iterrows():
        paper_id = row['paper_id']
        G.add_node(paper_id, 
                   title=row['title'],
                   citation_count=row['citation_count'])
    
    # For future implementation: add edges from citing to cited papers
    # This would require a more detailed dataset that shows which papers cite which
    
    return G

def create_topic_similarity_network(topic_model_df, threshold=0.7):
    """
    Create an undirected graph where nodes are papers and edges represent topic similarity
    
    Parameters:
    -----------
    topic_model_df : DataFrame
        DataFrame containing topic model data
    threshold : float
        Similarity threshold for creating edges (0 to 1)
    
    Returns:
    --------
    networkx.Graph
        Undirected graph representing the topic similarity network
    """
    G = nx.Graph()
    
    # Get paper IDs and topic columns
    paper_ids = topic_model_df['ArticleID'].tolist()
    
    # Filter for topic model columns
    topic_model_prefixes = ['LDA', 'HDP', 'CTM', 'DLDA', 'DHDP', 'DCTM']
    topic_cols = [col for col in topic_model_df.columns 
                 if any(prefix in col for prefix in topic_model_prefixes)]
    
    # Add nodes (papers)
    for paper_id in paper_ids:
        citation_count = 0
        if 'CITATIONCOUNT' in topic_model_df.columns:
            paper_row = topic_model_df[topic_model_df['ArticleID'] == paper_id]
            if not paper_row.empty:
                citation_count = paper_row['CITATIONCOUNT'].values[0]
        
        G.add_node(paper_id, citation_count=citation_count)
    
    # Add edges based on cosine similarity of topic distributions
    for i, id1 in enumerate(paper_ids):
        vec1 = topic_model_df.loc[topic_model_df['ArticleID'] == id1, topic_cols].values.flatten()
        
        for j, id2 in enumerate(paper_ids[i+1:], i+1):
            vec2 = topic_model_df.loc[topic_model_df['ArticleID'] == id2, topic_cols].values.flatten()
            
            # Calculate cosine similarity
            similarity = cosine_similarity(vec1, vec2)
            
            # Add edge if similarity is above threshold
            if similarity >= threshold:
                G.add_edge(id1, id2, weight=similarity)
    
    return G

def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2) if norm_v1 > 0 and norm_v2 > 0 else 0

def calculate_katz_centrality(G, alpha=0.1, max_iter=1000):
    """
    Calculate Katz centrality for all nodes
    
    Parameters:
    -----------
    G : networkx.Graph
        Graph to analyze
    alpha : float
        Attenuation factor
    max_iter : int
        Maximum number of iterations
    
    Returns:
    --------
    dict
        Dictionary of nodes with Katz centrality as the value
    """
    try:
        # Calculate Katz centrality
        katz_centrality = nx.katz_centrality(G, alpha=alpha, max_iter=max_iter)
        return katz_centrality
    except:
        # If calculation fails (e.g., due to convergence issues), try a different alpha
        try:
            alpha = 0.01  # Try a smaller alpha
            katz_centrality = nx.katz_centrality(G, alpha=alpha, max_iter=max_iter)
            return katz_centrality
        except:
            # If still fails, return degree centrality as a fallback
            return nx.degree_centrality(G)

def calculate_harmonic_centrality(G):
    """
    Calculate harmonic centrality for all nodes
    
    Parameters:
    -----------
    G : networkx.Graph
        Graph to analyze
    
    Returns:
    --------
    dict
        Dictionary of nodes with harmonic centrality as the value
    """
    return nx.harmonic_centrality(G)

def detect_communities(G, resolution=1.0):
    """
    Detect communities in a graph using the Louvain algorithm
    
    Parameters:
    -----------
    G : networkx.Graph
        Graph to analyze
    resolution : float
        Resolution parameter for community detection
    
    Returns:
    --------
    dict
        Dictionary of nodes with community as the value
    """
    try:
        # Try to use community_louvain
        partition = community_louvain.best_partition(G, resolution=resolution)
        return partition
    except:
        # If community_louvain fails, use connected components as a simple alternative
        communities = {}
        for i, component in enumerate(nx.connected_components(G)):
            for node in component:
                communities[node] = i
        return communities

def analyze_network(G, include_communities=True):
    """
    Analyze a network and return various metrics
    
    Parameters:
    -----------
    G : networkx.Graph
        Graph to analyze
    include_communities : bool
        Whether to include community detection
    
    Returns:
    --------
    dict
        Dictionary of network metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['num_edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)
    
    # Connected components
    if not G.is_directed():
        metrics['num_components'] = nx.number_connected_components(G)
        largest_cc = max(nx.connected_components(G), key=len)
        metrics['largest_component_size'] = len(largest_cc)
        metrics['largest_component_ratio'] = len(largest_cc) / G.number_of_nodes()
    
    # Centrality measures
    metrics['degree_centrality'] = nx.degree_centrality(G)
    metrics['katz_centrality'] = calculate_katz_centrality(G)
    metrics['harmonic_centrality'] = calculate_harmonic_centrality(G)
    
    # Community detection
    if include_communities and not G.is_directed():
        metrics['communities'] = detect_communities(G)
        
        # Count communities
        community_sizes = {}
        for node, community in metrics['communities'].items():
            if community not in community_sizes:
                community_sizes[community] = 0
            community_sizes[community] += 1
        
        metrics['num_communities'] = len(community_sizes)
        metrics['community_sizes'] = community_sizes
    
    return metrics

def plot_network(G, node_color='community', node_size='centrality', centrality_type='katz',
                layout='spring', title="Network Visualization"):
    """
    Create a network visualization
    
    Parameters:
    -----------
    G : networkx.Graph
        Graph to visualize
    node_color : str
        Method to color nodes ('community', 'centrality', 'citations')
    node_size : str
        Method to size nodes ('centrality', 'citations', 'degree', 'uniform')
    centrality_type : str
        Type of centrality to use ('katz', 'harmonic', 'degree')
    layout : str
        Layout algorithm ('spring', 'circular', 'kamada_kawai', 'spectral')
    title : str
        Plot title
    
    Returns:
    --------
    fig
        Matplotlib figure
    """
    # Calculate layout
    if layout == 'spring':
        pos = nx.spring_layout(G)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    else:
        pos = nx.spring_layout(G)
    
    # Determine node sizes
    node_sizes = []
    if node_size == 'uniform':
        node_sizes = [300] * G.number_of_nodes()
    elif node_size == 'degree':
        node_sizes = [300 * (nx.degree_centrality(G)[node] + 0.05) for node in G.nodes()]
    elif node_size == 'citations':
        node_sizes = [300 * (G.nodes[node].get('citation_count', 0) / 10 + 0.1) for node in G.nodes()]
    elif node_size == 'centrality':
        if centrality_type == 'katz':
            centrality = calculate_katz_centrality(G)
        elif centrality_type == 'harmonic':
            centrality = calculate_harmonic_centrality(G)
        else:
            centrality = nx.degree_centrality(G)
        node_sizes = [3000 * (centrality[node] + 0.05) for node in G.nodes()]
    
    # Determine node colors
    node_colors = []
    if node_color == 'community':
        communities = detect_communities(G)
        node_colors = [communities[node] for node in G.nodes()]
    elif node_color == 'centrality':
        if centrality_type == 'katz':
            centrality = calculate_katz_centrality(G)
        elif centrality_type == 'harmonic':
            centrality = calculate_harmonic_centrality(G)
        else:
            centrality = nx.degree_centrality(G)
        node_colors = [centrality[node] for node in G.nodes()]
    elif node_color == 'citations':
        node_colors = [G.nodes[node].get('citation_count', 0) for node in G.nodes()]
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                          alpha=0.7, cmap=plt.cm.viridis)
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    
    # Add color bar if using continuous colors
    if node_color in ['centrality', 'citations']:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(
            vmin=min(node_colors), vmax=max(node_colors)))
        sm.set_array([])
        plt.colorbar(sm, label=node_color.capitalize())
    
    # Add title and remove axis
    plt.title(title)
    plt.axis('off')
    
    return plt.gcf()

def create_interactive_network_graph(G, metrics, title="Interactive Network Visualization"):
    """
    Create an interactive network visualization using Plotly
    
    Parameters:
    -----------
    G : networkx.Graph
        Graph to visualize
    metrics : dict
        Network metrics from analyze_network()
    title : str
        Plot title
    
    Returns:
    --------
    fig
        Plotly figure object
    """
    # Calculate layout
    pos = nx.spring_layout(G, seed=42)
    
    # Get node positions
    x_nodes = [pos[node][0] for node in G.nodes()]
    y_nodes = [pos[node][1] for node in G.nodes()]
    
    # Create node traces
    node_trace = go.Scatter(
        x=x_nodes, y=y_nodes,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=[],
            color=[],
            colorbar=dict(
                thickness=15,
                title='Katz Centrality',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)
        )
    )
    
    # Set node sizes and colors based on Katz centrality
    katz_centrality = metrics['katz_centrality']
    node_sizes = [50 * (katz_centrality[node] * 10 + 0.5) for node in G.nodes()]
    node_colors = [katz_centrality[node] for node in G.nodes()]
    
    # Create hover text
    node_text = []
    for node in G.nodes():
        text = f"Paper ID: {node}<br>"
        if 'title' in G.nodes[node]:
            text += f"Title: {G.nodes[node]['title']}<br>"
        if 'author' in G.nodes[node]:
            text += f"Author: {G.nodes[node]['author']}<br>"
        if 'citation_count' in G.nodes[node]:
            text += f"Citations: {G.nodes[node]['citation_count']}<br>"
        text += f"Katz Centrality: {katz_centrality[node]:.4f}<br>"
        text += f"Harmonic Centrality: {metrics['harmonic_centrality'][node]:.4f}<br>"
        if 'communities' in metrics:
            text += f"Community: {metrics['communities'][node]}"
        node_text.append(text)
    
    node_trace.marker.color = node_colors
    node_trace.marker.size = node_sizes
    node_trace.text = node_text
    
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    
    return fig

def analyze_community_citation_patterns(G, metrics):
    """
    Analyze citation patterns by community
    
    Parameters:
    -----------
    G : networkx.Graph
        Graph to analyze
    metrics : dict
        Network metrics from analyze_network()
    
    Returns:
    --------
    DataFrame
        DataFrame containing community citation statistics
    """
    if 'communities' not in metrics:
        return pd.DataFrame()
    
    communities = metrics['communities']
    
    # Group papers by community
    community_papers = {}
    for node, community in communities.items():
        if community not in community_papers:
            community_papers[community] = []
        community_papers[community].append(node)
    
    # Calculate statistics for each community
    community_stats = []
    for community, papers in community_papers.items():
        # Skip tiny communities
        if len(papers) < 3:
            continue
            
        # Get citation counts
        citation_counts = [G.nodes[paper].get('citation_count', 0) for paper in papers]
        
        # Calculate centrality statistics
        katz_values = [metrics['katz_centrality'][paper] for paper in papers]
        harmonic_values = [metrics['harmonic_centrality'][paper] for paper in papers]
        
        # Store statistics
        community_stats.append({
            'Community': community,
            'Size': len(papers),
            'Avg Citations': np.mean(citation_counts),
            'Median Citations': np.median(citation_counts),
            'Max Citations': max(citation_counts),
            'Min Citations': min(citation_counts),
            'Std Dev Citations': np.std(citation_counts),
            'Total Citations': sum(citation_counts),
            'Avg Katz Centrality': np.mean(katz_values),
            'Avg Harmonic Centrality': np.mean(harmonic_values),
        })
    
    return pd.DataFrame(community_stats)

def identify_bridge_papers(G, metrics, top_n=10):
    """
    Identify papers that act as bridges between communities
    
    Parameters:
    -----------
    G : networkx.Graph
        Graph to analyze
    metrics : dict
        Network metrics from analyze_network()
    top_n : int
        Number of top bridge papers to return
    
    Returns:
    --------
    DataFrame
        DataFrame containing the top bridge papers
    """
    if 'communities' not in metrics:
        return pd.DataFrame()
    
    # Calculate betweenness centrality
    betweenness = nx.betweenness_centrality(G)
    
    # Calculate the number of neighboring communities for each node
    communities = metrics['communities']
    neighbor_communities = {}
    
    for node in G.nodes():
        # Get node's community
        node_community = communities[node]
        
        # Get neighbor communities
        neighbor_comms = set()
        for neighbor in G.neighbors(node):
            neighbor_comm = communities[neighbor]
            if neighbor_comm != node_community:
                neighbor_comms.add(neighbor_comm)
        
        neighbor_communities[node] = len(neighbor_comms)
    
    # Combine metrics for bridge detection
    bridge_scores = []
    for node in G.nodes():
        # Skip nodes with no neighboring communities
        if neighbor_communities[node] == 0:
            continue
            
        # Calculate bridge score (combination of betweenness and neighboring communities)
        bridge_score = betweenness[node] * neighbor_communities[node]
        
        bridge_scores.append({
            'Paper ID': node,
            'Title': G.nodes[node].get('title', 'Unknown'),
            'Citations': G.nodes[node].get('citation_count', 0),
            'Community': communities[node],
            'Betweenness': betweenness[node],
            'Neighboring Communities': neighbor_communities[node],
            'Bridge Score': bridge_score,
            'Katz Centrality': metrics['katz_centrality'][node],
            'Harmonic Centrality': metrics['harmonic_centrality'][node]
        })
    
    # Convert to dataframe and sort
    bridge_df = pd.DataFrame(bridge_scores)
    bridge_df = bridge_df.sort_values('Bridge Score', ascending=False)
    
    return bridge_df.head(top_n)

def analyze_low_cited_papers_network_position(G, metrics, low_threshold=5):
    """
    Analyze the network position of low-cited papers
    
    Parameters:
    -----------
    G : networkx.Graph
        Graph to analyze
    metrics : dict
        Network metrics from analyze_network()
    low_threshold : int
        Threshold for defining low-cited papers
    
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    # Identify low-cited papers
    low_cited = [node for node in G.nodes() if G.nodes[node].get('citation_count', 0) <= low_threshold]
    
    if not low_cited:
        return {"error": "No low-cited papers found."}
    
    # Get centrality measures for low-cited papers
    katz_low = {node: metrics['katz_centrality'][node] for node in low_cited}
    harmonic_low = {node: metrics['harmonic_centrality'][node] for node in low_cited}
    
    # Get community distribution of low-cited papers
    community_distribution = {}
    if 'communities' in metrics:
        for node in low_cited:
            community = metrics['communities'][node]
            if community not in community_distribution:
                community_distribution[community] = 0
            community_distribution[community] += 1
    
    # Find low-cited papers with high centrality (potential "hidden gems")
    katz_sorted = sorted(katz_low.items(), key=lambda x: x[1], reverse=True)
    harmonic_sorted = sorted(harmonic_low.items(), key=lambda x: x[1], reverse=True)
    
    hidden_gems_katz = katz_sorted[:10]
    hidden_gems_harmonic = harmonic_sorted[:10]
    
    # Comparison with high-cited papers
    high_cited = [node for node in G.nodes() if G.nodes[node].get('citation_count', 0) > low_threshold]
    
    avg_katz_low = np.mean(list(katz_low.values())) if katz_low else 0
    avg_harmonic_low = np.mean(list(harmonic_low.values())) if harmonic_low else 0
    
    avg_katz_high = np.mean([metrics['katz_centrality'][node] for node in high_cited]) if high_cited else 0
    avg_harmonic_high = np.mean([metrics['harmonic_centrality'][node] for node in high_cited]) if high_cited else 0
    
    return {
        'low_cited_count': len(low_cited),
        'avg_katz_low': avg_katz_low,
        'avg_harmonic_low': avg_harmonic_low,
        'avg_katz_high': avg_katz_high,
        'avg_harmonic_high': avg_harmonic_high,
        'katz_diff_pct': ((avg_katz_high - avg_katz_low) / avg_katz_high) * 100 if avg_katz_high > 0 else 0,
        'harmonic_diff_pct': ((avg_harmonic_high - avg_harmonic_low) / avg_harmonic_high) * 100 if avg_harmonic_high > 0 else 0,
        'community_distribution': community_distribution,
        'hidden_gems_katz': hidden_gems_katz,
        'hidden_gems_harmonic': hidden_gems_harmonic
    }