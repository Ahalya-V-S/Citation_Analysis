import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Topic Model Analysis",
                   page_icon="📊",
                   layout="wide")

st.title("Topic Model Analysis")

# Check if required data is available
if 'citation_data' not in st.session_state or 'topic_model_data' not in st.session_state:
    st.error(
        "Missing required datasets. Please upload both citation and topic model data in the main page."
    )
    st.stop()

# Get the data
citation_df = st.session_state['citation_data']
topic_model_df = st.session_state['topic_model_data']

# Create tabs
tab1, tab2, tab3 = st.tabs(
    ["Topic Weight Analysis", "Topic Weights and Citations", "Paper Explorer"])

# Identify available topic models
model_columns = [
    col for col in topic_model_df.columns
    if col not in ['ArticleID', 'CITATIONCOUNT', 'Article Id']
]
model_prefixes = ['LDA', 'HDP', 'CTM', 'DLDA', 'DHDP', 'DCTM']
model_types = []
topic_counts = []

for col in model_columns:
    for prefix in model_prefixes:
        if col.startswith(prefix):
            model_types.append(prefix)
            if '5' in col and 5 not in topic_counts:
                topic_counts.append(5)
            if '10' in col and 10 not in topic_counts:
                topic_counts.append(10)

model_types = sorted(list(set(model_types)))
topic_counts = sorted(list(set(topic_counts)))

if not model_types or not topic_counts:
    st.error(
        "No valid topic model data found. Expected columns like LDA5, DCTM5, etc."
    )
    st.stop()


def get_model_column(model_name, topic_model_df):
    """Helper function to detect the column for a given model."""
    if model_name in topic_model_df.columns:
        return model_name
    return None


with tab1:
    st.header("Topic Weight Analysis")

    # Diagnostic: Display topic_model_df info
    st.subheader("Data Inspection")
    st.write("Columns in topic_model_df:", topic_model_df.columns.tolist())
    st.write("Sample data (first 5 rows):")
    st.dataframe(topic_model_df.head())

    # Select model type and topic count
    col1, col2 = st.columns(2)
    with col1:
        selected_model = st.selectbox("Select topic model type",
                                      options=model_types)
    with col2:
        selected_topic_count = st.selectbox("Select number of topics",
                                            options=topic_counts)

    model_name = f"{selected_model}{selected_topic_count}"
    st.subheader(f"Topic Model: {model_name}")

    # Get model column
    model_col = get_model_column(model_name, topic_model_df)
    if not model_col:
        st.error(
            f"No data found for {model_name}. Available columns: {topic_model_df.columns.tolist()}"
        )
        st.stop()

    st.write(f"Detected column: {model_col}")

    # Validate topic weights
    if topic_model_df[model_col].dtype not in [
            'float64', 'float32'
    ] or not topic_model_df[model_col].between(0, 1).all():
        st.error(
            f"Column {model_col} contains invalid topic weights (expected floats between 0 and 1)."
        )
        st.write("Sample values:", topic_model_df[model_col].unique()[:10])
        st.stop()

    # Plot weight distribution
    fig = px.histogram(topic_model_df,
                       x=model_col,
                       nbins=50,
                       title=f"Distribution of Topic Weights ({model_name})",
                       labels={
                           model_col: "Topic Weight",
                           "count": "Number of Papers"
                       },
                       color_discrete_sequence=['#3366CC'])
    fig.update_layout(xaxis_title="Topic Weight",
                      yaxis_title="Number of Papers",
                      xaxis=dict(range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)

    # Weight statistics
    weight_stats = topic_model_df[model_col].describe()
    st.subheader("Topic Weight Statistics")
    st.write(pd.DataFrame(weight_stats).T)

    st.markdown("""
    **Interpretation:**

    - **Topic Weight Distribution**: Shows how topic weights are distributed across papers.
    - **Statistics**: Provides mean, standard deviation, and quartiles of the topic weights.
    """)

with tab2:
    st.header("Topic Weights and Citations")

    col1, col2 = st.columns(2)
    with col1:
        selected_model = st.selectbox("Select topic model type",
                                      options=model_types,
                                      key="tab2_model")
    with col2:
        selected_topic_count = st.selectbox("Select number of topics",
                                            options=topic_counts,
                                            key="tab2_topics")

    model_name = f"{selected_model}{selected_topic_count}"
    model_col = get_model_column(model_name, topic_model_df)

    if not model_col:
        st.error(
            f"No data found for {model_name}. Available columns: {topic_model_df.columns.tolist()}"
        )
        st.stop()

    st.write(f"Detected column: {model_col}")

    # Prepare data for correlation analysis
    if 'CITATIONCOUNT' in topic_model_df.columns:
        topic_citation_df = topic_model_df.copy()
    elif 'ArticleID' in topic_model_df.columns and 'Article Id' in citation_df.columns:
        topic_citation_df = pd.merge(topic_model_df,
                                     citation_df[['Article Id', 'Cited By']],
                                     left_on='ArticleID',
                                     right_on='Article Id',
                                     how='inner')
        topic_citation_df['CITATIONCOUNT'] = topic_citation_df['Cited By']
    else:
        st.error("Cannot link topic model data with citation counts.")
        st.stop()

    # Calculate correlation
    corr = topic_citation_df[model_col].corr(
        topic_citation_df['CITATIONCOUNT'])
    st.write(
        f"Correlation between {model_name} weights and citations: {corr:.3f}")

    # Plot correlation
    fig = px.scatter(topic_citation_df,
                     x=model_col,
                     y='CITATIONCOUNT',
                     title=f"Topic Weight vs. Citation Count ({model_name})",
                     labels={
                         model_col: "Topic Weight",
                         'CITATIONCOUNT': "Citation Count"
                     },
                     trendline="ols")
    st.plotly_chart(fig, use_container_width=True)

    # High vs. low cited papers
    st.subheader("Topic Weights in High vs. Low Cited Papers")
    citation_percentile = st.slider("Citation threshold percentile",
                                    min_value=10,
                                    max_value=90,
                                    value=75,
                                    step=5)

    high_threshold = np.percentile(topic_citation_df['CITATIONCOUNT'],
                                   citation_percentile)
    low_threshold = np.percentile(topic_citation_df['CITATIONCOUNT'],
                                  100 - citation_percentile)

    st.write(f"High citation threshold: ≥ {high_threshold} citations")
    st.write(f"Low citation threshold: ≤ {low_threshold} citations")

    high_cited_df = topic_citation_df[topic_citation_df['CITATIONCOUNT'] >=
                                      high_threshold]
    low_cited_df = topic_citation_df[topic_citation_df['CITATIONCOUNT'] <=
                                     low_threshold]

    st.write(f"Number of high-cited papers: {len(high_cited_df)}")
    st.write(f"Number of low-cited papers: {len(low_cited_df)}")

    high_weight_mean = high_cited_df[model_col].mean()
    low_weight_mean = low_cited_df[model_col].mean()

    comparison_df = pd.DataFrame({
        'Group': ['High-Cited', 'Low-Cited'],
        'Average Weight': [high_weight_mean, low_weight_mean],
        'Difference': [
            high_weight_mean - low_weight_mean,
            low_weight_mean - high_weight_mean
        ]
    })

    st.write("Topic weight comparison:")
    st.dataframe(comparison_df)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=['High-Cited', 'Low-Cited'],
               y=[high_weight_mean, low_weight_mean],
               marker_color=['#3366CC', '#FF9900']))
    fig.update_layout(
        title=f"Topic Weight: High vs. Low Cited Papers ({model_name})",
        xaxis_title="Citation Group",
        yaxis_title="Average Topic Weight")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Interpretation:**

    - **Correlation**: Shows how topic weights correlate with citation counts.
    - **Weight Comparison**: Compares average topic weights between high and low cited papers.
    """)

with tab3:
    st.header("Paper Explorer")

    col1, col2 = st.columns(2)
    with col1:
        selected_model = st.selectbox("Select topic model type",
                                      options=model_types,
                                      key="tab3_model")
    with col2:
        selected_topic_count = st.selectbox("Select number of topics",
                                            options=topic_counts,
                                            key="tab3_topics")

    model_name = f"{selected_model}{selected_topic_count}"
    model_col = get_model_column(model_name, topic_model_df)

    if not model_col:
        st.error(
            f"No data found for {model_name}. Available columns: {topic_model_df.columns.tolist()}"
        )
        st.stop()

    st.write(f"Detected column: {model_col}")

    paper_ids = topic_model_df['ArticleID'].tolist()
    paper_options = [
        f"Paper {paper_id} ({topic_model_df.loc[topic_model_df['ArticleID'] == paper_id, 'CITATIONCOUNT'].values[0]} citations)"
        for paper_id in paper_ids[:100]
    ]

    selected_paper = st.selectbox("Select a paper to analyze:",
                                  options=paper_options)
    selected_paper_id = int(selected_paper.split(" ")[1].split(" ")[0])

    paper_data = topic_model_df[topic_model_df['ArticleID'] ==
                                selected_paper_id]
    if len(paper_data) == 0:
        st.error(f"Paper ID {selected_paper_id} not found in dataset.")
        st.stop()

    paper_weight = paper_data[model_col].values[0]
    citations = paper_data['CITATIONCOUNT'].values[0]

    st.subheader("Paper Information")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Paper ID", selected_paper_id)
        st.metric("Citations", citations)
    with col2:
        st.metric(f"{model_name} Weight", f"{paper_weight:.3f}")

    st.subheader("Similar Papers")

    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) *
                                 np.linalg.norm(v2)) if np.linalg.norm(
                                     v1) > 0 and np.linalg.norm(v2) > 0 else 0

    similarities = []
    for idx, row in topic_model_df.iterrows():
        paper_id = row['ArticleID']
        if paper_id != selected_paper_id:
            sim = cosine_similarity([paper_weight], [row[model_col]])
            similarities.append({
                'Paper ID': paper_id,
                'Similarity': sim,
                'Citations': row['CITATIONCOUNT']
            })

    sim_df = pd.DataFrame(similarities).sort_values('Similarity',
                                                    ascending=False)
    st.write("Papers with similar topic weights:")
    st.dataframe(sim_df.head(10))

    fig = px.scatter(
        sim_df.head(50),
        x='Similarity',
        y='Citations',
        hover_name='Paper ID',
        title="Citations vs. Weight Similarity (50 most similar papers)")
    fig.add_trace(
        go.Scatter(x=[1.0],
                   y=[citations],
                   mode='markers',
                   marker=dict(color='red', size=12, symbol='star'),
                   name=f"Selected Paper ({selected_paper_id})"))
    st.plotly_chart(fig, use_container_width=True)

    similarity_citation_corr = sim_df['Similarity'].corr(sim_df['Citations'])
    st.write(
        f"Correlation between weight similarity and citations: {similarity_citation_corr:.3f}"
    )
