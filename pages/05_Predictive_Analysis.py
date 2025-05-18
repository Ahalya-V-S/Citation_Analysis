import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import RFECV
from scipy import stats

st.set_page_config(
    page_title="Predictive Analysis",
    page_icon="🔮",
    layout="wide"
)

st.title("Predictive Analysis of Citation Rates")

# Check if required data is available
if 'citation_data' not in st.session_state:
    st.error("No citation data available. Please upload data in the main page.")
    st.stop()

# Get the data
citation_df = st.session_state['citation_data']

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "Citation Prediction",
    "Feature Importance",
    "Paper Success Potential"
])

with tab1:
    st.header("Citation Count Prediction Model")
    
    # Create a combined feature dataset
    st.subheader("Prepare Data for Modeling")
    
    # Extract features from citation data
    features_df = pd.DataFrame()
    
    # Add paper ID
    if 'Article Id' in citation_df.columns:
        features_df['Article Id'] = citation_df['Article Id']
    
    # Add citation count
    features_df['Citation Count'] = citation_df['Cited By']
    
    # Extract year columns
    year_cols = [col for col in citation_df.columns if col.isdigit() and 1992 <= int(col) <= 2023]
    
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
    
    # Check if topic model data is available
    has_topic_data = 'topic_model_data' in st.session_state
    
    # Add topic model features if available
    if has_topic_data:
        topic_model_df = st.session_state['topic_model_data']
        
        # Check if ArticleID column exists in topic model data
        if 'ArticleID' in topic_model_df.columns and 'Article Id' in features_df.columns:
            # Rename ID column for merging
            topic_temp = topic_model_df.rename(columns={'ArticleID': 'Article Id'})
            
            # Get topic model columns (exclude ID and citation count)
            topic_cols = [col for col in topic_temp.columns 
                         if col != 'Article Id' and col != 'CITATIONCOUNT']
            
            # Merge topic data with features
            if topic_cols:
                topic_subset = topic_temp[['Article Id'] + topic_cols]
                features_df = pd.merge(features_df, topic_subset, on='Article Id', how='left')
                
                st.success(f"Added {len(topic_cols)} topic model features to prediction model.")
    
    # Check for text features in session state
    has_text_features = 'paper_text_features' in st.session_state
    
    if has_text_features:
        text_features_df = st.session_state['paper_text_features']
        
        # Merge text features if article IDs match
        if 'Article Id' in features_df.columns and 'paper_id' in text_features_df.columns:
            text_features_df = text_features_df.rename(columns={'paper_id': 'Article Id'})
            
            # Get text feature columns
            text_cols = [col for col in text_features_df.columns if col != 'Article Id']
            
            if text_cols:
                text_subset = text_features_df[['Article Id'] + text_cols]
                features_df = pd.merge(features_df, text_subset, on='Article Id', how='left')
                
                st.success(f"Added {len(text_cols)} text analysis features to prediction model.")
    
    # Display available features
    st.write(f"Total available features: {features_df.shape[1] - 2}")  # Subtract ID and target columns
    
    # Split into features and target
    X = features_df.drop(['Citation Count'], axis=1)
    if 'Article Id' in X.columns:
        X = X.drop(['Article Id'], axis=1)
    y = features_df['Citation Count']
    
    # Check if we have sufficient numeric features
    numeric_features = X.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_features) < 2:
        st.error("Not enough numeric features available for modeling. Please ensure the dataset contains at least 2 numeric features.")
        st.stop()
    
    # Keep only numeric features
    X = X[numeric_features]
    
    # Handle missing values
    X = X.fillna(0)
    
    # Set up prediction mode options
    prediction_mode = st.radio(
        "Prediction Target",
        ["Predict Total Citations", "Predict Early vs. Late Citation Ratio"],
        horizontal=True
    )
    
    if prediction_mode == "Predict Total Citations":
        # Standard citation count prediction
        target = "Citation Count"
        y_target = y
    else:
        # Calculate early vs late citation ratio if possible
        if 'Early Citations' in X.columns and len(year_cols) >= 6:
            # Define early (first 3 years) and late (years 4-6) periods
            early_years = year_cols[:3]
            late_years = year_cols[3:6]
            
            early_citations = citation_df[early_years].sum(axis=1)
            late_citations = citation_df[late_years].sum(axis=1)
            
            # Calculate ratio (avoid division by zero)
            ratio = early_citations / (late_citations + 1)
            
            # Use ratio as target
            target = "Early-to-Late Citation Ratio"
            y_target = ratio
            
            # Remove early citation features to avoid leakage
            if 'Early Citations' in X.columns:
                X = X.drop(['Early Citations'], axis=1)
            if 'First Year Citations' in X.columns:
                X = X.drop(['First Year Citations'], axis=1)
            
            st.info("Predicting the ratio of citations in years 1-3 to citations in years 4-6")
        else:
            st.error("Cannot predict early vs. late citation ratio. Dataset needs early citation data and at least 6 years of citation history.")
            st.stop()
    
    # Allow user to select model
    model_type = st.selectbox(
        "Select prediction model",
        ["Random Forest", "Gradient Boosting", "Linear Regression", "Ridge Regression", "Lasso Regression"]
    )
    
    # Set up model
    if model_type == "Random Forest":
        model = RandomForestRegressor(random_state=42)
    elif model_type == "Gradient Boosting":
        model = GradientBoostingRegressor(random_state=42)
    elif model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "Ridge Regression":
        model = Ridge(random_state=42)
    elif model_type == "Lasso Regression":
        model = Lasso(random_state=42)
    
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    # Train model button
    if st.button("Train Prediction Model"):
        with st.spinner("Training model..."):
            try:
                # Check for valid data
                if X.empty or len(y_target) == 0:
                    st.error("Empty dataset. Cannot train the model.")
                    st.stop()
                
                # Check for NaN values
                if X.isna().any().any() or np.isnan(y_target).any():
                    st.warning("Dataset contains NaN values. Filling missing values with zeros.")
                    X = X.fillna(0)
                    y_target = np.nan_to_num(y_target, nan=0.0)
                
                # Check for constant features
                constant_features = []
                for col in X.columns:
                    if X[col].nunique() <= 1:
                        constant_features.append(col)
                
                if constant_features:
                    st.warning(f"Removed {len(constant_features)} constant features that would cause errors in model training.")
                    X = X.drop(columns=constant_features)
                
                if X.shape[1] == 0:
                    st.error("No valid features remain after preprocessing. Cannot train the model.")
                    st.stop()
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_target, test_size=0.2, random_state=42
                )
                
                # Fit model
                pipeline.fit(X_train, y_train)
                
                # Make predictions
                y_pred = pipeline.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation with error handling
                try:
                    cv_scores = cross_val_score(pipeline, X, y_target, cv=min(5, len(X)), scoring='r2')
                except Exception as e:
                    st.warning(f"Cross-validation failed: {str(e)}")
                    cv_scores = np.array([r2])  # Use test set R² as fallback
                
                # Display metrics
                st.subheader("Model Performance")
                
                col1, col2, col3, col4 = st.columns(4)
            
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
                st.stop()
            
            with col1:
                st.metric("R² Score", f"{r2:.3f}")
            
            with col2:
                st.metric("RMSE", f"{rmse:.3f}")
            
            with col3:
                st.metric("MAE", f"{mae:.3f}")
            
            with col4:
                st.metric("5-Fold CV R²", f"{cv_scores.mean():.3f}")
            
            # Interpretation of metrics
            st.markdown("""
            **Interpretation of Metrics:**
            
            - **R² Score**: Proportion of variance explained by the model (higher is better, max 1.0)
            - **RMSE**: Root Mean Squared Error (lower is better)
            - **MAE**: Mean Absolute Error (lower is better)
            - **CV R²**: Cross-validated R² score (more reliable estimate of model performance)
            """)
            
            # Plot actual vs predicted
            st.subheader("Actual vs Predicted Values")
            
            fig = px.scatter(
                x=y_test,
                y=y_pred,
                labels={'x': f'Actual {target}', 'y': f'Predicted {target}'},
                title="Actual vs Predicted Values"
            )
            
            # Add perfect prediction line
            fig.add_trace(
                go.Scatter(
                    x=[y_test.min(), y_test.max()],
                    y=[y_test.min(), y_test.max()],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Perfect Prediction'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Save trained model and data in session state
            st.session_state['trained_model'] = pipeline
            st.session_state['model_features'] = X.columns.tolist()
            st.session_state['model_target'] = target
            st.session_state['feature_data'] = X
            st.session_state['target_data'] = y_target
            
            # Success message
            st.success("Model trained successfully!")

with tab2:
    st.header("Feature Importance Analysis")
    
    # Check if model is trained
    if 'trained_model' not in st.session_state:
        st.warning("Please train a prediction model in the 'Citation Prediction' tab first.")
        st.stop()
    
    # Get trained model and feature data
    pipeline = st.session_state['trained_model']
    feature_names = st.session_state['model_features']
    X = st.session_state['feature_data']
    y = st.session_state['target_data']
    target = st.session_state['model_target']
    
    # Extract model from pipeline
    model = pipeline.named_steps['model']
    
    # Get feature importance based on model type
    if hasattr(model, 'feature_importances_'):
        # Tree-based models (Random Forest, Gradient Boosting)
        importance_type = "Feature Importance"
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Linear models (Linear Regression, Ridge, Lasso)
        importance_type = "Coefficient Magnitude"
        importances = np.abs(model.coef_)
    else:
        st.error("Cannot extract feature importance from this model type.")
        st.stop()
    
    # Create dataframe with feature importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Display feature importance
    st.subheader(f"{importance_type} for {target} Prediction")
    
    # Create bar chart
    fig = px.bar(
        importance_df,
        x='Feature',
        y='Importance',
        title=f"{importance_type} for Predicting {target}",
        color='Importance',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis_title="Feature",
        yaxis_title=importance_type,
        xaxis={'categoryorder':'total descending', 'tickangle': 45}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display table
    st.write("Feature importance values:")
    st.dataframe(importance_df)
    
    # Alternative to SHAP - Feature importance breakdown
    st.subheader("Feature Impact Analysis")
    
    st.write("This analysis shows how each feature contributes to the prediction model.")
    
    # Basic feature importance visualization
    if st.button("Run Detailed Feature Analysis"):
        with st.spinner("Analyzing feature contributions..."):
            try:
                # Get trained model and feature names from session state
                pipeline = st.session_state['trained_model']
                feature_names = st.session_state['model_features']
                model = pipeline.named_steps['model']
                X = st.session_state['feature_data']
                
                # Scale features as the model would
                scaler = pipeline.named_steps['scaler']
                X_scaled = scaler.transform(X)
                
                # Create a more detailed visualization based on model type
                if hasattr(model, 'feature_importances_'):
                    # For tree-based models, create more detailed feature importance plot
                    importance_type = "Feature Importance"
                    importances = model.feature_importances_
                    feature_importance = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    # Create horizontal bar chart
                    fig = px.bar(
                        feature_importance,
                        y='Feature',
                        x='Importance',
                        orientation='h',
                        title="Feature Importance Breakdown",
                        color='Importance',
                        color_continuous_scale='viridis'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional insights for tree-based models
                    st.subheader("Model Insights")
                    if isinstance(model, RandomForestRegressor):
                        st.write(f"Number of trees: {model.n_estimators}")
                        st.write(f"Max depth: {model.max_depth or 'None (unlimited)'}")
                    elif isinstance(model, GradientBoostingRegressor):
                        st.write(f"Number of estimators: {model.n_estimators}")
                        st.write(f"Learning rate: {model.learning_rate}")
                        
                else:
                    # For linear models, show coefficient magnitude and direction
                    importance_type = "Coefficient Magnitude"
                    importances = np.abs(model.coef_)
                    coef = model.coef_
                    feature_importance = pd.DataFrame({
                        'Feature': feature_names,
                        'Coefficient': coef,
                        'Absolute Value': importances
                    }).sort_values('Absolute Value', ascending=False)
                    
                    # Create horizontal bar chart with positive/negative coloring
                    fig = px.bar(
                        feature_importance,
                        y='Feature',
                        x='Coefficient',
                        orientation='h',
                        title="Feature Coefficient Analysis",
                        color='Coefficient',
                        color_continuous_scale='RdBu_r'
                    )
                    
                    # Add vertical line at x=0
                    fig.add_shape(
                        type="line",
                        x0=0, y0=-0.5,
                        x1=0, y1=len(feature_names)-0.5,
                        line=dict(color="black", width=1, dash="dash")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interpretation for linear models
                    st.markdown("""
                    **Interpretation:**
                    - **Positive coefficients** (blue) indicate features that increase citation predictions
                    - **Negative coefficients** (red) indicate features that decrease citation predictions
                    - **Larger absolute values** indicate stronger influence on the model
                    """)
                
                # Success message
                st.success("Feature analysis completed!")
            except Exception as e:
                st.error(f"Error analyzing features: {str(e)}")
                
    # Interpretation
    st.markdown("""
    **Feature Impact Interpretation:**
    
    - Features at the top of the list have the strongest influence on predictions
    - The relative importance/coefficient values show how much each feature contributes
    """)
    
    # Feature exploration section
    if 'trained_model' in st.session_state and 'model_features' in st.session_state:
        st.subheader("Individual Feature Impact")
        
        # Get necessary data from session state
        model_features = st.session_state['model_features']
        target_data = st.session_state['target_data']
        X = st.session_state['feature_data']
        model_target = st.session_state['model_target']
        
        # Feature selector
        selected_feature = st.selectbox(
            "Select feature to explore",
            options=model_features
        )
        
        if selected_feature:
            try:
                # Create scatter plot
                fig = px.scatter(
                    x=X[selected_feature],
                    y=target_data,
                    labels={
                        'x': selected_feature,
                        'y': model_target
                    },
                    title=f"Impact of {selected_feature} on {model_target}",
                    trendline="ols"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate correlation
                corr = X[selected_feature].corr(target_data)
                direction = "positive" if corr > 0 else "negative"
                strength = "strong" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.3 else "weak"
                
                # Interpretation
                st.markdown(f"""
                **Feature Impact Interpretation for {selected_feature}:**
                
                This plot shows how {selected_feature} relates to {model_target} in the dataset.
                - The correlation is {strength} and {direction} (correlation coefficient: {corr:.3f})
                - The trend line shows the general relationship between this feature and citations
                - Each point represents an individual paper in the dataset
                """)
            except Exception as e:
                st.error(f"Error plotting feature relationship: {str(e)}")
    else:
        st.info("Train a model first to explore individual feature impacts.")

with tab3:
    st.header("Paper Success Potential Analysis")
    
    st.write("This tool helps analyze what factors might contribute to a paper's citation success potential.")
    
    # Check if model is trained
    if 'trained_model' not in st.session_state:
        st.warning("Please train a prediction model in the 'Citation Prediction' tab first.")
        st.stop()
    
    # Get model and feature information
    model_target = st.session_state['model_target']
    feature_names = st.session_state['model_features']
    
    # Create a simulator for exploring how changing features affects predicted success
    st.subheader("Feature Impact Simulator")
    
    st.write("Adjust feature values to see how they might affect citation potential:")
    
    # Select base paper (optional)
    base_paper_option = st.radio(
        "Start with:",
        ["Average paper values", "Select a specific paper"],
        horizontal=True
    )
    
    # Get feature data
    X = st.session_state['feature_data']
    
    # Create base feature values
    if base_paper_option == "Average paper values":
        # Use mean values
        base_values = X.mean().to_dict()
    else:
        # Let user select a paper
        if 'Article Id' in citation_df.columns:
            paper_options = citation_df['Article Id'].tolist()
            selected_paper = st.selectbox(
                "Select a paper as base",
                options=paper_options
            )
            
            # Get paper features
            if 'Article Id' in features_df.columns and selected_paper in features_df['Article Id'].values:
                paper_row = features_df[features_df['Article Id'] == selected_paper]
                base_values = {}
                
                for feature in feature_names:
                    if feature in paper_row.columns:
                        base_values[feature] = paper_row[feature].iloc[0]
                    else:
                        base_values[feature] = 0
            else:
                st.error("Could not find selected paper in feature data.")
                base_values = X.mean().to_dict()
        else:
            st.error("Cannot select specific paper: no paper IDs available.")
            base_values = X.mean().to_dict()
    
    # Create input widgets for selected features
    st.write("Adjust feature values:")
    
    # Determine which features to expose for adjustment
    if len(feature_names) > 10:
        # For models with many features, let user select which to adjust
        selected_features = st.multiselect(
            "Select features to adjust",
            options=feature_names,
            default=feature_names[:5]
        )
    else:
        selected_features = feature_names
    
    # Create a dictionary to store adjusted values
    adjusted_values = base_values.copy()
    
    # Create sliders for numeric features
    for feature in selected_features:
        # Get min, max, and mean values
        min_val = float(X[feature].min())
        max_val = float(X[feature].max())
        mean_val = float(X[feature].mean())
        
        # Set step size
        range_val = max_val - min_val
        step = range_val / 100 if range_val > 0 else 0.1
        
        # For boolean/binary features, use 0-1 range with 1.0 step
        if set(X[feature].unique()).issubset({0, 1}) or feature in ['Title Has Colon', 'Title Has Question']:
            min_val, max_val, step = 0.0, 1.0, 1.0
        
        # Create slider
        value = st.slider(
            feature,
            min_value=min_val,
            max_value=max_val,
            value=float(base_values[feature]),
            step=step
        )
        
        # Store adjusted value
        adjusted_values[feature] = value
    
    # Create a button to predict with adjusted values
    if st.button("Predict Citation Potential"):
        # Create input array for prediction
        input_array = np.array([adjusted_values[feature] for feature in feature_names]).reshape(1, -1)
        
        # Make prediction
        pipeline = st.session_state['trained_model']
        prediction = pipeline.predict(input_array)[0]
        
        # Display prediction
        st.subheader("Prediction Result")
        
        if model_target == "Citation Count":
            st.metric("Predicted Citations", f"{prediction:.1f}")
            
            # Compare to average
            avg_citations = citation_df['Cited By'].mean()
            pct_diff = ((prediction - avg_citations) / avg_citations) * 100
            
            st.write(f"This is {abs(pct_diff):.1f}% {'higher' if pct_diff > 0 else 'lower'} than the average citation count ({avg_citations:.1f}).")
            
            # Citation potential rating
            percentile_position = stats.percentileofscore(citation_df['Cited By'], prediction)
            
            # Determine rating
            if percentile_position >= 90:
                rating = "Excellent"
                rating_color = "green"
            elif percentile_position >= 75:
                rating = "Very Good"
                rating_color = "lightgreen"
            elif percentile_position >= 50:
                rating = "Good"
                rating_color = "blue"
            elif percentile_position >= 25:
                rating = "Fair"
                rating_color = "orange"
            else:
                rating = "Low"
                rating_color = "red"
            
            # Display rating
            st.markdown(f"<h3 style='color: {rating_color}'>Citation Potential: {rating}</h3>", unsafe_allow_html=True)
            st.write(f"This prediction is at the {percentile_position:.1f}th percentile of all papers in the dataset.")
        else:
            # For ratio prediction
            st.metric("Predicted Early-to-Late Citation Ratio", f"{prediction:.2f}")
            
            # Interpret ratio
            if prediction > 1.5:
                st.markdown("<h3 style='color: orange'>High initial impact, likely to decline</h3>", unsafe_allow_html=True)
                st.write("This paper is predicted to receive significantly more citations in early years compared to later years.")
            elif prediction < 0.5:
                st.markdown("<h3 style='color: green'>Sleeping beauty pattern</h3>", unsafe_allow_html=True)
                st.write("This paper is predicted to receive more citations in later years, possibly indicating delayed recognition.")
            else:
                st.markdown("<h3 style='color: blue'>Balanced citation pattern</h3>", unsafe_allow_html=True)
                st.write("This paper is predicted to receive relatively balanced citations between early and later years.")
        
        # What-if analysis
        st.subheader("Improvement Suggestions")
        
        # Identify top features that could be improved
        importance_improvements = []
        
        # Get feature importance
        model = pipeline.named_steps['model']
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importances = np.abs(model.coef_)
        else:
            importances = np.ones(len(feature_names))
        
        # Create a list of (feature, importance, current value, max value)
        for i, feature in enumerate(feature_names):
            if feature in selected_features:
                importance = importances[i]
                current = adjusted_values[feature]
                max_val = X[feature].max()
                min_val = X[feature].min()
                mean_val = X[feature].mean()
                
                # For positive correlation features, check if we can increase
                # For simplicity, assume higher values are better (this is a simplification)
                room_for_improvement = max_val - current
                
                if room_for_improvement > 0:
                    improvement_potential = importance * (room_for_improvement / (max_val - min_val) if max_val > min_val else 0)
                    importance_improvements.append((feature, importance, current, max_val, improvement_potential))
        
        # Sort by improvement potential
        importance_improvements.sort(key=lambda x: x[4], reverse=True)
        
        # Display top suggestions
        if importance_improvements:
            st.write("Consider improving these features to potentially increase citation impact:")
            
            for feature, importance, current, max_val, potential in importance_improvements[:3]:
                st.markdown(f"**{feature}**: Current value: {current:.2f}, Potential target: {max_val:.2f}")
        else:
            st.write("No specific improvement suggestions available.")
        
        # Prediction insights
        st.subheader("Prediction Insights")
        
        st.markdown("""
        **Important Factors for Citation Success:**
        
        - **Early Citations**: Papers that receive citations in their first years tend to accumulate more citations long-term
        - **Author Count**: More authors can lead to wider dissemination through larger networks
        - **Title Properties**: Clear, descriptive titles with appropriate length may attract more readers
        - **Topic Selection**: Certain research topics naturally attract more attention and citations
        
        **Remember**: While these predictions can guide expectations, citation counts are influenced by many factors including research quality, novelty, and broader scientific trends.
        """)

