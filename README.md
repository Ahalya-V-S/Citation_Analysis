# Research Paper Citation Analysis Platform

This application analyzes research paper citation data to identify factors that influence citation rates. It provides multi-dimensional analysis through various analytical approaches presented in an interactive Streamlit interface.

## Features

- **Citation Trends Analysis**: Examine how citations evolve over time and identify patterns
- **Text Analysis**: Analyze linguistic features like readability, complexity, and word usage
- **Topic Modeling**: Explore topic distributions and their relationship with citations
- **Correlation Analysis**: Identify factors that correlate with higher citation counts
- **Predictive Analysis**: Use machine learning to predict citation potential

## Setup Instructions

1. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

2. **Download Required NLTK Data**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')
   ```

3. **Install spaCy English Language Model**:
   ```
   python -m spacy download en_core_web_sm
   ```

4. **Run the Application**:
   ```
   streamlit run app.py
   ```

## Data Requirements

The application requires three main data files:

1. **Citation Data CSV**: Contains article IDs, titles, authors, and yearly citation counts
2. **Topic Model Data CSV**: Contains topic distributions across papers
3. **Paper Text Files**: ZIP file containing the full text of papers

## Application Structure

- `app.py`: Main application file
- `pages/`: Streamlit pages for different analyses
- `utils/`: Utility functions for data processing and visualization
- `.streamlit/`: Streamlit configuration

## Dependencies

Main dependencies include:
- streamlit
- pandas
- numpy
- matplotlib
- plotly
- scikit-learn
- nltk
- spacy
- textstat
- wordcloud

## Creating requirements.txt

To generate the requirements.txt file based on the installed packages:

```
pip freeze > requirements.txt
```

## Developer Notes

- The application uses a multi-page Streamlit interface
- For large datasets, optimize memory usage by loading data only when needed
- Topic modeling is performed externally; this application visualizes the results