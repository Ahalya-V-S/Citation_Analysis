# Sample Data Structure

This directory shows the expected format for data files used by the Citation Analysis Platform.

## Required Data Files

1. **Citation Data CSV** (`citation_data.csv`):
   - Must include the following columns:
     - `Article Id`: Unique identifier for each paper
     - `Title`: Paper title
     - `Author`: Author names (comma-separated)
     - `Cited By`: Total citation count
     - Year columns (e.g., `2010`, `2011`, etc.): Citation counts per year

   Example structure:
   ```
   Article Id,Title,Author,Cited By,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020
   paper1,"Machine Learning Methods",John Smith,320,5,12,18,25,30,40,45,50,45,30,20
   paper2,"Neural Networks",Jane Doe,180,0,2,5,10,15,20,30,35,28,20,15
   ```

2. **Topic Model Data CSV** (`topic_model.csv`):
   - Must include the following columns:
     - `ArticleID`: Matching the `Article Id` from citation data
     - `Topic1`, `Topic2`, etc.: Topic distribution scores for each paper
     - (Optional) `TopicName1`, `TopicName2`, etc.: Names of topics if available
     - (Optional) `CITATIONCOUNT`: Citation count (redundant with citation data)

   Example structure:
   ```
   ArticleID,Topic1,Topic2,Topic3,Topic4,Topic5,TopicName1,TopicName2,TopicName3,TopicName4,TopicName5
   paper1,0.45,0.25,0.15,0.10,0.05,"Machine Learning","Neural Networks","Data Mining","Statistics","Visualization"
   paper2,0.15,0.55,0.10,0.15,0.05,"Machine Learning","Neural Networks","Data Mining","Statistics","Visualization"
   ```

3. **Paper Text Files** (ZIP file):
   - A ZIP archive containing text files of papers
   - Each text file should be named with the paper's ID (matching `Article Id` from citation data)
   - Text files should contain the full text of the paper in plain text format

   Example structure inside ZIP:
   ```
   paper1.txt
   paper2.txt
   paper3.txt
   ...
   ```

## Data Loading Process

The application will prompt you to upload these files on the home page. Data will be stored in the session state and made available to all analysis pages.