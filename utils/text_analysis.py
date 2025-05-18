import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re
from collections import Counter
import textstat


# Download necessary NLTK resources at module load
def download_nltk_resources():
    try:
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        return True
    except Exception as e:
        print(f"Failed to download NLTK resources: {str(e)}")
        return False


# Cache the resource download for Streamlit
try:
    import streamlit as st

    @st.cache_resource
    def cached_download_nltk_resources():
        return download_nltk_resources()

    nltk_resources_available = cached_download_nltk_resources()
except ImportError:
    nltk_resources_available = download_nltk_resources()

# Check if resources are available
if not nltk_resources_available:
    print(
        "Warning: Some NLTK resources may not be available. Using fallbacks.")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


# Create simple placeholder for named entity recognition
class SimplifiedNER:

    def __call__(self, text):

        class Doc:

            def __init__(self, text):
                self.text = text
                self.ents = []

        return Doc(text)


nlp = SimplifiedNER()


# Simplified POS tagger in case NLTK tagger fails
class SimplifiedPOSTagger:

    def __init__(self):
        # Default dictionary of common word types
        self.word_types = {
            'the': 'DT',
            'a': 'DT',
            'an': 'DT',  # Determiners
            'is': 'VB',
            'are': 'VB',
            'was': 'VB',
            'were': 'VB',  # Verbs
            'this': 'DT',
            'that': 'DT',
            'these': 'DT',
            'those': 'DT',  # Determiners
            'in': 'IN',
            'on': 'IN',
            'at': 'IN',
            'by': 'IN',  # Prepositions
            'and': 'CC',
            'or': 'CC',
            'but': 'CC',  # Conjunctions
            'we': 'PRP',
            'i': 'PRP',
            'you': 'PRP',
            'he': 'PRP',
            'she': 'PRP',
            'they': 'PRP',  # Pronouns
        }

    def tag(self, tokens):
        tagged = []
        for token in tokens:
            # Check if in our dictionary
            if token.lower() in self.word_types:
                tagged.append((token, self.word_types[token.lower()]))
            # Basic heuristics for common patterns
            elif token.endswith('ly'):
                tagged.append((token, 'RB'))  # Adverb
            elif token.endswith('ed'):
                tagged.append((token, 'VBD'))  # Past tense verb
            elif token.endswith('ing'):
                tagged.append((token, 'VBG'))  # Gerund verb
            elif token.endswith('s'):
                tagged.append((token, 'NNS'))  # Plural noun
            else:
                tagged.append((token, 'NN'))  # Default to noun
        return tagged


# Function to safely use POS tagger
def safe_pos_tag(tokens):
    try:
        return nltk.pos_tag(tokens)
    except Exception as e:
        print(f"NLTK POS tagger failed: {e}. Using simplified POS tagger.")
        tagger = SimplifiedPOSTagger()
        return tagger.tag(tokens)


def preprocess_text(text):
    """
    Preprocess text for analysis

    Parameters:
    -----------
    text : str
        Raw text content

    Returns:
    --------
    str
        Preprocessed text
    list
        Tokenized words
    list
        Tokenized sentences
    """
    # Convert to lowercase
    text = text.lower()

    # Remove special LaTeX-like formatting if present
    text = re.sub(r'\\\\', ' ', text)
    text = re.sub(r'\\[a-zA-Z]+(\{.*?\})*', ' ', text)

    # Remove URLs and email addresses
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)

    # Remove numbers and special characters, but keep periods for sentence tokenization
    text = re.sub(r'[^a-zA-Z\s\.]', ' ', text)

    # Tokenize into sentences
    try:
        sentences = sent_tokenize(text)
    except LookupError:
        # Fallback: simple sentence tokenization on periods, exclamations, and question marks
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s',
                             text)

    # Process each sentence to get clean tokens
    all_tokens = []

    # Try to get stopwords, or use a simple set if NLTK fails
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        stop_words = {
            "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
            "at", "by", "for", "with", "about", "against", "between", "into",
            "through", "during", "before", "after", "above", "below", "to",
            "from", "up", "down", "in", "out", "on", "off", "over", "under",
            "again", "further", "then", "once", "here", "there", "when",
            "where", "why", "how", "all", "any", "both", "each", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only", "own",
            "same", "so", "than", "too", "very", "s", "t", "can", "will",
            "just", "don", "should", "now"
        }

    for sentence in sentences:
        # Remove periods and tokenize
        clean_sentence = re.sub(r'[^\w\s]', ' ', sentence)
        try:
            tokens = word_tokenize(clean_sentence)
        except LookupError:
            # Fallback: simple word splitting
            tokens = clean_sentence.split()

        # Remove stopwords and lemmatize
        filtered_tokens = [
            lemmatizer.lemmatize(token) for token in tokens
            if token not in stop_words and len(token) > 1
        ]

        all_tokens.extend(filtered_tokens)

    # Reconstruct preprocessed text
    preprocessed_text = ' '.join(all_tokens)

    return preprocessed_text, all_tokens, sentences


def extract_text_features(text):
    """
    Extract linguistic features from text

    Parameters:
    -----------
    text : str
        Raw text content

    Returns:
    --------
    dict
        Dictionary of text features
    """
    # Preprocess the text
    preprocessed_text, tokens, sentences = preprocess_text(text)

    # Count tokens and sentences
    token_count = len(tokens)
    sentence_count = len(sentences)

    # Avoid division by zero
    if sentence_count == 0:
        sentence_count = 1
    if token_count == 0:
        token_count = 1

    # Calculate average sentence and word lengths
    avg_sentence_length = token_count / sentence_count
    word_lengths = [len(token) for token in tokens]
    avg_word_length = sum(word_lengths) / token_count if tokens else 0

    # Calculate word length distribution
    word_length_dist = Counter(word_lengths)

    # Calculate lexical diversity (unique words / total words)
    lexical_diversity = len(set(tokens)) / token_count if tokens else 0

    # Calculate readability scores
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)

    # Get most common words (excluding very common ones)
    most_common_words = Counter(tokens).most_common(20)

    # Use our simplified NER (actual NER will not work without spaCy)
    entities = []
    entity_counts = Counter()

    # Count specific parts of speech using our safe tagger
    pos_tags = safe_pos_tag(tokens)
    pos_counts = Counter([tag for _, tag in pos_tags])

    # Count nouns, verbs, adjectives, adverbs
    noun_count = sum(1 for _, tag in pos_tags if tag.startswith('NN'))
    verb_count = sum(1 for _, tag in pos_tags if tag.startswith('VB'))
    adj_count = sum(1 for _, tag in pos_tags if tag.startswith('JJ'))
    adv_count = sum(1 for _, tag in pos_tags if tag.startswith('RB'))

    # Extract keyphrases
    keyphrases = extract_keyphrases(text, n=20)

    # Calculate ratios
    noun_ratio = noun_count / token_count if token_count else 0
    verb_ratio = verb_count / token_count if token_count else 0
    adj_ratio = adj_count / token_count if token_count else 0
    adv_ratio = adv_count / token_count if token_count else 0

    # Compile features into a dictionary
    features = {
        'token_count':
        token_count,
        'sentence_count':
        sentence_count,
        'avg_sentence_length':
        avg_sentence_length,
        'avg_word_length':
        avg_word_length,
        'lexical_diversity':
        lexical_diversity,
        'flesch_reading_ease':
        flesch_reading_ease,
        'flesch_kincaid_grade':
        flesch_kincaid_grade,
        'noun_count':
        noun_count,
        'verb_count':
        verb_count,
        'adj_count':
        adj_count,
        'adv_count':
        adv_count,
        'noun_ratio':
        noun_ratio,
        'verb_ratio':
        verb_ratio,
        'adj_ratio':
        adj_ratio,
        'adv_ratio':
        adv_ratio,
        'most_common_words':
        most_common_words,
        'entity_counts':
        dict(entity_counts),
        'keyphrases':
        keyphrases,
        'word_length_dist':
        dict(word_length_dist),
        # Additional advanced metrics
        'technical_complexity':
        (avg_word_length * 0.7) + (flesch_kincaid_grade * 0.3),
        'formality_score':
        (noun_ratio * 0.5) + (adj_ratio * 0.3) + (adv_ratio * 0.2)
    }

    return features


def compare_papers_text_features(papers_features, citation_counts=None):
    """
    Compare text features across papers

    Parameters:
    -----------
    papers_features : dict
        Dictionary mapping paper IDs to their text features
    citation_counts : dict, optional
        Dictionary mapping paper IDs to their citation counts

    Returns:
    --------
    DataFrame
        Comparison of key text features across papers
    """
    comparison_data = []

    for paper_id, features in papers_features.items():
        paper_data = {
            'paper_id': paper_id,
            'token_count': features['token_count'],
            'sentence_count': features['sentence_count'],
            'avg_sentence_length': features['avg_sentence_length'],
            'avg_word_length': features['avg_word_length'],
            'lexical_diversity': features['lexical_diversity'],
            'flesch_reading_ease': features['flesch_reading_ease'],
            'flesch_kincaid_grade': features['flesch_kincaid_grade'],
            'noun_ratio': features['noun_ratio'],
            'verb_ratio': features['verb_ratio'],
            'adj_ratio': features['adj_ratio'],
            'adv_ratio': features['adv_ratio'],
        }

        # Add advanced metrics if available
        if 'technical_complexity' in features:
            paper_data['technical_complexity'] = features[
                'technical_complexity']

        if 'formality_score' in features:
            paper_data['formality_score'] = features['formality_score']

        # Add citation count if available
        if citation_counts and paper_id in citation_counts:
            paper_data['citation_count'] = citation_counts[paper_id]

        comparison_data.append(paper_data)

    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)

    return comparison_df


def analyze_common_keywords(papers_features, top_n=10):
    """
    Analyze common keywords across papers and their frequency

    Parameters:
    -----------
    papers_features : dict
        Dictionary mapping paper IDs to their text features
    top_n : int
        Number of top keywords to include

    Returns:
    --------
    tuple
        (keyword_df, keyword_frequency_df) - DataFrames containing keyword analysis
    """
    # Collect all keyphrases from all papers
    all_keyphrases = []
    paper_keyphrases = {}

    for paper_id, features in papers_features.items():
        if 'keyphrases' in features:
            keyphrases = features['keyphrases']
            paper_keyphrases[paper_id] = keyphrases
            all_keyphrases.extend(keyphrases)

    # Count keyphrase frequency across all papers
    keyphrase_counter = Counter(all_keyphrases)
    top_keyphrases = keyphrase_counter.most_common(top_n)

    # Create a dataframe of top keyphrases and their counts
    keyword_df = pd.DataFrame({
        'Keyword': [kw for kw, count in top_keyphrases],
        'Frequency': [count for kw, count in top_keyphrases]
    })

    # Create a matrix of papers x keyphrases
    # This shows which papers contain which keyphrases
    frequency_data = []

    for paper_id, keyphrases in paper_keyphrases.items():
        paper_data = {'paper_id': paper_id}
        keyphrases_set = set(keyphrases)

        for keyword, _ in top_keyphrases:
            paper_data[keyword] = 1 if keyword in keyphrases_set else 0

        frequency_data.append(paper_data)

    keyword_frequency_df = pd.DataFrame(frequency_data)

    return keyword_df, keyword_frequency_df


def extract_keyphrases(text, n=10):
    """
    Extract key phrases from text using a simple TF-IDF-like approach

    Parameters:
    -----------
    text : str
        Raw text content
    n : int
        Number of keyphrases to extract

    Returns:
    --------
    list
        List of extracted keyphrases
    """
    # Preprocess text
    preprocessed_text, tokens, _ = preprocess_text(text)

    # Get POS tags and extract noun phrases
    pos_tags = safe_pos_tag(tokens)

    # Generate bigrams and trigrams
    bigrams = [' '.join(tokens[i:i + 2]) for i in range(len(tokens) - 1)]
    trigrams = [' '.join(tokens[i:i + 3]) for i in range(len(tokens) - 2)]

    # Count frequencies
    unigram_freq = Counter(tokens)
    bigram_freq = Counter(bigrams)
    trigram_freq = Counter(trigrams)

    # Combine and sort by frequency
    all_phrases = []

    # Add top unigrams
    all_phrases.extend([(phrase, count)
                        for phrase, count in unigram_freq.most_common(n)])

    # Add top bigrams
    all_phrases.extend([(phrase, count)
                        for phrase, count in bigram_freq.most_common(n)])

    # Add top trigrams
    all_phrases.extend([(phrase, count)
                        for phrase, count in trigram_freq.most_common(n)])

    # Sort by frequency and take top n
    all_phrases.sort(key=lambda x: x[1], reverse=True)
    top_phrases = [phrase for phrase, _ in all_phrases[:n]]

    return top_phrases
