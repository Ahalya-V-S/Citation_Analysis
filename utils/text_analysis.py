import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re
from collections import Counter
import spacy
import textstat

# Download necessary NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Try to load spaCy model, use a simple fallback if it fails
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # Create simple placeholder for named entity recognition
    class SimplifiedNER:
        def __call__(self, text):
            class Doc:
                def __init__(self, text):
                    self.text = text
                    self.ents = []
            return Doc(text)
    nlp = SimplifiedNER()

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
    sentences = sent_tokenize(text)
    
    # Process each sentence to get clean tokens
    all_tokens = []
    stop_words = set(stopwords.words('english'))
    
    for sentence in sentences:
        # Remove periods and tokenize
        clean_sentence = re.sub(r'[^\w\s]', ' ', sentence)
        tokens = word_tokenize(clean_sentence)
        
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
    avg_word_length = sum(len(token) for token in tokens) / token_count if tokens else 0
    
    # Calculate lexical diversity (unique words / total words)
    lexical_diversity = len(set(tokens)) / token_count if tokens else 0
    
    # Calculate readability scores
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
    
    # Get most common words (excluding very common ones)
    most_common_words = Counter(tokens).most_common(20)
    
    # Use spaCy for named entity recognition if available
    try:
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        entity_counts = Counter([ent[1] for ent in entities])
    except:
        entities = []
        entity_counts = Counter()
    
    # Count specific parts of speech
    pos_tags = nltk.pos_tag(tokens)
    pos_counts = Counter([tag for _, tag in pos_tags])
    
    # Count nouns, verbs, adjectives, adverbs
    noun_count = sum(1 for _, tag in pos_tags if tag.startswith('NN'))
    verb_count = sum(1 for _, tag in pos_tags if tag.startswith('VB'))
    adj_count = sum(1 for _, tag in pos_tags if tag.startswith('JJ'))
    adv_count = sum(1 for _, tag in pos_tags if tag.startswith('RB'))
    
    # Compile features into a dictionary
    features = {
        'token_count': token_count,
        'sentence_count': sentence_count,
        'avg_sentence_length': avg_sentence_length,
        'avg_word_length': avg_word_length,
        'lexical_diversity': lexical_diversity,
        'flesch_reading_ease': flesch_reading_ease,
        'flesch_kincaid_grade': flesch_kincaid_grade,
        'noun_count': noun_count,
        'verb_count': verb_count,
        'adj_count': adj_count,
        'adv_count': adv_count,
        'noun_ratio': noun_count / token_count if token_count else 0,
        'verb_ratio': verb_count / token_count if token_count else 0,
        'adj_ratio': adj_count / token_count if token_count else 0,
        'adv_ratio': adv_count / token_count if token_count else 0,
        'most_common_words': most_common_words,
        'entity_counts': dict(entity_counts)
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
        
        # Add citation count if available
        if citation_counts and paper_id in citation_counts:
            paper_data['citation_count'] = citation_counts[paper_id]
        
        comparison_data.append(paper_data)
    
    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    return comparison_df

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
    
    # Generate bigrams and trigrams
    bigrams = [' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)]
    trigrams = [' '.join(tokens[i:i+3]) for i in range(len(tokens)-2)]
    
    # Count frequencies
    unigram_freq = Counter(tokens)
    bigram_freq = Counter(bigrams)
    trigram_freq = Counter(trigrams)
    
    # Combine and sort by frequency
    all_phrases = []
    
    # Add top unigrams
    all_phrases.extend([(phrase, count) for phrase, count in unigram_freq.most_common(n)])
    
    # Add top bigrams
    all_phrases.extend([(phrase, count) for phrase, count in bigram_freq.most_common(n)])
    
    # Add top trigrams
    all_phrases.extend([(phrase, count) for phrase, count in trigram_freq.most_common(n)])
    
    # Sort by frequency and take top n
    all_phrases.sort(key=lambda x: x[1], reverse=True)
    top_phrases = [phrase for phrase, _ in all_phrases[:n]]
    
    return top_phrases
