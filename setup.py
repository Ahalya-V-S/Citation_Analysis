#!/usr/bin/env python3
"""
Setup script for Citation Analysis Platform.
This script installs required dependencies and downloads necessary NLTK data.
"""

import os
import subprocess
import sys

def install_dependencies():
    """Install required Python packages."""
    print("Installing required Python packages...")
    dependencies = [
        "matplotlib>=3.4.0",
        "nltk>=3.6.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "plotly>=5.0.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "seaborn>=0.11.0",
        "spacy>=3.0.0",
        "streamlit>=1.20.0",
        "textstat>=0.7.0",
        "wordcloud>=1.8.0"
    ]
    
    # Install each dependency
    for package in dependencies:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError:
            print(f"Error installing {package}. Please install it manually.")

def download_nltk_data():
    """Download required NLTK datasets."""
    print("Downloading NLTK data...")
    import nltk
    nltk_data = ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"]
    for data in nltk_data:
        print(f"Downloading NLTK dataset: {data}")
        nltk.download(data)

def install_spacy_model():
    """Download SpaCy English language model."""
    print("Installing SpaCy English language model...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    except subprocess.CalledProcessError:
        print("Error installing SpaCy model. Please install it manually with:")
        print("python -m spacy download en_core_web_sm")

def main():
    """Main setup function."""
    print("Setting up Citation Analysis Platform...")
    
    # Install Python dependencies
    install_dependencies()
    
    # Download NLTK data
    try:
        download_nltk_data()
    except ImportError:
        print("NLTK not installed properly. Please run the setup again after fixing.")
    
    # Install SpaCy model
    install_spacy_model()
    
    print("\nSetup completed!")
    print("To run the application, use: streamlit run app.py")

if __name__ == "__main__":
    main()