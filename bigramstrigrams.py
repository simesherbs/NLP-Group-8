from sklearn.feature_extraction.text import CountVectorizer
import csv
import pandas as pd
from nltk import PorterStemmer, sent_tokenize, word_tokenize
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import re
import sys
import os

stop_words = set(stopwords.words("english"))
ps = PorterStemmer()
# Load the movie corpus


punctuation_list = ['.', ',', '"', ":", "-", "--", ";", ".", "?", "!", "(", ")", "'", "’", "`"]

def is_all_punc(str):
    for chr in str:
        if chr not in punctuation_list:
            return False
    return True

def generate_ngrams(text, n):
    """
    Generate n-grams from text.
    :param text: A string (sentence or document).
    :param n: The n in n-gram (e.g., 2 for bigram, 3 for trigram).
    :return: A list of n-grams as tuples.
    """
    tokens = text.split()  # Split the text into words
    ngrams = []
    sent_text = sent_tokenize(text=text)
    for sentence in sent_text:
        tokens = word_tokenize(sentence)
        for i in range(len(tokens)-n+1):
            no_stop_words = True
            stemmed = []
            ngram = tokens[i:i+n]
            not_punc = True
            proper_possessive = True
            for unigram in ngram:
                if "'s" in ngram or "s'" in ngram:
                    if not(len(ngram) == 3 and (ngram[1] == "'s" or ngram[1] == "s'")):
                        proper_possessive = False
                elif unigram in punctuation_list or is_all_punc(unigram):
                    not_punc = False
                    break
                elif unigram.lower() in stop_words or unigram in [' ']:
                    no_stop_words = False
                    break
                else:
                    stemmed.append(PorterStemmer.stem(self=ps, word=str(unigram).lower()))
            if (no_stop_words and not_punc and proper_possessive and stemmed != []):
                ngrams.append(tuple(stemmed))

           
    return ngrams

def generate_ngram_file(corpusfile):

    df = pd.read_csv(corpusfile)

    # Ensure the 'overview' column exists and drop any rows where it's missing
    if 'overview' not in df.columns:
        raise ValueError("The CSV does not have an 'overview' column.")
    df = df.dropna(subset=['overview'])


    # Generate bigrams and trigrams for each overview and keep results per entry
    entries_bigrams_trigrams = []
    
    for i, row in df.iterrows():
        overview = row['overview']
        unigrams = generate_ngrams(overview, 1)
        bigrams = generate_ngrams(overview, 2)
        trigrams = generate_ngrams(overview, 3)

        entries_bigrams_trigrams.append({
            "index": i,
            "unigrams": unigrams,
            "bigrams": bigrams,
            "trigrams": trigrams,
            "ngrams":unigrams+bigrams+trigrams,
            'genres': row['genres']
        })

    # Save the bigrams and trigrams for each entry to a CSV file
    with open(os.path.splitext(corpusfile)[0]+'_ngrams.csv', 'w', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Index', 'Unigrams', 'Bigrams', 'Trigrams', 'Ngrams', 'Genres'])
        for entry in entries_bigrams_trigrams:
            writer.writerow([
                entry['index'],
                ", ".join([" ".join(unigram) for unigram in entry['unigrams']]),
                ", ".join([" ".join(bigram) for bigram in entry['bigrams']]),
                ", ".join([" ".join(trigram) for trigram in entry['trigrams']]),
                ", ".join([" ".join(ngram) for ngram in entry["ngrams"]]),
                entry['genres']
            ])
    
    
    return os.path.splitext(corpusfile)[0] + "_ngrams.csv"

