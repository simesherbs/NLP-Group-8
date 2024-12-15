from typing import TypedDict
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2
from numpy import ndarray
from sklearn.preprocessing import MinMaxScaler
import numpy as np
scaler = MinMaxScaler()
from functools import reduce
import math
import csv
from CValue import get_cval
from collections import defaultdict

genres_arr = sorted([
    "Drama",
    "Comedy",
    "Thriller",
    "Action",
    "Adventure",
    "Horror",
    "Romance",
    "Family",
    "Crime",
    "Science Fiction",
    "Fantasy",
    "Animation",
    "Mystery",
    "History",
    "Music",
    "TV Movie",
    "War",
    "Documentary",
    "Western",
])

class Genre(TypedDict):
    genre: str
    tfidf_scores: DataFrame
    atfidf: dict
    chi2: dict
    df: DataFrame
    terms: ndarray
    CS: list

def get_vocab(df: DataFrame):
    vocab = set()
    for ngram in df['Ngrams'].unique():
        temp = [t for t in str(ngram).split(", ") if t not in [' ', '']]
        vocab.update(temp)
    vocab = sorted(list(vocab))
    return vocab

def tokenizer(doc):
    return str(doc).split(", ")

def get_TFIDF(Genre_Object: Genre, vocab:set[str], genre:str):
    
    if len(Genre_Object['df']) == 0:
        return pd.DataFrame()
    vectorizer = TfidfVectorizer(tokenizer=tokenizer, token_pattern=None, smooth_idf=True, norm=None)
    X = vectorizer.fit_transform(Genre_Object['df']).toarray()

    atfidf = np.mean(X, axis=0)
    res = pd.DataFrame({
        'term': Genre_Object['terms'],
        'score': atfidf
    }).sort_values(by='term')
  
    return dict(zip(res.term, res.score))
    


def get_chi2(Genre_Object: Genre, vocab: set[str], main_dict: DataFrame, genre:str):
    # X needs to be like:
    """
                advice assist attempt  
        Doc 1      1      0      1          
        DOc 2      0      0      1          
    
    y needs to be like:

        [1, 0] for <current genre>, meaning:
            Doc1 is in genre, Doc2 is not   
    """
    if len(Genre_Object['df']) == 0:
        return pd.DataFrame()
    vectorizer = CountVectorizer(tokenizer=tokenizer, token_pattern=None, vocabulary=vocab)
    X = vectorizer.fit_transform(main_dict['Ngrams'])
    
    y = main_dict['Genres'].apply(lambda genre_list: 1 if genre in genre_list else 0)
    
    chi2_scores, p_values = chi2(X, y)
    """
    print(len(vocab))
    print(X.shape)
    print(y.shape)
    print(chi2_scores.shape)
    print(p_values.shape)
    """
    chi2_results = pd.DataFrame({
        'term': vocab,
        'score': chi2_scores,
        'p_value': p_values
    }).sort_values(by='term')
    res = chi2_results[chi2_results['term'].isin(Genre_Object['terms'])].reset_index(drop=True)

    return dict(zip(res.term, res.score))
    
    

    











def aggregate_genres(Genre_Objects: dict[str, Genre], feature: str):
    dfs = []
    for genre, Genre_Object in Genre_Objects.items():
        dfs.append(Genre_Object[feature])
    combined_df = reduce(lambda left, right: pd.merge(left, right, on='term'), dfs)
    return combined_df

def get_normalized_dicts(Genre_Objects: dict[str, Genre], corpusfile):
    atfidf_min_max = [0.0,0.0]
    chi2_min_max = [0.0,0.0]
    for genre, GenreObj in Genre_Objects.items():
        atfidf_min = GenreObj['atfidf']['score'].min()
        atfidf_max = GenreObj['atfidf']['score'].max()
        chi2_min = GenreObj['chi2']['score'].min()
        chi2_max =GenreObj['chi2']['score'].max()
        if atfidf_min_max[0] == 0.0 or atfidf_min_max[0] > atfidf_min:
            atfidf_min_max[0] = atfidf_min
        if atfidf_min_max[1] == 0.0 or atfidf_min_max[1] < atfidf_max:
            atfidf_min_max[1] = atfidf_max
        if chi2_min_max[0] == 0.0 or chi2_min_max[0] > chi2_min:
            chi2_min_max[0] = chi2_min
        if chi2_min_max[1] == 0.0 or chi2_min_max[1] < chi2_max:
            chi2_min_max[1] = chi2_max
    
    normalized_dicts = defaultdict(dict)
    cval_dict = get_cval(corpusfile)
    for genre, GenreObj in Genre_Objects.items():
        
       
        GenreObj['atfidf']['score'] = (GenreObj['atfidf']['score'] - atfidf_min_max[0]) / (atfidf_min_max[1] - atfidf_min_max[0])
        GenreObj['chi2']['score'] = (GenreObj['chi2']['score'] - chi2_min_max[0]) / (chi2_min_max[1] - chi2_min_max[0])


        #GenreObj['atfidf']['score'].apply(lambda x: math.log(x) if x != 0 else x)
        normalized_dicts[genre]['atfidf'] = dict(zip(GenreObj['atfidf'].term, GenreObj['atfidf'].score))
        normalized_dicts[genre]['chi2'] = dict(zip(GenreObj['chi2'].term, GenreObj['chi2'].score))
        normalized_dicts[genre]['cval'] = cval_dict[genre]
        
    return normalized_dicts

def compute_CS_score(normalized_dict, weights: dict[str, float]):

    chi2_dict = normalized_dict['chi2']

    atfidf_dict = normalized_dict['atfidf']
    cval_dict = normalized_dict['cval']
    CS_dict = {}

    for term in chi2_dict.keys():

        CS = chi2_dict[term] * weights['chi2'] + atfidf_dict[term] * weights['atfidf']
        CS_dict[term] = CS
    return CS_dict
