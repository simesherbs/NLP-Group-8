import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from final_helpers import get_chi2, get_TFIDF, aggregate_genres, Genre, get_normalized_dicts, compute_CS_score, get_vocab
from bigramstrigrams import generate_ngram_file
from collections import defaultdict
import sys
from CValue import get_cval


def calc_CS(corpus_file):
    
    df = pd.read_csv(corpus_file)

    df["Genres"] = df["Genres"].apply(
        lambda x: x.split(", ")
    )  # Assuming genres are comma-separated


    unique_genres = set(genre for genre_list in df['Genres'] for genre in genre_list)

    # Convert to a sorted list (optional)
    genres_arr = sorted(unique_genres)

   
    Genre_Objects = {}
    vocab = set()

    # Split ngrams in the column and add tokens to the set
    vocab = get_vocab(df)

    for genre in genres_arr:
        gdf = df[df['Genres'].apply(lambda x: genre in x)]
        terms = set()
        for ngram in gdf['Ngrams'].unique():
            temp = str(ngram).split(", ")
            terms.update(temp)
        terms = (sorted(list(terms)))
        Genre_Objects[genre] = Genre(df=gdf['Ngrams'], genre=genre, terms=terms)

    CS_dict = {}

    cval_dict = get_cval(corpus_file)
    

    for genre in Genre_Objects.keys():
        atfidf_dict = get_TFIDF(Genre_Objects[genre], vocab=vocab, genre=genre)
        chi2_dict = get_chi2(Genre_Objects[genre], vocab, df, genre)
        
        CS_dict[genre] = {}
        for term in atfidf_dict.keys():

            if term in cval_dict[genre].keys():
                
                CS_dict[genre][term] = (atfidf_dict[term], chi2_dict[term], (cval_dict[genre][term]))
            else:
                CS_dict[genre][term] = (atfidf_dict[term], chi2_dict[term], 0.0)
    
    
    return CS_dict
#normalized_dict[genre][feature] = dict[term][score]







