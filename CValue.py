import pandas as pd
from collections import defaultdict, Counter
import ast  # To safely evaluate stringified lists in the CSV
import math
import statistics
from sklearn.preprocessing import MinMaxScaler


def find_nested_terms_and_frequencies_from_csv(csv_file):
    """
    Identifies nested terms and their frequencies from a CSV file with document-level n-grams and genres.

    Args:
        csv_file (str): Path to the CSV file with columns 'unigrams', 'bigrams', 'trigrams', and 'genres'.

    Returns:
        dict: A dictionary where keys are genres, and values are dictionaries mapping terms
              to a tuple of (nested terms, frequency).
    """
    # Load the CSV file
    df = pd.read_csv(csv_file)


    # Initialize the nested terms and frequencies data structure
    genre_term_data = defaultdict(lambda: defaultdict(lambda: {"nested_terms": [], "frequency": 0}))
    genre_term_freq = defaultdict(Counter)

    # Iterate over each row/document in the CSV
    for _, row in df.iterrows():
        # Parse the columns
        
        ngrams = str(row['Ngrams']).split(', ')
        unigrams = []
        bigrams = []
        trigrams = []
        genres = str(row['Genres']).split(', ')
        
        for t in ngrams:
            count = len(t.split(' '))
            if count == 1:
                unigrams.append(t)
            elif count == 2:
                bigrams.append(t)
            else:
                trigrams.append(t)

        # Combine n-grams for this document
        doc_ngrams = unigrams + bigrams + trigrams

        # Count term frequencies for the document
        term_freq = Counter(doc_ngrams)

        # Compare all terms within the document
        for term in doc_ngrams:
            for other_term in doc_ngrams:
                if term != other_term and term in other_term:
                    #print(term, other_term)
                   
                    for genre in genres:
                        
                        genre_term_data[genre][term]["nested_terms"].append(other_term)
                        #print(genre_term_data[genre][term]["nested_terms"])
                        
            # Update term frequency
            for genre in genres:
                genre_term_data[genre][term]["frequency"] += term_freq[term]
    return genre_term_data

def get_nested_freq_sum(nested_terms, main_dict, genre):
    sum = 0
    for term in nested_terms:
        sum += main_dict[genre][term]['frequency']
    return sum
def compute_c_value(candidate_str, genre_freq, nested_terms, main_dict, genre):
    """
    Computes the C-Value for a single term.
    """
    length = len(str(candidate_str).split(' '))
    if len(nested_terms) != 0:
        return math.log2(length + .5)* (genre_freq  - (1/len(nested_terms) * get_nested_freq_sum(nested_terms, main_dict, genre)))
    return (math.log2(length + .5)*genre_freq)   


def get_cval(csv_file):
      # Replace with your CSV file path

    nested_terms_with_freq = find_nested_terms_and_frequencies_from_csv(csv_file)


    c_value_dict = defaultdict(lambda: defaultdict(float))

    for genre, terms in nested_terms_with_freq.items():
        for term, data in terms.items():
            c_value_dict[genre][term] = compute_c_value(term, data['frequency'], data['nested_terms'], nested_terms_with_freq, genre)
            #print(f"  Term: {term}")
            #print(f"    C-Value: {compute_c_value(term, data['frequency'], data['nested_terms'], nested_terms_with_freq, genre)}")
            #print(f"    Nested Terms: {data['nested_terms']}")
        # Print just the first genre for clarity
    cval = pd.DataFrame.from_dict(c_value_dict)
    genres = cval.columns
    cval.rename(columns={'': 'term'}, inplace=True)
    cval.fillna(0, inplace=True)
    scaler = MinMaxScaler()
    
    normalized_cvalues = pd.DataFrame(
        scaler.fit_transform(cval), 
        index=cval.index, 
        columns=cval.columns
    )
    cval2 = cval.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    cval["variance"] = cval.iloc[:, 1:].var(axis=1)
    cval["mean"] = cval.iloc[:, 1:].mean(axis=1)
    cval["variance_to_mean_ratio"] = cval["variance"] / cval["mean"]
    filtered = cval[cval[genres].gt(5.0).any(axis=1)]
    
    cval_dict = normalized_cvalues.to_dict()
    #filtered.sort_values(by='variance_to_mean_ratio', ascending=False).to_csv('C-Value.csv')
    #print(filtered.sort_values(by='variance_to_mean_ratio', ascending=False))
    
    return cval_dict
